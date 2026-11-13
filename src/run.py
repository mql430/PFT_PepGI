import argparse
import itertools
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DGLBACKEND", "pytorch")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from data import SPLIT_SEED, make_loader, read_data, split_indices
from dps import DynamicParameterSelection
from model import Redumixdti


def model_config():
    return SimpleNamespace(
        Drug=SimpleNamespace(
            Node_In_Feat=75, Node_In_Embedding=128, Hidden_Layers=[128, 128, 128],
            Padding=True, GCN_Activation=None, Max_Nodes=365,
        ),
        Protein=SimpleNamespace(
            Embedding_Dim=128, Num_Filters=[128, 128, 128], Kernel_Size=[3, 6, 9],
            Padding=True, Max_Length=1200, After_CNN_Length=1185,
        ),
        MBCA=SimpleNamespace(Hidden_Size=128, Num_Heads=8),
        MLP=SimpleNamespace(In_Dim=128, Binary=2),
    )


def batches(loader, limit):
    return itertools.islice(loader, limit) if limit is not None else iter(loader)


def score_batch(model, batch, device):
    graph, proteins, labels, indices = batch
    logits = model(graph.to(device), proteins.to(device))[-1]
    return logits, labels.to(device, dtype=torch.long), indices


@torch.inference_mode()
def predict(model, loader, device, limit):
    model.eval()
    records = []
    loss_sum = 0.0
    for batch in batches(loader, limit):
        logits, labels, indices = score_batch(model, batch, device)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        probabilities = logits.softmax(dim=1)[:, 1].cpu().tolist()
        records.extend(zip(indices, probabilities, labels.cpu().tolist()))
    predictions = pd.DataFrame(records, columns=["index", "prob", "label"])
    if predictions.empty:
        raise ValueError("No evaluation rows")
    metrics = {"loss": loss_sum / len(predictions)}
    if predictions.label.nunique() == 2:
        metrics["auroc"] = roc_auc_score(predictions.label, predictions.prob)
        metrics["average_precision"] = average_precision_score(predictions.label, predictions.prob)
    return predictions, metrics


def train(args, model, loader, val_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    dps = None
    if args.stage in ("pepti", "pepgi"):
        dps = DynamicParameterSelection(model, args.keep_ratio, args.hard_mask)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_auc = float("-inf")
    best_stop = float("-inf")
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        if dps is not None:
            dps.start_epoch(epoch)
        for batch in batches(loader, args.limit_batches):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, labels, _ = score_batch(model, batch, device)
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if dps is not None:
                dps.before_step(optimizer)
            scaler.step(optimizer)
            scaler.update()

        _, metrics = predict(model, val_loader, device, args.limit_batches)
        auroc = metrics.get("auroc")
        if auroc is None:
            raise ValueError("Validation needs both classes. Increase the batch limit.")
        history.append({"epoch": epoch, **metrics})
        print(json.dumps(history[-1]), flush=True)
        if auroc > best_auc:
            best_auc = auroc
            torch.save(model.state_dict(), args.output / "best.pt")
        stop_score = auroc if args.stop_metric == "auroc" else -metrics["loss"]
        if stop_score > best_stop:
            best_stop = stop_score
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    pd.DataFrame(history).to_csv(args.output / "validation.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the GCN/CNN Pro-DPS model.")
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--stage", choices=["smgi", "pepti", "pepgi", "scratch"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", type=Path, help="Model state dict from the preceding stage")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--keep-ratio", type=float, default=0.6)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--stop-metric", choices=["loss", "auroc"], default="loss")
    parser.add_argument("--hard-mask", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Historical SmGI mixed precision")
    parser.add_argument("--limit-batches", type=int, help="Limit each split for smoke testing")
    parser.add_argument("--all-data", action="store_true",
                        help="Evaluate every row instead of the held-out split")
    parser.add_argument("--test-already-split", action="store_true",
                        help="Use every input row for the five training and validation folds")
    args = parser.parse_args()
    if args.init is None and (args.mode == "evaluate" or args.stage not in ("smgi", "scratch")):
        parser.error("evaluation and fine-tuning require --init")
    if args.batch_size < 1 or args.epochs < 1 or args.workers < 0 or args.patience < 1:
        parser.error("invalid batch size, epoch count, worker count or patience")
    if not 0 < args.keep_ratio <= 1 or args.lr <= 0:
        parser.error("invalid keep ratio or learning rate")
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("--limit-batches must be positive")
    if args.fp16 and (args.stage != "smgi" or args.mode != "train"):
        parser.error("--fp16 applies to SmGI pretraining")
    if args.all_data and args.mode != "evaluate":
        parser.error("--all-data applies only to evaluation")
    if args.test_already_split and args.all_data:
        parser.error("--test-already-split and --all-data cannot be combined")
    return args


def main():
    args = parse_args()
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    torch.manual_seed(SPLIT_SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    config = model_config()
    model = Redumixdti(config).to(device)
    if args.init is not None:
        state = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    frame = read_data(args.data)
    if args.all_data:
        train_ids = val_ids = None
        test_ids = np.arange(len(frame))
    else:
        train_ids, val_ids, test_ids = split_indices(
            frame, args.fold, test_already_split=args.test_already_split
        )
        if args.test_already_split:
            test_ids = val_ids
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )

    if args.mode == "train":
        loader = make_loader(frame, train_ids, config, args.batch_size, args.workers, shuffle=True)
        val_loader = make_loader(frame, val_ids, config, args.batch_size, args.workers)
        train(args, model, loader, val_loader, device)
    else:
        loader = make_loader(frame, test_ids, config, args.batch_size, args.workers)
        predictions, metrics = predict(model, loader, device, args.limit_batches)
        predictions.to_csv(args.output / "test_results.csv", index=False)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
