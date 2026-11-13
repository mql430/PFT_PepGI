from functools import partial

import numpy as np
import pandas as pd
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import graph_collate_func, integer_label_protein


SPLIT_SEED = 42


def read_data(path):
    frame = pd.read_csv(path, usecols=["SMILES", "Protein", "Y"])
    if frame.isna().any().any() or not frame.Y.isin([0, 1]).all():
        raise ValueError(f"{path} must contain SMILES, Protein and binary Y without missing values")
    return frame


def split_indices(frame, fold, test_already_split=False):
    train_val = np.arange(len(frame))
    test = None
    if not test_already_split:
        train_val, test = train_test_split(
            train_val, test_size=0.2, random_state=SPLIT_SEED
        )
    folds = KFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    train, val = list(folds.split(train_val))[fold]
    return train_val[train], train_val[val], test


class DTIDataset(Dataset):
    def __init__(self, frame, indices, config):
        self.indices = indices
        self.smiles = frame.SMILES.to_numpy()
        self.proteins = frame.Protein.to_numpy()
        self.labels = frame.Y.to_numpy()
        self.max_drug_nodes = config.Drug.Max_Nodes
        self.max_protein_length = config.Protein.Max_Length
        self.graph = partial(
            smiles_to_bigraph,
            add_self_loop=True,
            node_featurizer=CanonicalAtomFeaturizer(),
            edge_featurizer=CanonicalBondFeaturizer(self_loop=True),
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        graph = self.graph(smiles=self.smiles[index])
        if graph is None or graph.num_nodes() > self.max_drug_nodes:
            raise ValueError(f"Invalid or oversized molecule at CSV row {index}")
        features = graph.ndata.pop("h")
        atom_count, channels = features.shape
        padding = self.max_drug_nodes - atom_count
        graph.ndata["h"] = torch.cat((features, features.new_zeros((atom_count, 1))), dim=1)
        if padding:
            virtual = features.new_zeros((padding, channels + 1))
            virtual[:, -1] = 1
            graph.add_nodes(padding, {"h": virtual})
        graph = graph.add_self_loop()
        protein = integer_label_protein(self.proteins[index], self.max_protein_length)
        return graph, protein, self.labels[index], index


def make_loader(frame, indices, config, batch_size, workers, shuffle=False):
    return DataLoader(
        DTIDataset(frame, indices, config),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=graph_collate_func,
        num_workers=workers,
        prefetch_factor=2 if workers else None,
        persistent_workers=bool(workers),
    )
