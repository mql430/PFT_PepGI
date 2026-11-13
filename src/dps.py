import torch


class DynamicParameterSelection:

    def __init__(self, model, keep_ratio, hard_mask=False):
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        self.keep_ratio = keep_ratio
        self.hard_mask = hard_mask
        self.scores = tuple(torch.zeros_like(p) for p in self.parameters)
        self.masks = ()

    def start_epoch(self, epoch):
        if epoch % 2 == 0:
            self.masks = ()
            for score in self.scores:
                score.zero_()
            return

        scores = torch.cat([score.flatten().cpu() for score in self.scores])
        keep = max(1, int(scores.numel() * self.keep_ratio))
        if self.hard_mask:
            selected = torch.zeros_like(scores, dtype=torch.bool)
            selected[scores.topk(keep, sorted=False).indices] = True
            chunks = selected.split([score.numel() for score in self.scores])
            self.masks = tuple(
                chunk.reshape_as(score).to(score.device)
                for chunk, score in zip(chunks, self.scores)
            )
        else:
            rank = max(1, scores.numel() - keep)
            threshold = scores.kthvalue(rank).values.item()
            self.masks = tuple(score >= threshold for score in self.scores)
        active = sum(mask.count_nonzero().item() for mask in self.masks)
        print(f"DPS Stage II: {active}/{scores.numel()} selected", flush=True)

    @torch.no_grad()
    def before_step(self, optimizer):
        if not self.masks:
            for parameter, score in zip(self.parameters, self.scores):
                if parameter.grad is not None:
                    score.addcmul_(parameter.grad, parameter.grad)
            return

        for parameter, mask in zip(self.parameters, self.masks):
            if parameter.grad is None:
                continue
            parameter.grad.mul_(mask)
            if self.hard_mask:


                state = optimizer.state.get(parameter)
                if state:
                    state["exp_avg"].mul_(mask)
                    state["exp_avg_sq"].mul_(mask)
