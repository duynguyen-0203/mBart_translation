from abc import ABC
import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class ModelLoss(Loss):
    def __init__(self, criterion, model, optimizer, scheduler, max_grad_norm):
        self._criterion = criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, logits: torch.tensor, targets: torch.tensor):
        logits = logits.view(logits.size(0) * logits.size(1), -1)
        targets = targets.view(-1)
        loss = self._criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()

        return loss.item()
