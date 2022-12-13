import torch

class PolynomialLR:
    def __init__(self, optimizer, total_iters=30, power=0.9, last_epoch = 0):
        self._optimizer = optimizer
        self._total_iters = total_iters
        self._power = power
        self._last_epoch = last_epoch
        self._base_lr = self._optimizer.param_groups[0]['lr']

    def load_state_dict(self, state_dict) -> None:
        self._total_iters = state_dict['total_iters']
        self._power = state_dict['power']
        self._last_epoch = state_dict['last_epoch']

    def state_dict(self) -> dict:
        return {'total_iters': self._total_iters, 'power': self._power ,'last_epoch': self._last_epoch}

    def get_current_lr(self):
        return self._base_lr * pow(1 - self._last_epoch/self._total_iters, self._power)

    def update_lr(self):
        for g in self._optimizer.param_groups:
            g['lr'] = self._base_lr * pow(1 - self._last_epoch/self._total_iters, self._power)

    def step(self):
        self._optimizer.step()
        self._last_epoch += 1

    def zero_grad(self):
        self._optimizer.zero_grad()