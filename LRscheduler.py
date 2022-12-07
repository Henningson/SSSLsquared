import torch

class PolynomialLR:
    def __init__(self, optimizer, total_iters=30, power=0.9, last_epoch = -1):
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

    def step(self):
        self.last_epoch += 1

        self.optimizer.step()
        for g in self._optimizer.param_groups:
            g['lr'] = self._base_lr * pow(1 - self.last_epoch/self._total_iters, self._power)