import numpy as np


class Policy:
    def __init__(self, limit=1e-4) -> None:
        self.limit = limit

    def get_max(self, q) -> list:
        max_val = np.max(q)
        max_idx = np.where(np.abs(q-max_val) < self.limit)[0]
        return max_idx

    def get_action(self, q) -> int:
        max_idx = self.get_max(q)
        max_idx = np.random.choice(max_idx)
        return max_idx

    def get_value(self, q) -> float:
        return np.sum(q)

    def name(self) -> str:
        return 'Policy'


class ESoftPolicy(Policy):
    def __init__(self, eps=0.1, limit=1e-4) -> None:
        super().__init__(limit)
        self.eps = eps

    def get_action(self, q) -> int:
        max_idx = super().get_action(q)
        prob = np.full(len(q), self.eps/len(q))
        prob[max_idx] = 1 - self.eps + self.eps/len(q)
        sel_idx = np.random.choice(len(q), p=prob)
        return sel_idx

    def get_value(self, q) -> float:
        return np.sum(q)*self.eps + (1-self.eps)*np.max(q)

    def name(self) -> str:
        return 'E-Soft Policy'