import numpy as np

from .base import Algorithm


class Iterative(Algorithm):
    def __init__(self, env, gamma, limit=1e-4) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.limit = limit
        self.states = self.env.get_states()
        self.value = np.zeros(self.env.get_states_len())

    def execute(self) -> np.ndarray:
        pass

    def _value_state(self, i) -> np.ndarray:
        pass

    def policy(self, limit=1e-4) -> np.ndarray:
        labels = {}
        for i in self.states:
            next_vals = self._value_state(i)
            new_value = np.max(next_vals)
            labels[i] = np.where(np.abs(next_vals-new_value) < self.limit)[0]
        return labels

    def get_name(self) -> str:
        pass


class ValueIteration(Iterative):
    def __init__(self, env, gamma, limit=1e-4) -> None:
        super().__init__(env, gamma, limit)

    def _value_state(self, i) -> np.ndarray:
        cur_value = []
        for action in self.env.get_actions(i):
            [next_i, reward, _, prob] = self.env.step(i, action)
            cur = 0
            for nx, r, p in zip(next_i, reward, prob):
                cur += p * (r + self.gamma * self.value[nx])
            cur_value.append(cur)

        return cur_value

    def execute(self) -> np.ndarray:
        while True:
            delta = 0.0
            for i in self.states[1:-1]:
                next_vals = self._value_state(i)
                new_value = np.max(next_vals)
                delta = max(delta, np.abs(self.value[i] - new_value))
                self.value[i] = new_value

            if delta < self.limit:
                break

        return self.value

    def get_name(self) -> str:
        return 'Value Iteration'
