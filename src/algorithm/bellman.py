import numpy as np
from numpy.linalg import inv

from .base import Algorithm

class Bellman(Algorithm):
    def __init__(self, env, gamma) -> None:
        super().__init__(env)
        self.gamma = gamma

    def update_gamma(self, gamma) -> None:
        self.gamma = gamma

    def execute(self) -> np.ndarray:
        pass

    def policy(self) -> np.ndarray:
        optimal = self.execute()
        labels = {}
        n = self.env.get_states_len()
        for i in range(n):
            next_vals = []
            for action in self.env.get_actions(i):
                next_state, _, _ = self.env.step(i, action)
                next_vals.append(optimal[next_state])
            labels[i] = np.where(next_vals == np.max(next_vals))[0]
        return labels

    def get_name(self) -> str:
        pass


class BellmanEquation(Bellman):
    def __init__(self, env, gamma) -> None:
        super().__init__(env, gamma)

    def execute(self) -> np.ndarray:
        actions = self.env.get_actions(0)
        n = self.env.get_states_len()
        I = np.identity(n)
        R = np.zeros((n, 1))
        P = np.zeros((n, n))
        for i in range(n):
            for action in actions:
                [nx, r_a, prob, _] = self.env.step(i, action)
                R[i] += r_a * prob
                P[i][nx] += prob

        value = np.multiply(self.gamma, P)
        value = inv(I - value)
        value = np.dot(value, R)
        return value

    def get_name(self) -> str:
        return 'Bellman Equation'


class IterativeBellman(Bellman):
    def __init__(self, env, gamma) -> None:
        super().__init__(env, gamma)

    def _value_state(self, value, i) -> np.ndarray:
        cur_value = 0
        for action in self.env.get_actions(i):
            [next_i, reward, prob, _] = self.env.step(i, action)
            for nx, r, p in zip(next_i, reward, prob):
                cur_value += p * (r + self.gamma * value[nx])

        return cur_value

    def execute(self, limit=1e-4) -> np.ndarray:
        n = self.env.get_states_len()
        value = np.zeros(n)
        while True:
            new_value = np.zeros(n)
            for i in range(n):
                new_value[i] = self._value_state(value, i)

            if np.max(np.abs(value - new_value)) < limit:
                break
            else:
                value = new_value
        return new_value

    def get_name(self) -> str:
        return 'Iterative Equation'