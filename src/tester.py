import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from src.algorithm import Algorithm

class Tester:
    def __init__(self, env) -> None:
        self.env = env


    def _gen_labels(self, optimal):
        labels = {}
        n = self.env.get_states_len()
        for i in range(n):
            next_vals = []
            for action in self.env.get_actions():
                next_state, _, _ = self.env.step(i, action)
                next_vals.append(optimal[next_state])
            labels[i] = np.where(next_vals == np.max(next_vals))[0]
        return labels


    def test(self, algo) -> None:
        value = algo.execute()
        labels = algo.plot_policy()

        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(11, 5.5)
        self.env.plot_value(ax1, value)
        self.env.plot_policy(ax2, labels)
        fig.suptitle(f'Env: {self.env.get_name()} \n Algorithm: {algo.get_name()}')

        fig.tight_layout()