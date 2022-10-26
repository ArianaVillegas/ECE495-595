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
        fig, ax = plt.subplots(1)
        fig.suptitle(f'Env: {self.env.get_name()} \n Algorithm: {algo.get_name()}')
        self.env.plot_policy(ax, labels)
        fig, ax = plt.subplots(1)
        fig.suptitle(f'Env: {self.env.get_name()} \n Algorithm: {algo.get_name()}')
        self.env.plot_path(ax, algo.gen_episode())
        fig, ax = plt.subplots(1)
        fig.suptitle(f'Env: {self.env.get_name()} \n Algorithm: {algo.get_name()}')
        algo.plot_timesteps(ax)
        fig, ax = plt.subplots(1)
        fig.suptitle(f'Env: {self.env.get_name()} \n Algorithm: {algo.get_name()}')
        algo.plot_episode_len(ax)

        fig.tight_layout()