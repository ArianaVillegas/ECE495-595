import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
from numpy.linalg import inv

from src.policy import *

class Tester:
    def __init__(self, world_size, actions, actions_prob, action_figs, env) -> None:
        self.world_size = world_size
        self.actions = actions
        self.actions_prob = actions_prob
        self.action_figs = action_figs
        self.env = env

    def set_world_size(self, world_size) -> None:
        self.world_size = world_size

    def set_actions(self, actions) -> None:
        self.actions = actions

    def set_actions_prob(self, actions_prob) -> None:
        self.actions_prob = actions_prob

    def _draw_table(self, ax, image):
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = image.shape
        width, height = 1.0 / ncols, 1.0 / nrows
        
        # Add cells
        for (i, j), val in np.ndenumerate(image):
            tb.add_cell(i, j, width, height, text=val,
                        loc='center', facecolor='white')

        for i in range(len(image)):
            tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                        edgecolor='none', facecolor='none')
            tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                        edgecolor='none', facecolor='none')

        ax.add_table(tb)

        return ax

    
    def _iterative_test(self, policy, limit= 1e-4):
        value = np.zeros((self.world_size, self.world_size))
        while True:
            # keep iteration until convergence
            new_value = np.zeros((self.world_size, self.world_size))
            for i in range(self.world_size):
                for j in range(self.world_size):
                    new_value[i][j] = policy.execute(value, i, j)

            if np.sum(np.abs(value - new_value)) < limit:
                break
            else:
                value = new_value
        return new_value

    
    def _equation_test(self, gamma):
        n = self.world_size * self.world_size
        I = np.identity(n)
        R = np.zeros((n, 1))
        P = np.zeros((n, n))
        for i in range(self.world_size):
            for j in range(self.world_size):
                for action in self.actions:
                    [[nx, ny], r_a] = self.env.step([i, j], action)
                    R[i*self.world_size+j] += r_a * self.actions_prob
                    P[i*self.world_size+j][nx*self.world_size+ny] += self.actions_prob

        value = np.multiply(gamma, P)
        value = inv(I - value)
        value = np.dot(value, R)
        value = np.reshape(value, (self.world_size, self.world_size))
        return value


    def _gen_labels(self, optimal):
        labels = [[None for _ in range(self.world_size)] for _ in range(self.world_size)]
        for (i, j), _ in np.ndenumerate(optimal):
            next_vals = []
            for action in self.actions:
                next_state, _ = self.env.step([i, j], action)
                next_vals.append(optimal[next_state[0], next_state[1]])
            labels[i][j] = np.where(next_vals == np.max(next_vals))[0]

        return labels


    def test_grid_world(self, iterative=False, Policy=OptimalBellman, gamma=1) -> None:
        if iterative:
            policy = Policy(self.actions, self.actions_prob, self.env, gamma)
            value = self._iterative_test(policy)
        else:
            value = self._equation_test(gamma)
        labels = self._gen_labels(value)
        for (i, j), val in np.ndenumerate(labels):
            labels[i][j] = ''.join([self.action_figs[v] for v in val])

        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 5.5)
        ax1 = self._draw_table(ax1, np.round(value, decimals=1))
        ax2 = self._draw_table(ax2, np.array(labels))
        if iterative:
            fig.suptitle(f'Env: {self.env.get_name()}\n Policy: {policy.get_name()}')
        else:
            fig.suptitle(f'Env: {self.env.get_name()}')

        fig.tight_layout()