import numpy as np
from matplotlib.table import Table

from .base import Env


# N, S, E, W
ACTIONS = np.array([[0, -1],
                    [-1, 0],
                    [0, 1],
                    [1, 0]])
ACTIONS_FIGS = ['←', '↑', '→', '↓']


class GridWorld(Env):
    def __init__(self, world_size, a_pos, a_prime_pos, b_pos, b_prime_pos, action_prob) -> None:
        self.world_size = world_size
        self.a_pos = a_pos
        self.a_prime_pos = a_prime_pos
        self.b_pos = b_pos
        self.b_prime_pos = b_prime_pos
        self.action_prob = action_prob
        self.actions = ACTIONS
        self.states = np.array(range(world_size * world_size))

    def get_name(self) -> str:
        return 'Grid World'

    def _get_state(self, state):
        x, y = state
        return x * self.world_size + y

    def step(self, state, action) -> list:
        state = [state//self.world_size, state%self.world_size]
        if state == self.a_pos:
            reward = 10
            next_state = self.a_prime_pos
        elif state == self.b_pos:
            reward = 5
            next_state = self.b_prime_pos
        else:
            next_state = (np.array(state) + action).tolist()
            x, y = next_state
            if x < 0 or x >= self.world_size or y < 0 or y >= self.world_size:
                reward = -1.0
                next_state = state
            else:
                reward = 0
        
        return [[self._get_state(next_state)], [reward], [self.action_prob], [1]]


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

    def plot_value(self, ax, value) -> None:
        value = np.reshape(value, (self.world_size, self.world_size))
        ax = self._draw_table(ax, np.round(value, decimals=1))

    def plot_policy(self, ax, policy_map) -> None:
        labels = []
        for key in policy_map:
            labels.append(''.join([ACTIONS_FIGS[v] for v in policy_map[key]]))
        ax = self._draw_table(ax, np.reshape(np.array(labels), (self.world_size, self.world_size)))