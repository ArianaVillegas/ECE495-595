import numpy as np
from matplotlib.table import Table
import matplotlib.pyplot as plt

from .base import Env


# N, S, E, W
ACTIONS = np.array([[0, -1],
                    [-1, 0],
                    [0, 1],
                    [1, 0]])
ACTIONS_FIGS = ['←', '↑', '→', '↓']

#column wind from example 6.5 fig 6.10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


class WindyGridWorld(Env):
    def __init__(self, world_size, start_pos, end_pos) -> None:
        self.world_size = world_size
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.actions = ACTIONS
        self.states = np.array(range(world_size[0] * world_size[1]))
        self.wind = WIND

    def get_name(self) -> str:
        return 'Windy Grid World'

    def _get_state_idx(self, state):
        x, y = state
        return x * self.world_size[1] + y

    def _get_state_pair(self, state_idx):
        state = [state_idx//self.world_size[1], state_idx%self.world_size[1]] 
        return state

    def step(self, state_idx, action_idx) -> list:
        state = self._get_state_pair(state_idx) 
        action = self.get_actions(state_idx)[action_idx]
        
        next_state = np.array(state) + np.array(action)
        if next_state[0] < 0 or next_state[0] >= self.world_size[0] or next_state[1] < 0 or next_state[1] >= self.world_size[1]:
            next_state = state
        next_state_x = next_state[0] - WIND[state[1]]
        if next_state_x >= 0:
            next_state[0] = next_state_x
        
        if self._get_state_idx(next_state) == self._get_state_idx(self.end_pos):
            done = True
            reward = 0.0
        else:
            done = False
            reward = -1.0
        
        return self._get_state_idx(next_state), reward, done

    def reset(self):
        return self._get_state_idx(self.start_pos), False


    def _draw_table(self, ax, image):
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = image.shape
        width, height = 1.0 / ncols, 1.0 / nrows
        
        # Add cells
        for (i, j), val in np.ndenumerate(image):
            tb.add_cell(i, j, width, height, text=val,
                        loc='center', facecolor='white')

        for (i,j), val in np.ndenumerate(image):
            tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                        edgecolor='none', facecolor='none')
            tb.add_cell(-1, j, width, height / 2, text=j + 1, loc='center',
                        edgecolor='none', facecolor='none')

        ax.add_table(tb)
        ax.set_xticks(WIND)
        
        return ax

    def plot_value(self, ax, value) -> None:
        value = np.reshape(value, self.world_size)
        ax = self._draw_table(ax, np.round(value, decimals=1))

    def plot_policy(self, ax, policy_map) -> None:
        labels = []
        for key in policy_map:
            labels.append(''.join([ACTIONS_FIGS[v] for v in policy_map[key]]))
        ax = self._draw_table(ax, np.reshape(np.array(labels), self.world_size))

    def plot_path(self, ax, path) -> None:
        for i in range(0, len(path)):
            path[i] = self._get_state_pair(path[i])

        plt.yticks(np.linspace(self.world_size[0] - 0.5, -0.5, self.world_size[0] + 1))
        plt.xticks(np.linspace(-0.5, self.world_size[1] - 0.5, self.world_size[1] + 1))
        
        ax.set_xlim((-0.5, self.world_size[1] - 0.5))
        ax.set_ylim((self.world_size[0] - 0.5, -0.5))
        ax.grid(color='k')
        for i in range(1, len(path)):
            state = path[i-1]
            state_prime = path[i]
            s = np.stack((state, state_prime))
            ax.scatter(s[:,1], s[:,0], color='k')
            ax.plot(s[:,1], s[:,0],'-', color='k', linewidth=3)
        