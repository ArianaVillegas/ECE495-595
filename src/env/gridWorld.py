import numpy as np
from matplotlib.table import Table

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

    def _get_state(self, state):
        x, y = state
        return x * self.world_size[0] + y

    def get_action_idx(self, state, action) -> int:
        for i, cur_action in enumerate(self.get_actions(state)):
            if all(cur_action == action):
                return i

    def step(self, state, action) -> list:
        state = [state//self.world_size[0], state%self.world_size[0]] 
        
        next_state = (np.array(state) + action - [WIND[state[1]], 0]).tolist() #add wind here indexed by x
        x, y = next_state
        if x < 0 or x >= self.world_size[1] or y < 0 or y >= self.world_size[0]:
            next_state = state
        
        if state == self.end_pos:
            done = True
            reward = 0.0
        else:
            done = False
            reward = -1.0
        
        return self._get_state(next_state), reward, done

    def reset(self):
        return self._get_state(self.start_pos), False


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
        ax.get_xticklabels(str(WIND))
        return ax

    def plot_value(self, ax, value) -> None:
        value = np.reshape(value, self.world_size[::-1])
        ax = self._draw_table(ax, np.round(value, decimals=1))

    def plot_policy(self, ax, policy_map) -> None:
        labels = []
        for key in policy_map:
            labels.append(''.join([ACTIONS_FIGS[v] for v in policy_map[key]]))
        ax = self._draw_table(ax, np.reshape(np.array(labels), self.world_size[::-1]))