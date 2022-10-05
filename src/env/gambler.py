import numpy as np
import matplotlib.pyplot as plt

from .base import Env

class Gambler(Env):
    def __init__(self, goal, prob_h) -> None:
        self.states = np.array(range(goal+1))
        self.goal = goal
        self.prob_h = prob_h

    def get_name(self) -> str:
        return 'Gambler Problem'

    def get_actions(self, state) -> np.ndarray:
        return np.array(range(min(state, self.goal - state) + 1))

    def get_states(self) -> np.ndarray:
        return self.states[1:-1]

    def step(self, state, action) -> list:
        coin = [1-self.prob_h, self.prob_h]
        next_state = [state, state]
        reward = [0, 0]

        if state%100:
            for i, c in enumerate([-1,1]):
                next_state[i] = state + c*action
                if next_state[i] == 100:
                    reward[i] = 1

        return [next_state, reward, [], coin]

    def plot_value(self, ax, value) -> None:
        ax.set_xticks([1,25, 50, 75, 99])
        ax.axis(ymin=0,ymax=1.05)
        ax.plot(self.states[:-1], value[:-1])
        ax.set_xlabel('Capital')
        ax.set_ylabel('Value estimates')

    def plot_policy(self, ax, policy_map, min_val=True) -> None:
        x_unique = []
        y_unique = []
        y = []
        for key in policy_map:
            opt = list(set(policy_map[key]))
            [x_unique.append(key) for _ in opt]
            [y.append(o) for o in opt]
            # if len(opt) > 1 and 0 in opt:
            #     opt.remove(0)
            if min_val:
                y_unique.append(min(opt))
            else:
                y_unique.append(max(opt))
        ax.set_xticks([1,25, 50, 75, 99]) 
        ax.bar(list(policy_map.keys()), y_unique, align='center', alpha=0.5)
        ax.scatter(x_unique, y, s=0.5)
        ax.set_xlabel('Capital')
        ax.set_ylabel('Final policy (stake)')