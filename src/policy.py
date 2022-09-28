import numpy as np

class PolicyEval:
    def __init__(self, actions, action_prob, env, gamma=0.9) -> None:
        self.actions = actions
        self.action_prob = action_prob
        self.env = env
        self.gamma = gamma

    def execute(self) -> int:
        pass

    def get_name(self) -> str:
        pass

class Bellman(PolicyEval):
    def __init__(self, actions, action_prob, env, gamma=0.9) -> None:
        super().__init__(actions, action_prob, env, gamma)

    def execute(self, value, i, j) -> float:
        cur_value = 0
        for action in self.actions:
            [(next_i, next_j), reward] = self.env.step([i, j], action)
            cur_value += self.action_prob * (reward + self.gamma * value[next_i, next_j])

        return cur_value

    def get_name(self) -> str:
        return 'Bellman'

class OptimalBellman(PolicyEval):
    def __init__(self, actions, action_prob, env, gamma=0.9) -> None:
        super().__init__(actions, action_prob, env, gamma)

    def execute(self, value, i, j) -> float:
        cur_value = []
        for action in self.actions:
            [(next_i, next_j), reward] = self.env.step([i, j], action)
            cur_value.append(reward + self.gamma * value[next_i, next_j])

        return np.max(cur_value)

    def get_name(self) -> str:
        return 'Optimal Bellman'