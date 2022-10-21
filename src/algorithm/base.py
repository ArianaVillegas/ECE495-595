import numpy as np

from src.algorithm.policy import Policy

class Algorithm:
    def __init__(self, env) -> None:
        self.env = env
        self.value = np.zeros(self.env.get_states_len())
        self.q = [np.zeros_like(self.env.get_actions(state), dtype=float) for state in self.env.states]

    def _get_action(self, state) -> int:
        actions = self.env.get_actions(state)
        state_idx = np.where(self.env.states == state)[0][0]
        idx = self._policy.get_action(self.q[state_idx])
        return actions[idx]

    def execute(self) -> np.ndarray:
        final_policy = Policy()
        for i in range(self.env.get_states_len()):
            self.value[i] = final_policy.get_value(self.q[i])
        return self.value

    def reset(self) -> np.ndarray:
        pass

    def plot_policy(self) -> np.ndarray:
        labels = {}
        final_policy = Policy()
        for state in self.states:
            state_idx = np.where(self.env.states == state)[0][0]
            labels[state] = final_policy.get_max(self.q[state_idx])
        return labels

    def get_name(self) -> str:
        pass