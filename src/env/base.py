import numpy as np

class Env:
    def __init__(self) -> None:
        self.actions = np.array([])
        self.states = np.array([])
        self.reward = np.array([])

    def get_name(self) -> str:
        pass

    def get_actions(self, state_idx) -> np.ndarray:
        return self.actions

    def get_actions_len(self, state_idx) -> np.ndarray:
        return self.get_actions(state_idx).shape[0]

    def get_action_idx(self, state, action) -> int:
        for i, cur_action in enumerate(self.get_actions(state)):
            if cur_action == action:
                return i

    def get_states(self) -> np.ndarray:
        return self.states

    def get_states_len(self) -> np.ndarray:
        return self.states.shape[0]

    def get_state_idx(self, state) -> int:
        for i, cur_state in enumerate(self.get_states()):
            if cur_state == state:
                return i

    def is_done(self, state) -> bool:
        return False

    def step(self) -> list:
        pass

    def plot_value(self, ax, value) -> None:
        pass

    def plot_policy(self, ax, policy) -> None:
        pass

