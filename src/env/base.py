import numpy as np

class Env:
    def __init__(self) -> None:
        self.actions = np.array([])
        self.states = np.array([])
        self.reward = np.array([])

    def get_name(self) -> str:
        pass

    def get_actions(self, state) -> np.ndarray:
        return self.actions

    def get_actions_len(self, state) -> np.ndarray:
        return self.get_actions(state).shape[0]

    def get_states(self) -> np.ndarray:
        return self.states

    def get_states_len(self) -> np.ndarray:
        return self.states.shape[0]

    def is_done(self, state) -> bool:
        return False

    def step(self) -> list:
        pass

    def plot_value(self, ax, value) -> None:
        pass

    def plot_policy(self, ax, policy) -> None:
        pass

