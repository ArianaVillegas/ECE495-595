import numpy as np

from src.algorithm.base import Algorithm
from src.algorithm.policy import Policy

class OffPolicy(Algorithm):
    def __init__(self, env, policy, gamma = 1) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.policy = policy

    def get_name(self) -> str:
        return "Off Policy"


class QLearning(OffPolicy):
    def __init__(self, env, policy, gamma = 1) -> None:
        super().__init__(env, policy, gamma)

    def execute(self, n_episodes=100000, alpha=0.5) -> np.ndarray:
        for _ in range(n_episodes):
            state, done = self.env.reset()
            while not done:
                action = super()._get_action(state)
                next_state, reward, done = self.env.step(state, action)

                state_idx = np.where(self.env.states == next_state)[0][0]
                action_idx = np.where(self.env.get_actions(state) == action)[0][0]
                self.q[state_idx][action_idx] += alpha*(reward + self.gamma*np.max(self.q[state_idx]) - self.q[state_idx][action_idx])

                state = next_state
        
        super().execute()

        return self.value

    def get_name(self) -> str:
        return "Q-Learning"