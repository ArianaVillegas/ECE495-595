import numpy as np

from src.algorithm.base import Algorithm
from src.algorithm.policy import Policy

class OffPolicy(Algorithm):
    def __init__(self, env, policy, gamma = 1) -> None:
        super().__init__(env)
        self.gamma = gamma
        self._policy = policy

    def get_name(self) -> str:
        return "Off Policy"


class QLearning(OffPolicy):
    def __init__(self, env, policy, gamma = 1) -> None:
        super().__init__(env, policy, gamma)

    def execute(self, n_episodes=200, alpha=0.5, steps=3000) -> np.ndarray:
        self.timesteps = []
        for i in range(n_episodes):
            state_idx, done = self.env.reset()
            t = 0
            while not done and t < steps:
                action_idx = super()._get_action(state_idx)
                next_state_idx, reward, done = self.env.step(state_idx, action_idx)

                self.q[state_idx][action_idx] += alpha*(reward + self.gamma*np.max(self.q[next_state_idx]) - self.q[state_idx][action_idx])

                state_idx = next_state_idx
                self.timesteps.append(i)
                t += 1
        
        super().execute()

        return self.value

    def get_name(self) -> str:
        return "Q-Learning"