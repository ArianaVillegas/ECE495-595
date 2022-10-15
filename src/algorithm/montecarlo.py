import numpy as np
import matplotlib.pyplot as plt

from src.algorithm.iterative import Iterative


class MonteCarlo(Iterative):
    def __init__(self, env, gamma, epsilon=None) -> None:
        super().__init__(env, gamma)
        self.eps = epsilon

    def _get_action(self, state) -> int:
        cur_values = super()._value_state(state)
        max_idx = np.argmax(cur_values)
        actions = self.env.get_actions(state)
        if self.eps:
            prob = np.full(len(actions), self.eps/len(actions))
            prob[max_idx] = 1 - self.eps + self.eps/len(actions)
        else:
            prob = np.zeros(len(actions))
            prob[max_idx] = 1
        return np.random.choice(actions, p=prob)

    def get_episode(self) -> list:
        states = []
        actions = []
        rewards = []

        state = np.random.choice(self.states)
        while True:
            states.append(state)

            action = self._get_action(state)
            actions.append(action)

            [next_states, cur_rewards, _, prob] = self.env.step(state, action)
            rd = np.random.choice(len(prob), p=prob)
            state, reward = next_states[rd], cur_rewards[rd]
            rewards.append(reward)

            if self.env.is_done(state):
                break

        return states, actions, rewards

    def name(self) -> str:
        return 'Monte Carlo'


class FVMonteCarlo(MonteCarlo):
    def __init__(self, env, gamma, epsilon=0.1) -> None:
        super().__init__(env, gamma, epsilon)

    def execute(self, n_episodes=1) -> np.ndarray:
        for _ in range(n_episodes):
            states, actions, rewards = self.get_episode()
            for state, action in zip(states, actions):
                pass

    def name(self) -> str:
        return 'First Visit Monte Carlo'
