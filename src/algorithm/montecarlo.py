import numpy as np
import matplotlib.pyplot as plt

from src.algorithm.iterative import Iterative
from src.algorithm.policy import Policy


class MonteCarlo(Iterative):
    def __init__(self, env, gamma, policy) -> None:
        super().__init__(env, gamma)
        self._policy = policy
        self.q = [np.zeros_like(self.env.get_actions(state), dtype=float) for state in self.env.states]
        self.qcnt = [np.zeros_like(self.env.get_actions(state)) for state in self.env.states]

    def _get_action(self, state) -> int:
        actions = self.env.get_actions(state)
        state_idx = np.where(self.env.states == state)[0][0]
        idx = self._policy.get_action(self.q[state_idx])
        return actions[idx]

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

    def policy(self) -> np.ndarray:
        labels = {}
        final_policy = Policy()
        for state in self.states:
            state_idx = np.where(self.env.states == state)[0][0]
            labels[state] = final_policy.get_max(self.q[state_idx])
        return labels

    def name(self) -> str:
        return 'Monte Carlo'


class FVMonteCarlo(MonteCarlo):
    def __init__(self, env, gamma, epsilon=0.1) -> None:
        super().__init__(env, gamma, epsilon)

    def execute(self, n_episodes=500000) -> np.ndarray:
        for _ in range(n_episodes):
            states, actions, rewards = self.get_episode()
            reward = 0
            # print(states)
            for i in range(len(states))[::-1]:
                state = states[i] 
                action = actions[i]
                reward += rewards[i]
                if (state, action) in zip(states[:i], actions[:i]):
                    continue
                state_idx = np.where(self.env.states == state)[0][0]
                action_idx = np.where(self.env.get_actions(state) == action)[0][0]
                self.qcnt[state_idx][action_idx] += 1
                self.q[state_idx][action_idx] += (reward - self.q[state_idx][action_idx])/self.qcnt[state_idx][action_idx]
            # [print(qi) for qi in self.q]

        for i in range(self.env.get_states_len()):
            self.value[i] = self._policy.get_value(self.q[i])
        
        # [print(qi) for qi in self.q]
        
        return self.value

    def name(self) -> str:
        return 'First Visit Monte Carlo'
