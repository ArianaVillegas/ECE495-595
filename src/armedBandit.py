import numpy as np

from policy import *

class ArmedBandit:
    def __init__(self, n, t, init, policy) -> None:
        self.arms = n
        self.steps = t
        self.policy = policy
        self.dist = np.random.normal(0, 1, n)
        self.best = np.argmax(self.dist)
        self.reward_steps = np.zeros(t)
        self.fraction_steps = np.zeros(t)

        # Accum variables
        self.acc_reward = 0
        self.acc_steps = np.zeros(n)
        self.acc_reward_arms = np.full(n, init, dtype=float)

    def get_dist(self) -> np.array:
        return self.dist

    def get_reward_steps(self) -> np.array:
        return self.reward_steps

    def get_best_fraction(self) -> np.array:
        return self.fraction_steps

    def _get_action(self, i) -> int:
        if isinstance(self.policy, Greedy):
            return self.policy.execute(self.acc_reward_arms)
        if isinstance(self.policy, EGreedy):
            return self.policy.execute(i, self.arms, self.acc_reward_arms)
        if isinstance(self.policy, UCB):
            return self.policy.execute(i, self.acc_reward_arms, self.acc_steps)


    def play(self, i) -> None:
        a = self._get_action(i)
        self.acc_steps[a] += 1
        
        # Calculate reward
        reward = np.random.normal(self.dist[a], 1)
        self.acc_reward = self.acc_reward + (reward - self.acc_reward) / (i+1)
        self.acc_reward_arms[a] = self.acc_reward_arms[a] + (reward - self.acc_reward_arms[a]) / (self.acc_steps[a])

    def simulate(self) -> None:
        for i in range(self.steps):
            self.play(i)
            self.reward_steps[i] = self.acc_reward
            self.fraction_steps[i] = self.acc_steps[self.best]/(i+1)