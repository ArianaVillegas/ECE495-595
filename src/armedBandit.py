import numpy as np

from src.policy import *

class ArmedBandit:
    def __init__(self, n, t, init, policy, mu=1) -> None:
        self.arms = n
        self.steps = t
        self.policy = policy
        self.dist = np.random.normal(mu, 1, n)
        self.best = np.argmax(self.dist)
        self.reward_steps = np.zeros(t)
        self.fraction_steps = np.zeros(t)

        # Accum variables
        self.acc_reward = 0
        self.acc_steps = np.zeros(n)
        self.acc_reward_arms = np.full(n, init, dtype=float)

    def get_name(self) -> str:
        return 'ArmedBandit'

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
        self.cur_action = self._get_action(i)
        self.acc_steps[self.cur_action] += 1
        
        # Calculate reward
        self.cur_reward = np.random.normal(self.dist[self.cur_action], 1)
        self.acc_reward = self.acc_reward + (self.cur_reward - self.acc_reward) / (i+1)
        self.acc_reward_arms[self.cur_action] = self.acc_reward_arms[self.cur_action] + (self.cur_reward - self.acc_reward_arms[self.cur_action]) / (self.acc_steps[self.cur_action])

    def simulate(self) -> None:
        for i in range(self.steps):
            self.play(i)
            self.reward_steps[i] = self.acc_reward
            self.fraction_steps[i] = self.acc_steps[self.best]/(i+1)



class GradientArmedBandit(ArmedBandit):
    def __init__(self, n, t, init, alpha, mu=1, baseline=False) -> None:
        super().__init__(n, t, init, None, mu)
        self.alpha = alpha
        self.H = np.full(n, init, dtype=float)
        self.baseline = baseline

    def get_name(self) -> str:
        return 'GradientArmedBandit'

    def _softmax(self, x) -> np.array:
        self.prob = np.exp(x) / np.sum(np.exp(x), axis=0)

    def _get_action(self, i) -> int:
        a = np.random.choice(np.arange(self.arms), p = self.prob)
        return a

    def play(self, i) -> None:
        self._softmax(self.H)
        super().play(i)
        
        # Preferences
        avg = self.acc_reward if self.baseline else 0
        for arm in range(self.arms):
            self.H[arm] += self.alpha * (self.cur_reward - avg) * ((arm==self.cur_action) - self.prob[arm])
