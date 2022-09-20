import matplotlib.pyplot as plt
import numpy as np

from src.armedBandit import *
from src.policy import *

class Tester:
    def __init__(self, n, t, init=0, mu=0) -> None:
        self.n = n
        self.t = t
        self.init = init
        self.mu = mu

    def set_n(self, n) -> None:
        self.n = n

    def set_t(self, t) -> None:
        self.t = t

    def set_init(self, init) -> None:
        self.init = init

    def set_mu(self, mu) -> None:
        self.mu = mu

    def _get_bandit(self, Bandit, Policy, scenario):
        if issubclass(Bandit, GradientArmedBandit) and Bandit != ArmedBandit:
            bandit = Bandit(self.n, self.t, self.init, scenario[0], mu=self.mu, baseline=scenario[1])
            # scenario[1] = 'baseline' if scenario[1] else 'no baseline'
            return None, bandit
        else:
            policy = Policy(scenario)
            bandit = Bandit(self.n, self.t, self.init, policy, mu=self.mu)
            return policy, bandit


    def testBandit(self, Bandit, Policy, scenarios, steps=1000) -> None:
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10.5, 5.5)
        
        for scenario in scenarios:
            r = np.zeros(self.t)
            p = np.zeros(self.t)
            for step in range(steps):
                policy, bandit = self._get_bandit(Bandit, Policy, scenario)
                bandit.simulate()
                r += (bandit.get_reward_steps() - r) / (step+1)
                p += (bandit.get_best_fraction() - p) / (step+1)
            
            ax1.plot(range(self.t), r, label=[scenario[0], 'baseline' if scenario[1] else 'no baseline'] if isinstance(scenario, list) else scenario)
            ax2.plot(range(self.t), p, label=[scenario[0], 'baseline' if scenario[1] else 'no baseline'] if isinstance(scenario, list) else scenario)

        ax1.legend()
        ax1.set(xlabel='Iteration', ylabel='Average Reward')

        ax2.legend()
        ax2.set(xlabel='Iteration', ylabel='Fraction of optimal action')

        title = bandit.get_name() + (' and {} Policy'.format(policy.get_name()) if Policy else '') + ' and Init {}'.format(self.init)
        fig.suptitle(title)
        fig.tight_layout()