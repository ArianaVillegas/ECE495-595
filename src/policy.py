import numpy as np

class Policy:
    def __init__(self) -> None:
        pass

    def execute(self) -> int:
        pass

    def get_name(self) -> str:
        pass

class Greedy(Policy):
    def execute(self, acc_reward_arms) -> int:
        a = np.argmax(acc_reward_arms)
        return a
    
    def get_name(self) -> str:
        return 'Greedy'

class EGreedy(Policy):
    def __init__(self, epsilon = 0.1) -> None:
        super().__init__()
        self.epsilon = epsilon

    def execute(self, i, arms, acc_reward_arms) -> int:
        rd = np.random.rand()
        if i == 0 or rd < self.epsilon:
            a = np.random.choice(arms)
        else:
            a = np.argmax(acc_reward_arms)
        return a

    def get_name(self) -> str:
        return 'EGreedy'

class UCB(Policy):
    def __init__(self, confidence = 2) -> None:
        super().__init__()
        self.confidence = confidence
    
    def execute(self, i, acc_reward_arms, acc_steps) -> int:
        At = []
        for x, y in zip(acc_reward_arms, acc_steps):
            if i and y:
                At.append(x + self.confidence*np.sqrt(np.log(i) / y))
            else:
                At.append(x + np.iinfo(np.int32).max)
        a = np.argmax(np.array(At))
        return a
    
    def get_name(self) -> str:
        return 'Upper Confidence Bounds'