import numpy as np

from src.algorithm.base import Algorithm

class OnPolicy(Algorithm):
    def __init__(self, env, policy, gamma = 1) -> None:
        super().__init__(env)
        self.gamma = gamma
        self._policy = policy

    def get_name(self) -> str:
        return "On Policy"


class SARSA(OnPolicy):
    def __init__(self, env, policy, gamma = 1) -> None:
        super().__init__(env, policy, gamma)

    def execute(self, n_episodes=500, alpha=0.5, steps=1500) -> np.ndarray:
        for _ in range(n_episodes):
            state, done = self.env.reset()
            action = super()._get_action(state)
            t = 0
            while not done and t < steps:
                next_state, reward, done = self.env.step(state, action)
                next_action = super()._get_action(next_state)

                state_idx = self.env.get_state_idx(state)
                action_idx = self.env.get_action_idx(state, action)
                next_state_idx = self.env.get_state_idx(next_state) 
                next_action_idx = self.env.get_action_idx(next_state, next_action) 
                self.q[state_idx][action_idx] += alpha*(reward + self.gamma*self.q[next_state_idx][next_action_idx] - self.q[state_idx][action_idx])

                state, action = next_state, next_action
                t += 1

        return super().execute()

    def get_name(self) -> str:
        return "SARSA"


class SARSALambda(OnPolicy):
    def __init__(self, env, policy, gamma = 1) -> None:
        super().__init__(env, policy, gamma)

    def execute(self, n_episodes=100000, alpha=0.5, lam=0.9, trace_type='accum') -> np.ndarray:
        for _ in range(n_episodes):
            e = [np.zeros_like(self.env.get_actions(state), dtype=float) for state in self.env.states]
            state, done = self.env.reset()
            action = super()._get_action(state)
            while not done:
                next_state, reward, done = self.env.step(state, action)
                next_action = super()._get_action(next_state)

                state_idx = np.where(self.env.states == state)[0][0]
                action_idx = np.where(self.env.get_actions(state) == action)[0][0]
                next_state_idx = np.where(self.env.states == next_state)[0][0]
                next_action_idx = np.where(self.env.get_actions(state) == next_action)[0][0]

                delta = reward + self.gamma*self.q[next_state_idx][next_action_idx] - self.q[state_idx][action_idx]
                if trace_type == 'accum':
                    e[state_idx][action_idx] += 1
                elif trace_type == 'dutch':
                    e[state_idx][action_idx] = (1 - alpha) * e[state_idx][action_idx] + 1
                elif trace_type == 'replace':
                    e[state_idx][action_idx] = 1
                else:
                    raise Exception(f"Trace type {trace_type} not defined")

                for state in self.env.states:
                    for action in self.env.get_actions(state):
                        state_idx = np.where(self.env.states == state)[0][0]
                        action_idx = np.where(self.env.get_actions(state) == action)[0][0]
                        self.q[state_idx][action_idx] += alpha*delta*e[state_idx][action_idx]
                        e[state_idx][action_idx] = self.gamma*lam*e[state_idx][action_idx]

                state, action = next_state, next_action
        
        super().execute()

        return self.value

    def get_name(self) -> str:
        return "SARSA(λ)"
