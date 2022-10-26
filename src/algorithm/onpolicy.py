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

    def execute(self, n_episodes=200, alpha=0.5, steps=3000) -> np.ndarray:
        self.timesteps = []
        for i in range(n_episodes):
            state_idx, done = self.env.reset()
            action_idx = super()._get_action(state_idx)
            t = 0
            while not done and t < steps:
                next_state_idx, reward, done = self.env.step(state_idx, action_idx)
                next_action_idx = super()._get_action(next_state_idx)

                self.q[state_idx][action_idx] += alpha*(reward + self.gamma*self.q[next_state_idx][next_action_idx] - self.q[state_idx][action_idx])

                state_idx, action_idx = next_state_idx, next_action_idx
                self.timesteps.append(i)
                t += 1

        return super().execute()

    def get_name(self) -> str:
        return "SARSA"


class SARSALambda(OnPolicy):
    def __init__(self, env, policy, gamma = 1) -> None:
        super().__init__(env, policy, gamma)

    def execute(self, n_episodes=200, alpha=0.5, steps=3000, lam=0.9, trace_type='accum') -> np.ndarray:
        self.timesteps = []
        for i in range(n_episodes):
            e = [np.zeros(self.env.get_actions_len(state_idx), dtype=float) for state_idx in range(self.env.get_states_len())]
            state_idx, done = self.env.reset()
            action_idx = super()._get_action(state_idx)
            t = 0
            while not done and t < steps:
                next_state_idx, reward, done = self.env.step(state_idx, action_idx)
                next_action_idx = super()._get_action(next_state_idx)

                delta = reward + self.gamma*self.q[next_state_idx][next_action_idx] - self.q[state_idx][action_idx]
                if trace_type == 'accum':
                    e[state_idx][action_idx] += 1
                elif trace_type == 'dutch':
                    e[state_idx][action_idx] = (1 - alpha) * e[state_idx][action_idx] + 1
                elif trace_type == 'replace':
                    e[state_idx][action_idx] = 1
                else:
                    raise Exception(f"Trace type {trace_type} not defined")

                for state_idx in range(self.env.get_states_len()):
                    for action_idx in range(self.env.get_actions_len(state_idx)):
                        self.q[state_idx][action_idx] += alpha*delta*e[state_idx][action_idx]
                        e[state_idx][action_idx] = self.gamma*lam*e[state_idx][action_idx]

                state_idx, action_idx = next_state_idx, next_action_idx
                self.timesteps.append(i)
                t += 1
        
        super().execute()

        return self.value

    def get_name(self) -> str:
        return "SARSA(λ)"
