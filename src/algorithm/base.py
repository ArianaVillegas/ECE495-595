import numpy as np

from src.algorithm.policy import Policy

class Algorithm:
    def __init__(self, env) -> None:
        self.env = env
        self.timesteps = []
        self.states = self.env.get_states()
        self.value = np.zeros(self.env.get_states_len())
        self.q = [np.zeros(self.env.get_actions_len(state), dtype=float) for state in self.env.states]

    def _get_action(self, state_idx) -> int:
        idx = self._policy.get_action(self.q[state_idx])
        return idx

    def execute(self) -> np.ndarray:
        final_policy = Policy()
        for i in range(self.env.get_states_len()):
            self.value[i] = final_policy.get_value(self.q[i])
        return self.value

    def reset(self) -> np.ndarray:
        pass

    def gen_episode(self, max_steps=30):
        path = []
        self._policy = Policy()
        state_idx, done = self.env.reset()
        action_idx = self._get_action(state_idx)
        t = 0
        while not done and t < max_steps:
            path.append(state_idx)
            state_idx, _, done = self.env.step(state_idx, action_idx)
            action_idx = self._get_action(state_idx)
            t += 1
        path.append(state_idx)
        return path

    def plot_policy(self) -> np.ndarray:
        labels = {}
        final_policy = Policy()
        for state in self.states:
            state_idx = np.where(self.env.states == state)[0][0]
            labels[state] = final_policy.get_max(self.q[state_idx])
        return labels

    def plot_timesteps(self, ax) -> None:
        x = range(len(self.timesteps))
        ax.plot(x, self.timesteps)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episodes')

    def plot_episode_len(self, ax) -> None:
        x = list(set(self.timesteps))
        y = [self.timesteps.count(v) for v in x]
        ax.plot(x, y)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode length')

    def get_name(self) -> str:
        pass