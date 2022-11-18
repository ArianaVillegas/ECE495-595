import random
import numpy as np
from collections import deque

# Each experience have the following items:
# - state
# - action
# - next_state
# - reward
# - done

class ReplayBuffer:
    def __init__(self, max_length=100000) -> None:
        self.experiences = deque(maxlen=max_length)
    
    def record_experience(self, env, algo) -> None:
        time_step = env.reset()
        state = time_step.observation
        done = time_step.is_last()
        while not done:
            action = algo.get_action([state])
            time_step = env.step(action)
            next_state = time_step.observation
            reward = time_step.reward
            done = time_step.is_last()
            self.experiences.append((state, action, next_state, reward, done))
            state = next_state

    def fill_buffer(self, env, algo, init_collect_steps=100) -> None:
        for i in range(init_collect_steps):
            self.record_experience(env, algo)

    def get_batch_experience(self, batch_size=64) -> list:
        experiences = random.sample(self.experiences, batch_size)
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        for exp in experiences:
            state, action, next_state, reward, done = exp
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return (np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones))