import numpy as np


class Env:
    def __init__(self) -> None:
        pass

    def step(self) -> list:
        pass

    def get_name(self) -> str:
        pass


class GridWorld(Env):
    def __init__(self, world_size, a_pos, a_prime_pos, b_pos, b_prime_pos) -> None:
        self.world_size = world_size
        self.a_pos = a_pos
        self.a_prime_pos = a_prime_pos
        self.b_pos = b_pos
        self.b_prime_pos = b_prime_pos

    def step(self, state, action):
        if state == self.a_pos:
            return self.a_prime_pos, 10
        if state == self.b_pos:
            return self.b_prime_pos, 5

        next_state = (np.array(state) + action).tolist()
        x, y = next_state
        #if x and y out of world size give negative reward
        if x < 0 or x >= self.world_size or y < 0 or y >= self.world_size:
            reward = -1.0
            next_state = state
        else:
            reward = 0
        return [next_state, reward]

    def get_name(self) -> str:
        return 'Grid World'
