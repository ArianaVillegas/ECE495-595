import numpy as np

class Algorithm:
    def __init__(self, env) -> None:
        self.env = env

    def execute(self) -> np.ndarray:
        pass

    def policy(self) -> np.ndarray:
        pass

    def get_name(self) -> str:
        pass