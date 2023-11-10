from abc import ABC, abstractmethod

import numpy as np


class RuntimeModel(ABC):
    @abstractmethod
    def sample_runtime(self) -> float:
        """
        Sample model runtime from given model in milliseconds.
        """
        pass


class InstantRuntime(RuntimeModel):
    def __init__(self, delay) -> None:
        self.delay = delay

    def sample_runtime(self) -> float:
        return self.delay


class GaussianRuntime(RuntimeModel):
    def __init__(self, mean, variance) -> None:
        self.mean = mean
        self.variance = variance

    def sample_runtime(self) -> float:
        return max(float(np.random.normal(self.mean, np.sqrt(self.variance))), 0.0)
