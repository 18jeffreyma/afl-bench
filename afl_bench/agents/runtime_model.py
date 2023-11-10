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
    def sample_runtime(self) -> float:
        return 0.0


class GaussianRuntime(RuntimeModel):
    def __init__(self, mean=1, variance=1) -> None:
        self.mean = mean
        self.variance = variance

    def sample_runtime(self) -> float:
        return float(np.random.normal(self.mean, np.sqrt(self.variance)))
