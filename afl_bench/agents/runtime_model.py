
from abc import ABC, abstractmethod

class RuntimeModel(ABC):
    @abstractmethod
    def sample_runtime(self) -> float:
        """
        Sample model runtime from given model in milliseconds.
        """
        pass