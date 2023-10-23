
from abc import ABC, abstractmethod

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