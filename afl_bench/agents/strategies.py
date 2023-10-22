from typing import Any, Callable, List, Optional
from pydantic import BaseModel

class Strategy(BaseModel):
    name: str                               # Name of strategy.
    wait_for_full: bool                     # Whether aggregation buffer should wait until full to perform aggregation.
    buffer_size: Optional[int]              # Size of aggregation buffer.
    aggregate: Callable[[List[Any]], Any]   # TODO: determine type of list
    