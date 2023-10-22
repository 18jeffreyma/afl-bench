

from threading import Lock, Condition
from typing import List, Tuple

from afl_bench.types import ClientUpdate

class Buffer:
    def __init__(self, wait_for_full=True, n=None) -> None:
        """
        Initializes a thread safe buffer which receives updates in thread safe manner
        and retrieves n-length windows of items.

        Args:
            wait_for_full (bool, optional): whether we should wait for all n or 
                just return current state. Defaults to True.
            n (int, optional): number of items to return. Defaults to None.
        """
        assert wait_for_full or n is not None, "Must specify length if not waiting for full buffer."
 
        self.wait_for_full = wait_for_full
        self.n = n
 
        self.buffer = []

        self.mutex = Lock()
        self.full_cv = Condition(self.mutex)
    
    def add(self, item: ClientUpdate):
        """
        Add an item to the buffer, signalling if adding the item makes the
        buffer full.

        Args:
            item: item to be added to the buffer.
        """
        with self.mutex:
            self.buffer.append(item)
            if len(self.buffer) == self.n:
                self.full_cv.notify_all()
        
    def get_items(self) -> List[ClientUpdate]:
        """
        Get relevant buffer items given length of buffer requested.

        Returns:
            list: relevant buffer items given length of buffer requested and
            whether to wait for buffer to be full.
        """
        with self.mutex:
            if self.wait_for_full:
                while (len(self.buffer) < self.n):
                    self.full_cv.wait()
            
            # Slice out first length elements (or all elements if buffer is not full)
            slice_length = min(self.n, len(self.buffer))
                
            relevant_slice = self.buffer[:slice_length]
            self.buffer = self.buffer[slice_length:]
            
            return relevant_slice
      