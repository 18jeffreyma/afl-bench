from threading import Lock, Condition
import time
from typing import List

from afl_bench.types import ClientUpdate


class Buffer:
    def __init__(self, wait_for_full=True, n=None, ms_to_wait=None) -> None:
        """
        Initializes a thread safe buffer which receives updates in thread safe manner
        and retrieves n-length windows of items.

        Args:
            wait_for_full (bool, optional): whether we should wait for all n or
                just return current state. Defaults to True.
            n (int, optional): number of items that buffer is considered full. Defaults to None.
            ms_to_wait (int, optional): number of milliseconds to wait for buffer to be full.
                Defaults to None.
        """
        assert wait_for_full != (
            ms_to_wait is None
        ), "Must specify one of wait_for_full or ms_to_wait."
        assert not (
            wait_for_full and n is None
        ), "Must specify length if waiting for full buffer."

        self.wait_for_full = wait_for_full
        self.n = n
        self.ms_to_wait = ms_to_wait

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
                # Wait for the buffer to be full
                while len(self.buffer) < self.n:
                    self.full_cv.wait()
            else:
                # Otherwise wait a number of ms for the buffer.
                time.sleep(self.ms_to_wait / 1000)

            # Slice out first length elements (or all elements if buffer is not full)
            slice_length = min(self.n, len(self.buffer))

            relevant_slice = self.buffer[:slice_length]
            self.buffer = self.buffer[slice_length:]

            return relevant_slice
