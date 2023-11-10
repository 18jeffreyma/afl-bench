import unittest
from threading import Thread

import timeout_decorator

from afl_bench.agents.buffer import Buffer


class TestBuffer(unittest.TestCase):
    def test_no_wait(self):
        buffer = Buffer(wait_for_full=False, n=2)
        buffer.add(1)
        buffer.add(2)
        buffer.add(3)

        # Buffer result should get the first two elements on first
        # call, then the third element on the second call.
        self.assertEqual(buffer.get_items(), [1, 2])
        self.assertEqual(buffer.get_items(), [3])

    @timeout_decorator.timeout(1)
    def test_with_wait(self):
        # Testing ability to add to buffer from another thread while waiting.
        buffer = Buffer(wait_for_full=True, n=1)

        result = [None]

        def set_result(result):
            result[0] = buffer.get_items()

        thread = Thread(target=set_result, args=(result,), daemon=True)
        thread.start()

        # Background thread should be waiting right now for buffer to be full.
        buffer.add(1)
        thread.join()

        # Background thread should have set result to [1].
        self.assertEqual(result[0], [1])


if __name__ == "__main__":
    unittest.main()
