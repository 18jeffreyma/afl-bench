import logging
from abc import abstractmethod
from itertools import count
from threading import Condition, Lock, Thread
from typing import Optional, Tuple

from torch.utils.data import DataLoader

import wandb
from afl_bench.agents.buffer import Buffer
from afl_bench.agents.clients.simple import _test
from afl_bench.agents.common import get_parameters, set_parameters
from afl_bench.agents.strategies import ModelParams, Strategy

logger = logging.getLogger(__name__)


class ServerInterface:
    @abstractmethod
    def get_current_model(
        self, prev_version: Optional[int] = None
    ) -> Tuple[ModelParams, int]:
        """
        Get the current global model parameters.
        """

    @abstractmethod
    def broadcast_updated_model(
        self, old_params: ModelParams, new_params: ModelParams, version_number: int
    ):
        """
        Upon completion of local training, the client calls this method to
        send the updated model to the server. Note that old_params is sent
        since the client may have trained on an outdated model.

        Args:
            old_params: model parameters prior to local training (i.e. the old global model)
            new_params: model parameters after training (i.e. the new global model)
            version_number: the version number of the old global model used.
        """


class Server(ServerInterface):
    def __init__(
        self,
        initial_model,
        strategy: Strategy,
        num_aggregations: int,
        test_dataloader: DataLoader,
        device="cuda",
    ):
        self.model = initial_model
        self.version_number = 0

        self.strategy = strategy
        self.num_aggregations = num_aggregations

        self.buffer = Buffer(
            wait_for_full=strategy.wait_for_full,
            n=strategy.buffer_size,
            ms_to_wait=strategy.ms_to_wait,
        )

        self.model_mutex = Lock()
        self.model_cv = Condition(self.model_mutex)

        self.is_running = False
        self.thread = None

        self.test_dataloader = test_dataloader
        self.device = device

    def get_current_model(
        self, prev_version: Optional[int] = None
    ) -> Tuple[ModelParams, int]:
        with self.model_mutex:
            while (prev_version is not None) and (prev_version == self.version_number):
                # Wait until the model has been updated before letting agent pull.
                logger.info(
                    "Waiting for global model update from version %d...", prev_version
                )
                self.model_cv.wait()

            return get_parameters(self.model), self.version_number

    def broadcast_updated_model(
        self, old_params: ModelParams, new_params: ModelParams, version_number: int
    ):
        logger.info(
            "Received an update from a client from global model version %d.",
            version_number,
        )
        self.buffer.add((old_params, new_params, version_number))
        return self.thread is not None

    def run(self):
        """
        Start the server thread.
        """

        def run_impl():
            for i in count():
                if not self.is_running or i >= self.num_aggregations:
                    logger.info("Aggregation loop terminating...")
                    break

                # Waits until aggregation buffer is ready to dispense items when called.
                aggregated_updates = self.buffer.get_items()

                if len(aggregated_updates) == 0:
                    logger.info("No updates in buffer to aggregate, skipping.")
                    continue

                logger.info(
                    "Server thread running aggregation for new version %d with %d updates.",
                    self.version_number + 1,
                    len(aggregated_updates),
                )

                # Aggregate and update model.
                new_model = self.strategy.aggregate(
                    self.model.named_parameters(), aggregated_updates
                )

                # Acquire model lock and update current global model. Notify any waiting threads.
                with self.model_mutex:
                    logger.info(
                        "Server thread updating global model to version %d.",
                        self.version_number + 1,
                    )
                    set_parameters(self.model, new_model)

                    _, accuracy = _test(
                        self.model, self.test_dataloader, device=self.device
                    )
                    logger.info("Server test set accuracy: %s", accuracy)
                    wandb.log(
                        {
                            "server": {
                                **{
                                    "accuracy": accuracy,
                                    "version": self.version_number + 1,
                                    "num_updates": len(aggregated_updates),
                                },
                                "global_version": self.version_number + 1,
                            }
                        }
                    )

                    self.version_number += 1
                    self.model_cv.notify_all()

        # Initialize thread once only
        if self.thread is None:
            self.is_running = True
            self.thread = Thread(target=run_impl)
            self.thread.start()
        else:
            raise RuntimeError("Server thread already running!")

    def stop(self):
        """
        Stop the server thread.
        """
        if self.thread is not None:
            self.is_running = False
            self.thread.join()
            self.thread = None

    def join(self):
        """
        Wait for the thread to terminate.
        """
        if self.thread is not None:
            self.thread.join()
            self.thread = None
