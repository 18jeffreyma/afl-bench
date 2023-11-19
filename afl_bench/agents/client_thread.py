import logging
import random
import time
from threading import Thread

import numpy as np
import torch

import wandb
from afl_bench.agents.clients import Client
from afl_bench.agents.runtime_model import RuntimeModel
from afl_bench.agents.server import ServerInterface

logger = logging.getLogger(__name__)


class ClientThread:
    def __init__(
        self,
        client: Client,
        server: ServerInterface,
        runtime_model: RuntimeModel,
        client_id: int,
        start_seed=42,
    ) -> None:
        self.client = client
        self.server = server
        self.client_id = client_id
        self.runtime_model = runtime_model
        self.thread = None
        self.is_running = False
        self.start_seed = start_seed

    def run(self, train_config={}, eval_config={}):
        def run_impl():
            prev_version = None

            # Set seed based on client id (otherwise all threads will have same seed).
            random.seed(self.start_seed + self.client_id)
            torch.random.manual_seed(self.start_seed + self.client_id)
            np.random.seed(self.start_seed + self.client_id)

            while self.is_running:
                # Get latest global model and simulate client runtime.
                init_global_params, version = self.server.get_current_model(
                    prev_version=prev_version
                )

                logger.info(
                    "Client thread running local training on model version %d",
                    version,
                )

                # Simulate slow client runtime and fit model to local data.
                time.sleep(self.runtime_model.sample_runtime())
                new_parameters, _, new_metrics = self.client.fit(
                    init_global_params, train_config
                )

                wandb.log(
                    {
                        f"client.{self.client_id}": {
                            **new_metrics,
                            "global_version": version,
                        }
                    },
                )

                _, _, metrics = self.client.evaluate(new_parameters, eval_config)
                logger.info("Client thread %d metrics: %s", self.client_id, metrics)

                wandb.log(
                    {
                        f"client.{self.client_id}": {
                            **metrics,
                            "global_version": version,
                        }
                    },
                )

                # Broadcast updated model to server. If server indicates not running, stop.
                server_running = self.server.broadcast_updated_model(
                    self.client_id, init_global_params, new_parameters, version
                )
                prev_version = version

                if not server_running:
                    self.is_running = False

        # Initialize thread once only
        if self.thread is None:
            self.is_running = True
            self.thread = Thread(target=run_impl, daemon=True)
            self.thread.start()
        else:
            raise RuntimeError("Client thread already running!")

    def stop(self):
        """
        Stop the client thread.
        """
        if self.thread is not None:
            self.is_running = False
            self.thread.join()
            self.thread = None
