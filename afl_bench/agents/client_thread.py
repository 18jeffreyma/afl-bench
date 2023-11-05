import logging
import time
from threading import Thread

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
    ) -> None:
        self.client = client
        self.server = server
        self.client_id = client_id
        self.runtime_model = runtime_model
        self.thread = None
        self.is_running = False

    def run(self, train_config={}, eval_config={}):
        def run_impl():
            while self.is_running:
                logger.info("Client thread running local training.")

                # Get latest global model and simulate client runtime.
                init_global_params, version = self.server.get_current_model()

                # Simulate slow client runtime and fit model to local data.
                time.sleep(self.runtime_model.sample_runtime())
                new_parameters, _, new_metrics = self.client.fit(
                    init_global_params, train_config
                )

                wandb.log(
                    {
                        f"client/{self.client_id}": {
                            **new_metrics,
                            "global_version": version,
                        }
                    }
                )

                _, _, metrics = self.client.evaluate(new_parameters, eval_config)
                logger.info("Client thread: %s", metrics)

                wandb.log(
                    {
                        f"client/{self.client_id}": {
                            **metrics,
                            "global_version": version,
                        }
                    }
                )

                # Broadcast updated model to server.
                self.server.broadcast_updated_model(
                    init_global_params, new_parameters, version
                )

        # Initialize thread once only
        if self.thread is None:
            self.is_running = True
            self.thread = Thread(target=run_impl)
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
