
from abc import abstractmethod
import copy
from threading import Lock, Thread
from typing import List, Tuple

from afl_bench.agents.buffer import Buffer
from afl_bench.agents.common import get_parameters, set_parameters
from afl_bench.agents.strategies import ModelParams, Strategy


class ServerInterface:
    @abstractmethod
    def get_current_model(self) -> Tuple[ModelParams, int]:
        """
        Get the current global model parameters.
        """
        pass
    
    @abstractmethod
    def broadcast_updated_model(
        self, 
        old_params: ModelParams,
        new_params: ModelParams,
        version_number: int
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
        pass

class Server(ServerInterface):
    def __init__(self, initial_model, strategy: Strategy):
        self.current_model = initial_model
        self.version_number = 0
        self.strategy = strategy
        
        self.is_running = False
        self.thread = None
        
        self.buffer = Buffer(wait_for_full=strategy.wait_for_full, n=strategy.buffer_size)
        self.model_mutex = Lock()

    def get_current_model(self) -> Tuple[ModelParams, int]:
        with self.model_mutex:
            return get_parameters(self.current_model), self.version_number

    def broadcast_updated_model(
        self, 
        old_params: ModelParams, 
        new_params: ModelParams, 
        version_number: int
    ):
        self.buffer.add((old_params, new_params, version_number))
    
    def run(self):
        def run_impl():
            while self.is_running:
                # Wait until aggregation buffer is full and retrieve updates.
                # TODO: Add optional time delay?
                aggregated_updates = self.buffer.get_items()

                # Aggregate and update model.
                model_update = self.strategy.aggregate(self.current_model.parameters(), aggregated_updates)
                
                # Acquire model lock and update current global model.s
                with self.model_mutex:
                    set_parameters(self.current_model, model_update)
                    self.version_number += 1
                     
        # Initialize thread once only
        if self.thread is None:
            self.is_running = True
            self.thread = Thread(target=run_impl)
            self.thread.start()
        else:
            raise RuntimeError("Client thread already running!")
        
    def stop(self):
        if self.thread is not None:
            self.is_running = False
            self.thread.join()
            self.thread = None
        
    