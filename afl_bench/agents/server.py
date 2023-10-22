
from abc import abstractmethod
from threading import Thread

from afl_bench.agents.buffer import Buffer
from afl_bench.agents.strategies import Strategy


class ServerInterface:
    @abstractmethod
    def get_current_model(self):
        """
        Get the current global model parameters.
        """
        pass
    
    @abstractmethod
    def broadcast_updated_model(self, old_params, new_params, version_number):
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

    def get_current_model(self):
        return self.current_model, self.version_number

    def broadcast_updated_model(self, old_params, new_params, version_number):
        # TODO: Implement buffer for multiple clients.
        pass
        
    def run(self):
        def run_impl():
            while self.is_running:
                # Wait until aggregation buffer is full and retrieve updates.
                aggregated_updates = self.buffer.get_items()

                model_update = self.strategy.aggregate(aggregated_updates)

                # Update global model.
                # TODO: Implement model update.
                     
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
        
    