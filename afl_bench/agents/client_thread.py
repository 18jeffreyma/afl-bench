
from threading import Thread
import time
from flwr.client import NumPyClient

from afl_bench.agents.runtime_model import RuntimeModel
from afl_bench.agents.server import ServerInterface


class ClientThread:
    def __init__(
        self,
        client: NumPyClient,
        server: ServerInterface,
        runtime_model: RuntimeModel
    ) -> None:
        self.client = client
        self.server = server
        self.runtime_model = runtime_model
        self.thread = None
        self.is_running = False
    
    def run(self, train_config={}, eval_config={}):
        def run_impl():
            while self.is_running:
                # Get latest global model and simulate client runtime.
                init_global_params, version = self.server.get_latest_global_model()
                
                # Simulate slow client runtime and fit model to local data.
                time.sleep(self.runtime_model.sample_runtime())
                new_parameters, _, _ = self.client.fit(init_global_params, train_config)
    
                _, _, metrics = self.client.evaluate(new_parameters, eval_config)
                
                # Broadcast updated model to server.
                self.server.broadcast_updated_model(init_global_params, new_parameters, version)
        
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