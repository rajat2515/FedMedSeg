import multiprocessing as mp
import time
import flwr as fl

def start_server():
    strategy = fl.server.strategy.FedAvg(min_available_clients=2)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )

class DummyClient(fl.client.NumPyClient):
    def get_parameters(self, config): return []
    def fit(self, parameters, config): return [], 1, {}
    def evaluate(self, parameters, config): return 0.0, 1, {"accuracy": 1.0}

def start_client():
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=DummyClient())

if __name__ == "__main__":
    p_server = mp.Process(target=start_server)
    p_server.start()
    time.sleep(3)
    p_client1 = mp.Process(target=start_client)
    p_client2 = mp.Process(target=start_client)
    p_client1.start()
    p_client2.start()
    
    p_server.join()
    p_client1.join()
    p_client2.join()
    print("Done!")
