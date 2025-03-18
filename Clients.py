from flwr.client import NumPyClient
import utils
import mlflow
import time
import psutil
import torch
import torch.nn.utils.prune as prune


class FlCustomClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader, quantization_type="dynamic"):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.quant = quantization_type

        # Apply pruning
        self.net = utils.apply_pruning(self.net)

        for module in [self.net.fc1, self.net.fc2, self.net.fc3]:
            prune.remove(module, "weight")

        # Apply quantization
        if self.quant == "dynamic":
            self.net = torch.quantization.quantize_dynamic(self.net, {torch.nn.Linear}, dtype=torch.qint8)
        elif self.quant == "static":
            self.net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(self.net, inplace=True)
            example_input = torch.randn(1, 3, 32, 32).to(utils.DEVICE)
            self.net(example_input)
            torch.quantization.convert(self.net, inplace=True)    
    
    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return utils.get_parameters(self.net)
    
    def set_parameters(self, parameters):
        utils.set_parameters(self.net, parameters)
    
    def fit(self, parameters, config):
        with mlflow.start_run(run_name=f"Client {self.partition_id} Fit"):
            print(f"[Client {self.partition_id}] fit, config: {config}")
            self.set_parameters(parameters)

            # Enable gradients after pruning/quantization
            for param in self.net.parameters():
                param.requires_grad = True

            # Freeze lower layers
            for param in self.net.conv1.parameters():
                param.requires_grad = False
            for param in self.net.conv2.parameters():
                param.requires_grad = False
            
            mem_usage_before = psutil.Process().memory_info().rss / (1024 * 1024)
            start_time = time.time()
            
            utils.train(self.net, self.trainloader, self.partition_id, epochs=1)
            compressed_params = [utils.compress_gradients(param) for param in self.get_parameters(self.net)]
            
            end_time = time.time()
            mem_usage_after = psutil.Process().memory_info().rss / (1024 * 1024)

            epoch = config.get("epoch", 0)
            mlflow.log_metric("training_memory_usage", mem_usage_after - mem_usage_before, step=epoch)
            mlflow.log_metric("training_time", end_time - start_time, step=epoch)
            mlflow.log_param("quantization", self.quant)
            
            return compressed_params, len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        with mlflow.start_run(run_name=f"Client {self.partition_id} Evaluation"):
            print(f"[Client {self.partition_id}] evaluate, config: {config}")
            self.set_parameters(parameters)
            
            start_time = time.time()
            mem_usage_before = psutil.Process().memory_info().rss / (1024 * 1024)
            
            loss, accuracy = utils.test(self.net, self.valloader)
            
            end_time = time.time()
            mem_usage_after = psutil.Process().memory_info().rss / (1024 * 1024)
            
            mlflow.log_metric("evaluation_memory_usage", mem_usage_after - mem_usage_before)
            mlflow.log_metric("evaluation_time", end_time - start_time)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("loss", loss)
            
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}