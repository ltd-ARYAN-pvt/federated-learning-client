import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import time
import psutil
from utils import compress_gradients

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, quantization_type="dynamic"):
        self.model = model
        self.train_loader = train_loader
        self.quant=quantization_type
    
    def get_parameters(self):
        return [param.detach().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.dtype)
    
    def fit(self, parameters, config):
        with mlflow.start_run(run_name="Custom Client"):
            self.set_parameters(parameters)
            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
            loss_fn = nn.CrossEntropyLoss()

            # Monitor memory usage
            mem_usage_before = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
            start_time = time.time()
            
            # Selectively train only the last layer
            for param in self.model.fc1.parameters():
                param.requires_grad = False  # Freeze lower layers
            
            total_loss = 0
            for epoch in range(1):  # Simulated lightweight training
                for images, labels in self.train_loader:
                    optimizer.zero_grad()
                    output = self.model(images)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # Apply compression before sending updates
            gradients = [p.grad.numpy() for p in self.model.parameters() if p.grad is not None]
            compressed_gradients = [compress_gradients(g) for g in gradients]
            
            end_time = time.time()
            mem_usage_after = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Log metrics to MLflow
            mlflow.log_metric("training_loss", total_loss / len(self.train_loader), step=epoch)
            mlflow.log_metric("training_memory_usage", mem_usage_after - mem_usage_before, step=epoch)
            mlflow.log_metric("training_time", end_time - start_time, step=epoch)
            mlflow.log_param("pruning_amount", 0.3)
            mlflow.log_param("quantization", self.quant)
            
            return self.get_parameters(), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        with mlflow.start_run():
            self.set_parameters(parameters)
            start_time = time.time()
            mem_usage_before = psutil.Process().memory_info().rss / (1024 * 1024)
            
            accuracy = 0.9  # Placeholder
            
            end_time = time.time()
            mem_usage_after = psutil.Process().memory_info().rss / (1024 * 1024)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("evaluation_memory_usage", mem_usage_after - mem_usage_before)
            mlflow.log_metric("evaluation_time", end_time - start_time)
            return accuracy, len(self.train_loader.dataset), {}