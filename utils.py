import torch
import torch.nn.utils.prune as prune
import numpy as np
import torch.quantization
from collections import OrderedDict
from typing import List
import mlflow

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

# Apply model pruning
def apply_pruning(model, amount=0.3):
    parameters_to_prune = [
        (model.fc1, "weight"),
        (model.fc2, "weight"),
        (model.fc3, "weight"),
    ]
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )
    
    # Ensure parameters track gradients after pruning
    for module, _ in parameters_to_prune:
        for param in module.parameters():
            param.requires_grad = True  # ✅ Enable gradients
    
    print("✅ Applied Pruning")
    return model

# Gradient Compression (Simple Sparsification)
def compress_gradients(gradients, sparsity=0.5):
    threshold = np.percentile(np.abs(gradients), sparsity * 100)
    compressed = np.where(np.abs(gradients) > threshold, gradients, 0)
    return compressed

# Dynamic Quantization (Best for CPU-based inference)
def apply_dynamic_quantization(model):
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print("Applied Dynamic Quantization")
    return model

# Static Quantization (Best for memory efficiency on edge devices)
def apply_static_quantization(model, example_input):
    model.eval()  # Set to evaluation mode
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # Choose quantization config
    torch.quantization.prepare(model, inplace=True)  # Prepare model for calibration
    model(example_input)  # Run a forward pass to calibrate activations
    torch.quantization.convert(model, inplace=True)  # Convert model to quantized version
    print("Applied Static Quantization")
    return model

# def get_parameters(net) -> List[np.ndarray]:
#     params = []
#     for k, v in net.state_dict().items():
#         if isinstance(v, torch.Tensor):
#             params.append(v.cpu().numpy())
#         else:
#             # Option 1: Raise an error to let you know which key is problematic
#             raise ValueError(f"Non-tensor parameter found in state_dict: {k}: {v}")
#             # Option 2 (if you prefer to skip non-tensors, but be cautious about mismatches):
#             # continue
#     return params



# def set_parameters(net, parameters: List[np.ndarray]):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    # Use model.parameters() to get only trainable parameters (tensors)
    return [param.detach().cpu().numpy() for param in net.parameters()]

def set_parameters(net, parameters: List[np.ndarray]):
    for param, new_param in zip(net.parameters(), parameters):
        param.data = torch.from_numpy(new_param).to(param.device)


def train(net, trainloader, partition_id, epochs: int):
    """Train the network on the training set."""
    with mlflow.start_run(run_name=f"Training Client {partition_id} Model",nested=True):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters())

        # 🔹 Enable gradients after pruning/quantization
        for param in net.parameters():
            param.requires_grad = True
            
        net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in trainloader:
                images, labels = batch["img"], batch["label"]
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
            mlflow.log_metric("Loss", epoch_loss, step=epoch)
            mlflow.log_metric("Accuracy", epoch_acc, step=epoch)


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy