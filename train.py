import flwr
from flwr_datasets import FederatedDataset
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
import mlflow
from Clients import FlCustomClient
import warnings
import argparse

parser = argparse.ArgumentParser(description="Run Federated Learning with FLWR")
parser.add_argument("--disable-warnings", action="store_true", help="Suppress warnings")
args = parser.parse_args()

# Suppress warnings if flag is set
if args.disable_warnings:
    warnings.filterwarnings("ignore")

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  


print(f"Training on {utils.DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

# Load CIFAR-10 Federated Dataset
def load_datasets(partition_id: int, num_partitions: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=32)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, valloader, testloader

# Define a simple neural network
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
# Run FL Client
mlflow.set_experiment("FLWR_Client_Side_Optimization")
train_loader, val_loader, test_loader = load_datasets(partition_id=0, num_partitions=3)
model = Net().to(utils.DEVICE)

# Define the function to create clients
def numpyclient_fn(context: Context) -> flwr.client.Client:
    net = Net().to(utils.DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
    return FlCustomClient(partition_id, net, trainloader, valloader).to_client()

# Create ClientApp
numpyclient = ClientApp(client_fn=numpyclient_fn)

# Define Server Function
def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(config=config)

# Create ServerApp
server = ServerApp(server_fn=server_fn)

# Configure client resources
backend_config = {"client_resources": None}
if utils.DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}

NUM_PARTITIONS = 3

# Run the simulation
run_simulation(
    server_app=server,
    client_app=numpyclient,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)

torch.save(model.state_dict(), "optimized_model.pth")