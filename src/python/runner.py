#!/usr/bin/env python3
"""
NetSmith Training Runner
Loads architecture and config, builds PyTorch model, and trains it.
"""

import sys
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import time


class DynamicModel(nn.Module):
    """Dynamically build PyTorch model from architecture JSON"""

    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.layers = nn.ModuleDict()
        self.forward_order = []

        self._build_model()

    def _build_model(self):
        """Build model from architecture definition"""
        nodes = self.architecture['nodes']
        edges = self.architecture['edges']

        # Sort nodes topologically
        sorted_nodes = self._topological_sort(nodes, edges)

        # Build layers
        for node in sorted_nodes:
            layer_type = node['data']['layerType']
            params = node['data'].get('params', {})
            node_id = node['id']

            if layer_type == 'Input' or layer_type == 'Output':
                continue

            layer = self._create_layer(layer_type, params)
            if layer is not None:
                self.layers[node_id] = layer
                self.forward_order.append((node_id, layer_type))

    def _create_layer(self, layer_type, params):
        """Create a PyTorch layer from type and params"""
        if layer_type == 'Dense':
            # Note: in_features needs to be set dynamically or passed
            units = params.get('units', 128)
            return nn.LazyLinear(units)

        elif layer_type == 'Conv2D':
            filters = params.get('filters', 32)
            kernel_size = params.get('kernelSize', 3)
            padding = params.get('padding', 'same')
            padding_mode = 'same' if padding == 'same' else 0
            return nn.LazyConv2d(filters, kernel_size, padding=padding_mode)

        elif layer_type == 'Conv1D':
            filters = params.get('filters', 32)
            kernel_size = params.get('kernelSize', 3)
            padding = params.get('padding', 'same')
            padding_mode = 'same' if padding == 'same' else 0
            return nn.LazyConv1d(filters, kernel_size, padding=padding_mode)

        elif layer_type == 'MaxPool2D':
            pool_size = params.get('poolSize', 2)
            return nn.MaxPool2d(pool_size)

        elif layer_type == 'AvgPool2D':
            pool_size = params.get('poolSize', 2)
            return nn.AvgPool2d(pool_size)

        elif layer_type == 'Flatten':
            return nn.Flatten()

        elif layer_type == 'Dropout':
            rate = params.get('rate', 0.5)
            return nn.Dropout(rate)

        elif layer_type == 'BatchNorm':
            return nn.LazyBatchNorm2d()

        elif layer_type == 'Activation':
            activation = params.get('activation', 'relu')
            return self._get_activation(activation)

        return None

    def _get_activation(self, activation_type):
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'softmax': nn.Softmax(dim=1),
            'linear': nn.Identity()
        }
        return activations.get(activation_type, nn.ReLU())

    def _topological_sort(self, nodes, edges):
        """Simple topological sort of nodes"""
        # Find input node
        input_node = next((n for n in nodes if n['data']['layerType'] == 'Input'), None)
        if not input_node:
            return nodes

        # Build adjacency list
        adj = {node['id']: [] for node in nodes}
        for edge in edges:
            adj[edge['source']].append(edge['target'])

        # DFS
        visited = set()
        sorted_nodes = []

        def visit(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            node = next((n for n in nodes if n['id'] == node_id), None)
            if node:
                sorted_nodes.append(node)
            for neighbor in adj.get(node_id, []):
                visit(neighbor)

        visit(input_node['id'])
        return sorted_nodes

    def forward(self, x):
        """Forward pass through the network"""
        for layer_id, layer_type in self.forward_order:
            layer = self.layers[layer_id]
            x = layer(x)
        return x


class DummyDataset(Dataset):
    """Dummy dataset for testing - replace with actual data loading"""

    def __init__(self, size=1000, input_shape=(3, 32, 32), num_classes=10):
        self.size = size
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random data
        x = torch.randn(*self.input_shape)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def get_optimizer(name, parameters, lr):
    """Get optimizer by name"""
    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adamw': optim.AdamW,
        'adagrad': optim.Adagrad
    }
    opt_class = optimizers.get(name, optim.Adam)
    return opt_class(parameters, lr=lr)


def get_loss_function(name):
    """Get loss function by name"""
    losses = {
        'cross_entropy': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'binary_cross_entropy': nn.BCEWithLogitsLoss(),
        'huber': nn.HuberLoss()
    }
    return losses.get(name, nn.CrossEntropyLoss())


def emit_metrics(epoch, loss, val_loss=None, metrics=None):
    """Emit training metrics to stdout for extension to capture"""
    metrics_data = {
        'epoch': epoch,
        'loss': float(loss),
        'valLoss': float(val_loss) if val_loss is not None else None,
        'metrics': metrics or {},
        'timestamp': int(time.time() * 1000)
    }
    print(f"METRICS:{json.dumps(metrics_data)}", flush=True)


def train(run_path):
    """Main training function"""
    run_path = Path(run_path)

    # Load architecture and config
    with open(run_path / 'architecture.json') as f:
        architecture = json.load(f)

    with open(run_path / 'config.json') as f:
        config = json.load(f)

    print(f"Loaded architecture with {len(architecture['nodes'])} nodes")
    print(f"Training config: {config}")

    # Determine device
    device_name = config['device']
    if device_name == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")

    # Build model
    model = DynamicModel(architecture)
    model = model.to(device)

    print("Model architecture:")
    print(model)

    # Create dummy dataset (TODO: load real data)
    dataset = DummyDataset(size=1000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batchSize'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batchSize'],
        shuffle=False
    )

    # Setup optimizer and loss
    optimizer = get_optimizer(
        config['optimizer'],
        model.parameters(),
        config['learningRate']
    )
    criterion = get_loss_function(config['loss'])

    # Training loop
    print(f"Starting training for {config['epochs']} epochs...")

    for epoch in range(1, config['epochs'] + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Emit metrics
        emit_metrics(
            epoch,
            avg_train_loss,
            avg_val_loss,
            {'accuracy': train_accuracy, 'val_accuracy': val_accuracy}
        )

        print(f"Epoch {epoch}/{config['epochs']} - "
              f"Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save checkpoint
        if config['saveCheckpoints'] and epoch % config.get('checkpointFrequency', 5) == 0:
            checkpoint_path = run_path / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = run_path / 'weights.pt'
    torch.save(model.state_dict(), final_path)
    print(f"Training completed! Model saved to {final_path}")

    # Save metrics
    metrics_path = run_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'final_train_accuracy': train_accuracy,
            'final_val_accuracy': val_accuracy
        }, f, indent=2)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: runner.py <run_path>")
        sys.exit(1)

    run_path = sys.argv[1]

    try:
        train(run_path)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
