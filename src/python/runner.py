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


class AddLayer(nn.Module):
    """Layer that performs element-wise addition of two inputs (for skip connections)"""

    def forward(self, x1, x2):
        return x1 + x2


class DynamicModel(nn.Module):
    """Dynamically build PyTorch model from architecture JSON"""

    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.layers = nn.ModuleDict()
        self.forward_order = []
        self.node_inputs = {}  # Track which nodes provide input to each node

        self._build_model()

    def _build_model(self):
        """Build model from architecture definition"""
        nodes = self.architecture['nodes']
        edges = self.architecture['edges']

        # Validate architecture has nodes
        if not nodes or len(nodes) == 0:
            raise ValueError(
                "❌ Model has no layers!\n\n"
                "Please add at least one Input layer and one Output layer to your model."
            )

        # Check for Input and Output layers
        has_input = any(n['data']['layerType'] == 'Input' for n in nodes)
        has_output = any(n['data']['layerType'] == 'Output' for n in nodes)

        if not has_input:
            raise ValueError(
                "❌ Model is missing an Input layer!\n\n"
                "Every model must start with an Input layer.\n"
                "Add an Input layer from the Layer Palette on the left."
            )

        if not has_output:
            raise ValueError(
                "❌ Model is missing an Output layer!\n\n"
                "Every model must end with an Output layer.\n"
                "Add an Output layer from the Layer Palette on the left."
            )

        # Build node inputs mapping from edges
        for edge in edges:
            target = edge['target']
            source = edge['source']
            if target not in self.node_inputs:
                self.node_inputs[target] = []
            self.node_inputs[target].append(source)

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

        elif layer_type == 'Add':
            return AddLayer()

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
        # Track outputs from each node for skip connections
        outputs = {}

        # Find the input node ID
        input_node = next((n for n in self.architecture['nodes'] if n['data']['layerType'] == 'Input'), None)
        if input_node:
            outputs[input_node['id']] = x

        for layer_id, layer_type in self.forward_order:
            layer = self.layers[layer_id]

            # Get inputs for this node
            input_nodes = self.node_inputs.get(layer_id, [])

            if layer_type == 'Add' and len(input_nodes) >= 2:
                # Add layer needs two inputs
                # First input is the main path (sequential), second is the skip connection
                input1 = outputs.get(input_nodes[0], x)
                input2 = outputs.get(input_nodes[1], x)
                x = layer(input1, input2)
            elif input_nodes:
                # Regular layer - use the last output from previous node
                x = outputs.get(input_nodes[0], x)
                x = layer(x)
            else:
                # No specific input - use current x
                x = layer(x)

            # Store output for this node
            outputs[layer_id] = x

        return x


class NumpyDataset(Dataset):
    """Dataset wrapper for NumPy arrays"""

    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def load_dataset(dataset_path, config):
    """Load dataset from various formats"""
    import numpy as np
    from pathlib import Path

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")

    # NPZ files (NumPy arrays)
    if dataset_path.suffix == '.npz':
        data = np.load(dataset_path)
        print(f"NPZ file keys: {data.files}")

        # Try common key patterns
        if 'X_train' in data.files and 'y_train' in data.files:
            # Pre-split data
            X_train = data['X_train']
            y_train = data['y_train']
            X_val = data.get('X_val', data.get('X_test', None))
            y_val = data.get('y_val', data.get('y_test', None))

            print(f"Loaded pre-split data: train={X_train.shape}, val={X_val.shape if X_val is not None else 'None'}")

            # Normalize if needed (0-255 -> 0-1)
            if X_train.max() > 1.0:
                X_train = X_train / 255.0
                if X_val is not None:
                    X_val = X_val / 255.0

            # Add channel dimension if needed (for MNIST: 28x28 -> 28x28x1)
            if len(X_train.shape) == 3:  # (N, H, W)
                X_train = np.expand_dims(X_train, axis=-1)
                if X_val is not None:
                    X_val = np.expand_dims(X_val, axis=-1)

            # Convert to (N, C, H, W) for PyTorch
            if len(X_train.shape) == 4:  # (N, H, W, C)
                X_train = np.transpose(X_train, (0, 3, 1, 2))
                if X_val is not None:
                    X_val = np.transpose(X_val, (0, 3, 1, 2))

            train_dataset = NumpyDataset(X_train, y_train)
            val_dataset = NumpyDataset(X_val, y_val) if X_val is not None else None

            return train_dataset, val_dataset

        elif 'X' in data.files and 'y' in data.files:
            # Single dataset, need to split
            X = data['X']
            y = data['y']

            # Normalize
            if X.max() > 1.0:
                X = X / 255.0

            # Add channel dimension if needed
            if len(X.shape) == 3:
                X = np.expand_dims(X, axis=-1)

            # Convert to (N, C, H, W)
            if len(X.shape) == 4:
                X = np.transpose(X, (0, 3, 1, 2))

            dataset = NumpyDataset(X, y)

            # Split according to config
            split_ratio = config.get('splitRatio', [0.8, 0.2])
            train_size = int(len(dataset) * split_ratio[0])
            val_size = len(dataset) - train_size

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            print(f"Split dataset: train={train_size}, val={val_size}")
            return train_dataset, val_dataset

        else:
            raise ValueError(f"Unknown NPZ format. Available keys: {data.files}")

    # PyTorch tensors
    elif dataset_path.suffix in ['.pt', '.pth']:
        data = torch.load(dataset_path)
        # TODO: Implement PyTorch data loading
        raise NotImplementedError("PyTorch dataset loading not yet implemented")

    # Directory with images
    elif dataset_path.is_dir():
        # TODO: Implement image folder loading
        raise NotImplementedError("Image folder loading not yet implemented")

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


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


def emit_metrics(epoch, loss, val_loss=None, metrics=None, batch=None, total_batches=None):
    """Emit training metrics to stdout for extension to capture"""
    metrics_data = {
        'epoch': epoch,
        'loss': float(loss),
        'valLoss': float(val_loss) if val_loss is not None else None,
        'metrics': metrics or {},
        'timestamp': int(time.time() * 1000)
    }
    if batch is not None:
        metrics_data['batch'] = batch
    if total_batches is not None:
        metrics_data['totalBatches'] = total_batches
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

    # Initialize lazy modules with a dummy forward pass
    # Get input shape from the Input layer
    input_node = next((n for n in architecture['nodes'] if n['data']['layerType'] == 'Input'), None)
    if input_node:
        input_shape = input_node['data']['params'].get('inputShape', [28, 28, 1])
        # Convert HWC to NCHW format for PyTorch (batch_size=1, channels, height, width)
        if len(input_shape) == 3:
            dummy_input = torch.randn(1, input_shape[2], input_shape[0], input_shape[1]).to(device)
        elif len(input_shape) == 2:
            dummy_input = torch.randn(1, input_shape[0], input_shape[1]).to(device)
        elif len(input_shape) == 1:
            dummy_input = torch.randn(1, input_shape[0]).to(device)
        else:
            dummy_input = torch.randn(1, 1, 28, 28).to(device)

        # Initialize lazy modules
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"Warning: Could not initialize model with dummy input: {e}")

    # Validate model has trainable layers
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params == 0:
        raise ValueError(
            "❌ Model has no trainable layers!\n\n"
            "Your model only contains Input/Output layers or other non-trainable layers.\n"
            "Please add at least one trainable layer (Dense, Conv2D, etc.) between your Input and Output layers.\n\n"
            "Example: Input → Dense → Output"
        )
    print(f"Total trainable parameters: {num_params:,}")

    # Load dataset
    dataset_path = config.get('datasetPath')
    if not dataset_path:
        raise ValueError("No dataset path specified in config")

    train_dataset, val_dataset = load_dataset(dataset_path, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batchSize'],
        shuffle=True
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batchSize'],
            shuffle=False
        )
    else:
        val_loader = None

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
        total_batches = len(train_loader)

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

            # Print batch progress (not emitted as metrics)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                batch_loss = train_loss / (batch_idx + 1)
                batch_accuracy = train_correct / train_total
                print(f"  Batch [{batch_idx + 1}/{total_batches}] - Loss: {batch_loss:.4f}, Acc: {batch_accuracy:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        avg_val_loss = None
        val_accuracy = None

        if val_loader:
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
        metrics_dict = {'accuracy': train_accuracy}
        if val_accuracy is not None:
            metrics_dict['val_accuracy'] = val_accuracy

        emit_metrics(
            epoch,
            avg_train_loss,
            avg_val_loss,
            metrics_dict
        )

        # Print progress
        progress_msg = f"Epoch {epoch}/{config['epochs']} - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}"
        if avg_val_loss is not None:
            progress_msg += f", Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        print(progress_msg)

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
