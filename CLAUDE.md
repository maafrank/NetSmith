# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NetSmith is a VS Code extension for visually designing, configuring, and training neural networks locally. It provides a drag-and-drop interface using React Flow, PyTorch training orchestration via Python subprocess, and real-time metrics visualization.

## Build & Development Commands

```bash
# Install dependencies
npm install                    # Extension dependencies
cd webview && npm install      # WebView dependencies

# Build
npm run build                  # Build both extension and webview
npm run build:extension        # TypeScript compilation only
npm run build:webview          # React/Vite build only

# Development mode (auto-rebuild)
npm run watch                  # Watch extension (TypeScript)
npm run watch:webview          # Watch webview (Vite dev server)
npm run dev                    # Run both watches in parallel

# Running the extension
# Press F5 in VS Code to launch Extension Development Host
```

## Architecture

### Three-Layer Communication Pattern

1. **Extension Host (TypeScript)** - `src/`
   - Manages VS Code API integration, commands, and lifecycle
   - Orchestrates Python subprocess for training
   - Handles file I/O for projects and runs

2. **WebView UI (React)** - `webview/`
   - Visual graph editor using React Flow
   - Training configuration and metrics visualization
   - Communicates with extension via `postMessage` API

3. **Python Backend** - `src/python/runner.py`
   - Dynamically builds PyTorch models from JSON architecture
   - Runs training loops and emits metrics via stdout
   - Supports NPZ dataset loading with automatic preprocessing

### Message Passing Flow

```
WebView (React) <--postMessage--> ModelBuilderPanel <--> TrainingManager
                                                              |
                                                         Python Process
                                                         (stdout metrics)
```

**Key message types:**
- `ready` (webview → extension): Webview loaded, triggers dataset scan
- `runModel` (webview → extension): Trigger training with architecture + config
- `saveModel` (webview → extension): Save model with optional name
- `trainingMetrics` (extension → webview): Stream epoch metrics
- `trainingError` (extension → webview): Send detailed error message
- `pickDatasetFile` (webview → extension): Open file picker dialog
- `datasetPathSelected` (extension → webview): Return selected file path
- `scanForDatasets` (webview → extension): Request workspace scan
- `availableDatasets` (extension → webview): Send list of found datasets

### State Management Critical Pattern

**React Flow is the source of truth** for node positions and data. The Zustand store is only synced when needed (save/export).

```typescript
// App.tsx sync pattern
const syncToStore = useCallback(() => {
  setNodes(rfNodes as any);
  setEdges(rfEdges as any);
}, [rfNodes, rfEdges, setNodes, setEdges]);

// Only sync store → React Flow when new nodes are added
// IMPORTANT: Merge new nodes instead of replacing entire state
useEffect(() => {
  if (nodes.length > rfNodes.length) {
    const rfNodeIds = new Set(rfNodes.map(n => n.id));
    const newNodes = nodes.filter(n => !rfNodeIds.has(n.id));
    setRfNodes([...rfNodes, ...newNodes as any]);
  }
}, [nodes.length, nodes, rfNodes, setRfNodes]);
```

**Critical patterns:**
- **Do not create bidirectional syncing** - this causes infinite loops
- **Never replace entire React Flow state** when adding nodes - merge instead to preserve positions
- Always call `syncToStore()` before operations that need store data (like running training)
- React Flow zoom/pan state is independent - use `defaultViewport` to set initial zoom (e.g., `zoom: 0.8`)
- Avoid `fitView` prop as it causes unwanted zoom changes when nodes are added
- Default panel widths are 200px each (can be resized 50px-600px)

### Auto-Connect Feature

When clicking a layer in the palette, it automatically connects to the last layer in the network (the one with no outgoing edges). This is implemented in `store.ts` `addNode()` action:

```typescript
// Find nodes without outgoing edges
const outgoingEdges = new Set(state.edges.map(e => e.source));
const nodesWithoutOutgoing = state.nodes.filter(n => !outgoingEdges.has(n.id));

// If exactly one exists, auto-connect to new node
if (nodesWithoutOutgoing.length === 1) {
  const lastNode = nodesWithoutOutgoing[0];
  newEdges.push({
    id: `${lastNode.id}-${node.id}`,
    source: lastNode.id,
    target: node.id,
  });
}
```

### Node Deletion Sync Pattern

Both the delete button in PropertiesPanel and the Delete key work by updating both React Flow and store states:

```typescript
// PropertiesPanel delete button
const handleDelete = () => {
  deleteNode(id);      // Update store (also clears selectedNode)
  onDeleteNode(id);    // Update React Flow
};

// React Flow Delete key handling (in App.tsx)
const onNodesChange = useCallback((changes: any) => {
  onNodesChangeInternal(changes);
  const removedNodes = changes.filter((change: any) => change.type === 'remove');
  if (removedNodes.length > 0) {
    removedNodes.forEach((change: any) => {
      deleteNode(change.id);  // Sync deletion to store
    });
  }
}, [onNodesChangeInternal, deleteNode]);
```

Key: Edge syncing only happens when edge count changes to prevent "undo" behavior where deletions are restored.

### Project File Structure

```
.netsmith/                      # Generated per-workspace
├── config.json                 # Project settings
├── models/
│   └── model_*.json           # Saved architectures
├── runs/
│   └── run_*/
│       ├── architecture.json   # Model definition
│       ├── config.json         # Training hyperparameters
│       ├── metrics.json        # Final metrics
│       └── weights.pt          # Trained weights
└── blocks/                     # Reusable layer groups (future)
```

## Key Components

### ModelBuilderPanel (`src/webview/ModelBuilderPanel.ts`)
- Singleton webview panel manager
- Handles all message passing between extension and React UI
- Coordinates ProjectManager and TrainingManager
- Implements file picker for dataset selection (opens in workspace root by default)
- Auto-scans workspace for dataset files on webview ready
- Returns relative paths from workspace root for better portability

### TrainingManager (`src/training/TrainingManager.ts`)
- Spawns Python subprocess with `runner.py`
- Parses `METRICS:{json}` from stdout for real-time updates
- Captures stderr in a buffer to provide detailed error messages on failure
- Integrates with Python extension API to get active interpreter
- Implements topological sort for layer ordering

### Python Runner (`src/python/runner.py`)
- `DynamicModel` class builds PyTorch `nn.Module` from JSON
- Uses `LazyLinear`, `LazyConv2d`, and `LazyBatchNorm2d` for automatic input shape inference
- **CRITICAL**: Lazy modules must be initialized with a dummy forward pass before counting parameters:
  ```python
  # Get input shape from Input layer and create dummy tensor
  dummy_input = torch.randn(1, channels, height, width).to(device)
  with torch.no_grad():
      _ = model(dummy_input)  # Initialize lazy modules
  # Now safe to count parameters
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  ```
- **Validates model architecture** before training:
  - Checks for missing Input/Output layers
  - Validates model has trainable parameters
  - Provides clear, user-friendly error messages with suggestions
- `load_dataset()` supports NPZ format with:
  - Pre-split data (`X_train`, `y_train`, `X_test`, `y_test`)
  - Auto-normalization (0-255 → 0-1)
  - Channel dimension handling (HWC → CHW for PyTorch)
  - Automatic splitting if single dataset provided
- Prints batch progress every 10 batches (not emitted as metrics)
- Only emits epoch-level metrics to keep graph clean

### Zustand Store (`webview/src/store.ts`)
- Manages application state (nodes, edges, training config, metrics, errors)
- **Important**: `updateNode` must update both `nodes` array AND `selectedNode` to keep PropertiesPanel in sync
- Stores `trainingError` to display detailed error messages in UI
- Stores `showTrainingConfig` to control training panel expansion/collapse across the app

### Layer Components (`webview/src/components/`)
- **LayerPalette**: Click to add layers (auto-connects to last layer in network)
- **PropertiesPanel**: Layer configuration (calls both `updateNode` in store AND `onUpdateNode` callback for React Flow)
  - Delete button updates both store and React Flow states
  - Must accept `onDeleteNode` callback from parent
- **TrainingPanel**: Hyperparameter configuration and run controls
  - Auto-scans workspace for datasets on load
  - Shows dropdown of available datasets (relative paths)
  - Includes file picker button for manual selection
  - Prompts for model name on save (with sanitization)
  - Collapses when clicking outside the panel or on the React Flow canvas (via `onPaneClick`)
- **MetricsPanel**: Real-time training visualization with batch progress and minimize toggle
  - Displays detailed error messages in red banner when training fails
  - Shows full Python traceback in scrollable code block

## Type Definitions

Types are defined in **two locations**:
1. `src/types.ts` - Extension-side types (canonical)
2. `webview/src/types.ts` - WebView-side types (should mirror extension types)

**Always update both** when adding fields like `batch` and `totalBatches` to `TrainingMetrics`.

## Layer System

Supported layer types (in `LayerType` union):
- **Basic**: Input, Output
- **Dense**: Dense
- **Convolutional**: Conv2D, Conv1D
- **Pooling**: MaxPool2D, AvgPool2D, GlobalAvgPool2D, GlobalMaxPool2D
- **Normalization**: BatchNorm
- **Regularization**: Dropout
- **Activation**: Activation (relu, sigmoid, tanh, softmax, leaky_relu, elu, selu, gelu, swish, linear)
- **Utility**: Flatten, Reshape, UpSampling2D
- **Merge**: Add, Concat, Multiply, Subtract, Maximum, Minimum (for skip connections and multi-input layers)
- **Blocks**: Block (for reusable modules with internal architecture)

Each layer has:
- Visual node representation (`webview/src/components/nodes/index.tsx`)
- Parameter schema in `LayerParams` interface
- Python mapping in `DynamicModel._create_layer()`

### Merge Layers and Skip Connections

Merge layers (Add, Concat, etc.) require **two inputs** and are handled specially in the forward pass:
```python
# In runner.py forward() method
if layer_type in merge_layers and len(input_nodes) >= 2:
    input1 = outputs.get(input_nodes[0], x)  # Main path
    input2 = outputs.get(input_nodes[1], x)  # Skip connection
    x = layer(input1, input2)
```

The `AddLayer` class automatically handles dimension mismatches using 1x1 convolution projections for skip connections.

## Common Patterns

### Adding a New Layer Type

1. Add to `LayerType` union in both `src/types.ts` and `webview/src/types.ts`
2. Add parameters to `LayerParams` interface
3. Create node component in `webview/src/components/nodes/index.tsx`
4. Add to `LayerPalette.tsx` with default params
5. Add parameter UI in `PropertiesPanel.tsx`
6. Implement Python layer creation in `runner.py._create_layer()`

### Dataset Loading Extension

`load_dataset()` in `runner.py` uses file extension detection:
- `.npz` → NumPy arrays (implemented)
- `.pt`/`.pth` → PyTorch tensors (TODO)
- Directory → Image folder loader (TODO)
- `.h5`/`.hdf5` → HDF5 format (TODO)

### Panel Resize Constraints

Left and right panels have dynamic width constraints in `App.tsx`:
```typescript
const newWidth = Math.max(50, Math.min(600, e.clientX));
```
Minimum: 50px (allows near-complete minimization)
Maximum: 600px

### UI Interaction Patterns

**Click-outside behavior for collapsible panels:**
- Use Zustand store state (not local component state) for UI elements that need to respond to clicks on the React Flow canvas
- Add `onPaneClick` handler to ReactFlow component to trigger state changes
- Example: TrainingPanel uses `showTrainingConfig` in store, which is toggled by both the gear button and `onPaneClick` in App.tsx

## Important Implementation Details

### React Flow Configuration
```typescript
// Hide attribution link
proOptions={{ hideAttribution: true }}

// Edge arrows
const defaultEdgeOptions = {
  markerEnd: { type: MarkerType.ArrowClosed },
};

// Auto-Layout using Dagre (top-to-bottom)
dagreGraph.setGraph({ rankdir: 'TB', nodesep: 100, ranksep: 150 });
```

### Batch Progress Display
Python prints batch progress to console but only emits epoch-level metrics:
```python
# Print to console (not emitted)
if (batch_idx + 1) % 10 == 0:
    print(f"  Batch [{batch_idx + 1}/{total_batches}] - Loss: {batch_loss:.4f}")

# Only emit at end of epoch
emit_metrics(epoch, avg_train_loss, avg_val_loss, metrics_dict)
```
This keeps the metrics graph clean with one point per epoch.

### Input Validation
All numeric inputs in `PropertiesPanel.tsx` must validate before updating:
```typescript
const value = parseInt(e.target.value);
if (!isNaN(value) && value > 0) {
  handleParamChange('units', value);
}
```
This prevents NaN from appearing in the UI when backspacing.

### Error Handling Pattern
Training errors flow from Python → Extension → WebView:
1. **Python validation** (`runner.py`): Validates architecture before training starts
   - Missing Input/Output layers
   - No trainable parameters
   - Invalid layer configurations
2. **Stderr capture** (`TrainingManager.ts`): Buffers all stderr output
3. **Error display** (`MetricsPanel.tsx`): Shows errors in prominent red banner with full traceback

Example Python validation:
```python
if num_params == 0:
    raise ValueError(
        "❌ Model has no trainable layers!\n\n"
        "Your model only contains Input/Output layers.\n"
        "Please add at least one trainable layer (Dense, Conv2D, etc.).\n\n"
        "Example: Input → Dense → Output"
    )
```

## Python Requirements

Install PyTorch and dependencies:
```bash
pip install torch numpy
```

For NPZ dataset support: NumPy is required (included above)

## Testing Training

Use MNIST NPZ format with keys:
- `X_train`, `y_train` (required)
- `X_test`/`X_val`, `y_test`/`y_val` (optional)

The loader automatically:
- Normalizes pixel values (0-255 → 0-1)
- Adds channel dimension (28×28 → 1×28×28)
- Converts to PyTorch format (NHWC → NCHW)
