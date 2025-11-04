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
- `runModel` (webview → extension): Trigger training with architecture + config
- `trainingMetrics` (extension → webview): Stream batch/epoch metrics
- `pickDatasetFile` (webview → extension): Open file picker dialog
- `datasetPathSelected` (extension → webview): Return selected file path

### State Management Critical Pattern

**React Flow is the source of truth** for node positions and data. The Zustand store is only synced when needed (save/export).

```typescript
// App.tsx sync pattern
const syncToStore = useCallback(() => {
  setNodes(rfNodes as any);
  setEdges(rfEdges as any);
}, [rfNodes, rfEdges, setNodes, setEdges]);

// Only sync store → React Flow on length changes (new nodes added)
useEffect(() => {
  if (nodes.length > rfNodes.length) {
    setRfNodes(nodes as any);
  }
}, [nodes.length]);
```

**Do not create bidirectional syncing** - this causes infinite loops. Always call `syncToStore()` before operations that need store data (like running training).

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
- Implements file picker for dataset selection

### TrainingManager (`src/training/TrainingManager.ts`)
- Spawns Python subprocess with `runner.py`
- Parses `METRICS:{json}` from stdout for real-time updates
- Integrates with Python extension API to get active interpreter
- Implements topological sort for layer ordering

### Python Runner (`src/python/runner.py`)
- `DynamicModel` class builds PyTorch `nn.Module` from JSON
- Uses `LazyLinear` and `LazyConv2d` for automatic input shape inference
- `load_dataset()` supports NPZ format with:
  - Pre-split data (`X_train`, `y_train`, `X_test`, `y_test`)
  - Auto-normalization (0-255 → 0-1)
  - Channel dimension handling (HWC → CHW for PyTorch)
  - Automatic splitting if single dataset provided
- Emits metrics every 10 batches during training

### Zustand Store (`webview/src/store.ts`)
- Manages application state (nodes, edges, training config, metrics)
- **Important**: `updateNode` must update both `nodes` array AND `selectedNode` to keep PropertiesPanel in sync

### Layer Components (`webview/src/components/`)
- **LayerPalette**: Drag-and-drop layer creation
- **PropertiesPanel**: Layer configuration (calls both `updateNode` in store AND `onUpdateNode` callback for React Flow)
- **TrainingPanel**: Hyperparameter configuration and run controls
- **MetricsPanel**: Real-time training visualization with batch progress and minimize toggle

## Type Definitions

Types are defined in **two locations**:
1. `src/types.ts` - Extension-side types (canonical)
2. `webview/src/types.ts` - WebView-side types (should mirror extension types)

**Always update both** when adding fields like `batch` and `totalBatches` to `TrainingMetrics`.

## Layer System

Supported layer types (in `LayerType` union):
- Input, Dense, Conv2D, Conv1D
- MaxPool2D, AvgPool2D, Flatten
- Dropout, BatchNorm, Activation, Output
- Block (for reusable modules - future feature)

Each layer has:
- Visual node representation (`webview/src/components/nodes/index.tsx`)
- Parameter schema in `LayerParams` interface
- Python mapping in `DynamicModel._create_layer()`

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

## Important Implementation Details

### Edge Arrows
Use `MarkerType.ArrowClosed` from React Flow:
```typescript
const defaultEdgeOptions = {
  markerEnd: { type: MarkerType.ArrowClosed },
};
```

### Auto-Layout
Uses Dagre graph layout library with top-to-bottom orientation:
```typescript
dagreGraph.setGraph({ rankdir: 'TB', nodesep: 100, ranksep: 150 });
```

### Batch Progress Display
Python emits batch-level metrics every 10 batches:
```python
if (batch_idx + 1) % 10 == 0:
    emit_metrics(epoch, loss, metrics=..., batch=batch_idx+1, total_batches=len(train_loader))
```

### Input Validation
All numeric inputs in `PropertiesPanel.tsx` must validate before updating:
```typescript
const value = parseInt(e.target.value);
if (!isNaN(value) && value > 0) {
  handleParamChange('units', value);
}
```
This prevents NaN from appearing in the UI when backspacing.

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
