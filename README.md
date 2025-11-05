# NetSmith — Visual Neural Network Designer for VS Code

[![VS Marketplace Version](https://img.shields.io/visual-studio-marketplace/v/MatthewFrank.netsmith)](https://marketplace.visualstudio.com/items?itemName=MatthewFrank.netsmith)
[![Installs](https://img.shields.io/visual-studio-marketplace/i/MatthewFrank.netsmith)](https://marketplace.visualstudio.com/items?itemName=MatthewFrank.netsmith)

## Overview
**NetSmith** is a VS Code extension for **visually designing, configuring, and training neural networks locally**.
It provides:
- A **drag-and-drop layer editor** with intelligent auto-connection
- **Reusable block templates** (ResNet, U-Net, Transformers, etc.)
- Real-time **training visualization** powered by PyTorch
- **Export to PyTorch and ONNX** formats
- Integrated **project and run management**

All training runs execute on your local hardware using your configured Python interpreter.

## Getting Started

### Launch NetSmith

1. **Open the Command Palette**: Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. **Search for**: `NetSmith: New Model`
3. **Press Enter** to open the visual model builder

You can also access NetSmith commands from:
- The **Command Palette** (`Cmd/Ctrl+Shift+P`) - Search "NetSmith"
- The **NetSmith sidebar icon** in the Activity Bar (left side of VS Code)

### Your First Model

1. **Add layers**: Click layers in the left palette (they auto-connect)
2. **Configure**: Click any layer to edit its properties in the right panel
3. **Auto-arrange**: Click 🔄 to organize your network
4. **Set up training**: Click ⚙️ to configure dataset, epochs, and hyperparameters
5. **Run**: Click ▶️ to start training
6. **Watch**: View real-time metrics and loss curves

---

## Installation

### From VS Code Marketplace (Recommended)
1. Open VS Code
2. Press `Cmd/Ctrl+Shift+X` to open Extensions
3. Search for **"NetSmith"**
4. Click **Install**

Or install directly from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=MatthewFrank.netsmith).

### Prerequisites
- **Python 3.10+** with PyTorch 2.0+ and NumPy installed:
  ```bash
  pip install torch numpy
  ```
- A Python interpreter configured in VS Code

---

## Key Features

### Visual Model Builder
- Built with **React Flow** in a VS Code WebView
- **20+ layer types**: Dense, Conv2D, Conv1D, Pooling, BatchNorm, Dropout, Activations, Flatten, Reshape, UpSampling, and more
- **6 merge layer types** for skip connections: Add, Concat, Multiply, Subtract, Maximum, Minimum
- **Auto-connect**: New layers automatically connect to the last layer in your network
- **Auto-layout**: Automatic graph arrangement using Dagre algorithm
- Interactive property panel for layer configuration
- Resizable side panels (50px-600px) for flexible workspace

### Pre-built Block Templates
Create complex architectures instantly with 15+ built-in templates:
- **ResNet-style**: SkipConnection, Bottleneck, SEResNet
- **Mobile**: DepthwiseSeparable, InvertedResidual
- **Inception/SqueezeNet**: Fire, Inception
- **DenseNet**: Dense blocks with dense connectivity
- **U-Net**: UNetEncoder, UNetDecoder, AttentionGate
- **Transformers**: Attention, Transformer, TransformerEncoder, TransformerDecoder
- **Basic**: ConvBNRelu, SE (Squeeze-Excitation)

Blocks are expanded and flattened before training with intelligent edge rewiring.

### Local Training Orchestration
- Spawns **Python subprocess** using your VS Code Python interpreter
- **Lazy module initialization**: Automatic shape inference for Linear, Conv2D, and BatchNorm layers
- **NPZ dataset support** with automatic preprocessing:
  - Auto-normalization (0-255 → 0-1)
  - Channel dimension handling (HWC → CHW)
  - Automatic train/test splitting
- **Real-time metrics streaming** via stdout parsing
- Comprehensive **error validation** with helpful suggestions
- Saves architecture, config, metrics, and weights per run

### Metrics & Visualization
- Real-time charts for training/validation loss and accuracy
- Batch progress display (shows batch X/Y during training)
- Minimize/expand toggle for workspace flexibility
- Detailed error messages with full Python traceback
- Clean epoch-level visualization (no per-batch clutter)

### Export System
**PyTorch Export**: Generate standalone `.py` file with model class
- Uses lazy modules for automatic shape inference
- Includes usage example and forward pass code
- No trained weights required

**ONNX Export**: Export trained models for cross-framework compatibility
- Select from completed training runs
- Includes dynamic batch size support
- Requires `weights.pt` from completed run

### Project & Run Management
Workspace structure:
```
.netsmith/
├─ config.json              # Project settings
├─ models/
│   └─ model_*.json         # Saved architectures
├─ runs/
│   └─ run_*/
│       ├─ architecture.json
│       ├─ config.json
│       ├─ metrics.json
│       └─ weights.pt
└─ blocks/                  # Future: Custom reusable blocks
```

### Dataset Management
- **Automatic workspace scanning** for `.npz` files on startup
- **File picker integration** for manual dataset selection
- Relative path storage for project portability
- Support for pre-split data (`X_train`/`y_train`/`X_test`/`y_test`)

### Extension Commands
| Command | Action |
|----------|--------|
| `NetSmith: New Model` | Opens the visual model builder |
| `NetSmith: Open Project` | Initialize/open NetSmith project |
| `NetSmith: Run Model` | Start training with current architecture |
| `NetSmith: Stop Training` | Terminate running training process |  

---

## Architecture

### Three-Layer Communication Pattern

```
┌──────────────────────────────────────────┐
│         VS Code Extension Host           │
│           (TypeScript/Node.js)           │
│  - Commands & lifecycle management       │
│  - Python subprocess orchestration       │
│  - Project/run file I/O                  │
└──────────────┬───────────────────────────┘
               │ postMessage API
┌──────────────┴───────────────────────────┐
│           WebView UI (React)             │
│  - React Flow visual graph editor        │
│  - Layer palette & properties panel      │
│  - Training config & metrics dashboard   │
│  - Recharts visualization                │
└──────────────┬───────────────────────────┘
               │ stdin/stdout
┌──────────────┴───────────────────────────┐
│        Python Backend (runner.py)        │
│  - Dynamic PyTorch model builder         │
│  - NPZ dataset loader with preprocessing │
│  - Training loop with metric emission    │
└──────────────┬───────────────────────────┘
               │
┌──────────────┴───────────────────────────┐
│      .netsmith/ (Workspace Storage)      │
│  - JSON architecture & config files      │
│  - Trained weights (.pt)                 │
│  - Per-run metrics & logs                │
└──────────────────────────────────────────┘
```

### Message Flow Examples
- `ready` → Extension scans workspace for datasets
- `runModel` → Extension spawns Python with architecture JSON
- `trainingMetrics` → Extension streams epoch data to WebView
- `saveModel` → Extension writes to `.netsmith/models/`
- `pickDatasetFile` → Opens VS Code file picker dialog

---

## Tech Stack
| Component | Technology |
|-----------|------------|
| **Extension** | TypeScript, VS Code Extension API |
| **UI Framework** | React, React Flow, Zustand (state) |
| **Visualization** | Recharts |
| **Layout** | Dagre (auto-layout algorithm) |
| **Training** | Python 3.10+, PyTorch 2.0+ |
| **Datasets** | NumPy (NPZ format) |
| **Export** | PyTorch script export, ONNX export |
| **IPC** | VS Code `postMessage` API, Python stdout |

---

## Future Enhancements
- **Custom block saving**: Save your own reusable block templates
- **ONNX import**: Load pre-trained ONNX models
- **Additional dataset formats**: PyTorch tensors (.pt), image folders, HDF5
- **Inference mode**: Test trained models directly in the UI
- **Hyperparameter search**: Grid search and Bayesian optimization
- **Model comparison**: Side-by-side run analysis
- **TensorBoard integration**: Export metrics for TensorBoard visualization

---

## Tech Notes

### Lazy Module Pattern
NetSmith uses PyTorch's lazy modules (`LazyLinear`, `LazyConv2d`, `LazyBatchNorm2d`) to automatically infer layer dimensions:
```python
# No need to specify input features - PyTorch figures it out!
nn.LazyLinear(128)  # Input size determined on first forward pass
```

### Block Flattening
Block nodes are recursively flattened before training with intelligent edge rewiring:
- Block → Block: Connects last internal node of source to first of target
- Block → Layer: Uses last internal node as connection source
- Layer → Block: Uses first internal node as target (with skip connections for Add layers)

### State Management
React Flow is the source of truth for node positions. The Zustand store is only synced when needed (save/export/train) to avoid infinite loops.

---

## Contributing & Development

Contributions are welcome! See our [GitHub repository](https://github.com/maafrank/NetSmith) for issues and pull requests.

### Development Setup

#### Prerequisites
- **Node.js 18+** and npm
- **Python 3.10+** with PyTorch 2.0+
- **VS Code** (latest stable version)

#### Installation

```bash
# 1. Clone and navigate
git clone https://github.com/maafrank/NetSmith.git
cd NetSmith

# 2. Install extension dependencies
npm install

# 3. Install webview dependencies
cd webview && npm install && cd ..

# 4. Install Python dependencies
pip install torch numpy

# 5. Build the extension
npm run build
```

### Running the Extension

1. Open the project in VS Code
2. Press **F5** to launch Extension Development Host
3. In the new window: **Cmd/Ctrl+Shift+P** → `NetSmith: New Model`

### Quick Start Guide

1. **Add layers**: Click layers in the left palette (auto-connects to previous layer)
2. **Or add blocks**: Click block templates for complex architectures
3. **Configure layers**: Click a node, edit properties in the right panel
4. **Auto-layout**: Click 🔄 to arrange nodes automatically
5. **Configure training**: Click ⚙️ in the training panel
   - Select dataset from dropdown or click 📁 to browse
   - Set epochs, batch size, learning rate, optimizer
6. **Run**: Click ▶️ **Run Model** button
7. **View metrics**: Watch real-time loss/accuracy charts
8. **Export**: Click 📤 to export as PyTorch script or ONNX

### Development Commands

```bash
# Development (watch mode with hot reload)
npm run dev                # Watch both extension and webview

# Individual watches
npm run watch              # Extension only
npm run watch:webview      # Webview only (Vite dev server)

# Build
npm run build              # Build everything
npm run build:extension    # TypeScript compilation
npm run build:webview      # React/Vite build
```

**Note**: Extension code changes require reloading the Extension Development Host (Cmd/Ctrl+R). Webview changes hot-reload automatically in watch mode.

---

## Project Structure

```
NetSmith/
├── src/                          # Extension source (TypeScript)
│   ├── extension.ts              # Main entry point, command registration
│   ├── types.ts                  # Shared type definitions
│   ├── webview/
│   │   └── ModelBuilderPanel.ts # WebView manager & message handler
│   ├── project/
│   │   └── ProjectManager.ts    # .netsmith/ file operations
│   ├── training/
│   │   ├── TrainingManager.ts   # Python subprocess orchestration
│   │   └── blockFlattener.ts    # Block → layer expansion logic
│   ├── export/
│   │   ├── pythonExporter.ts    # PyTorch script generation
│   │   └── onnxExporter.ts      # ONNX export coordinator
│   └── python/
│       ├── runner.py             # PyTorch training script
│       ├── onnx_exporter.py      # ONNX export helper
│       └── requirements.txt
│
├── webview/                      # React UI (TypeScript + Vite)
│   ├── src/
│   │   ├── App.tsx               # Main React Flow canvas
│   │   ├── store.ts              # Zustand state management
│   │   ├── types.ts              # WebView type definitions
│   │   ├── blockTemplates.ts    # Pre-built block architectures
│   │   └── components/
│   │       ├── LayerPalette.tsx  # Draggable layer list
│   │       ├── PropertiesPanel.tsx # Layer config editor
│   │       ├── TrainingPanel.tsx   # Hyperparameter config
│   │       ├── MetricsPanel.tsx    # Real-time charts
│   │       └── nodes/
│   │           └── index.tsx       # Custom React Flow nodes
│   ├── dist/                     # Built webview bundle
│   └── index.html                # WebView HTML template
│
├── dist/                         # Compiled extension JS
└── package.json                  # Extension manifest
```

**Generated per workspace:**
```
.netsmith/
├── config.json                   # Project settings
├── models/
│   └── model_20250104_143022.json # Saved architectures
├── runs/
│   └── run_20250104_143045/
│       ├── architecture.json      # Model definition
│       ├── config.json            # Training hyperparameters
│       ├── metrics.json           # Final epoch metrics
│       └── weights.pt             # Trained model weights
└── blocks/                       # (Future) Custom blocks
```

---

## Testing with MNIST

Download MNIST in NPZ format:
```python
import numpy as np
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
np.savez('mnist.npz',
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test)
```

Place `mnist.npz` in your workspace and it will appear in the dataset dropdown.

---

## Support & Feedback

- **Issues**: [GitHub Issues](https://github.com/maafrank/NetSmith/issues)
- **Marketplace**: [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=MatthewFrank.netsmith)
- **Repository**: [GitHub Repository](https://github.com/maafrank/NetSmith)

## License

See [LICENSE.txt](LICENSE.txt) for details.

---