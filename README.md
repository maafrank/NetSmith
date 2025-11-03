# NetSmith

# NetSmith — Visual Neural Network Designer for VS Code

## Overview
**NetSmith** is a VS Code extension that lets users **visually design, configure, and train neural networks locally**.  
It targets both technical and non-technical users, providing:
- A **drag-and-drop layer editor**
- **AI-assisted model design** from natural language
- Real-time **training visualization** powered by PyTorch or TensorFlow
- Integrated **project and run management**

All training runs execute on the user’s local hardware; NetSmith only orchestrates and visualizes results.  

---

## Goals
1. Provide an intuitive graphical interface for designing neural networks inside VS Code.  
2. Enable direct local execution (no external servers).  
3. Stream live training metrics (loss, accuracy, epoch, etc.) into a rich UI dashboard.  
4. Support AI-driven model suggestions from plain-English prompts.  
5. Support importing/exporting architectures and weights.  
6. Allow project and run tracking within the workspace.  

---

## Core Features

### 1. Visual Model Builder
- Built using **React Flow** inside a VS Code WebView.  
- Nodes represent layers (Dense, Conv2D, BatchNorm, Activation, etc.).  
- Connections define tensor flow.  
- Right-side property panel for layer configuration.  
- Group nodes into **Blocks** (e.g., residual module) reusable across models.  
- Export model to `model.py` (PyTorch) or `model.json` (Keras).

### 2. AI Model Assistant
- Natural-language → architecture suggestions using OpenAI GPT-5 API.  
- Example:  
  > “I want a convolutional network for image segmentation”  
  Produces a UNet-style architecture graph pre-populated in the builder.  
- Can also recommend hyperparameters, loss functions, or optimizers.  

### 3. Local Training Orchestration
- The extension spawns a **Python subprocess** using the user’s selected interpreter.  
- The Python backend:  
  - Parses architecture definition.  
  - Loads data from local workspace or user-specified directory.  
  - Runs training loop with live metric emission (via stdout or WebSocket).  
  - Saves model weights, metrics, and logs under `/runs/`.  
- Metrics streamed to the WebView for real-time charting.

### 4. Metric & Log Dashboard
- Real-time charts for loss, accuracy, and custom metrics.  
- Console log viewer for training output.  
- “Pause”, “Stop”, “Resume” training controls.  
- Run comparison view for analyzing multiple experiments.

### 5. Project & Run Management
Each workspace project includes:
```
.netsmith/
├─ config.json
├─ models/
│   └─ model_01.json
├─ runs/
│   ├─ run_001/
│   │   ├─ metrics.json
│   │   ├─ weights.pt
│   │   └─ logs.txt
└─ blocks/
    └─ resblock.json
```
- Configurable workspace-level settings.  
- Automatic indexing of previous runs and checkpoints.  

### 6. Weight Import & Reverse Reconstruction
- User can import `.pt`, `.pth`, or `.h5` weights.  
- NetSmith inspects the file to reconstruct a compatible layer graph.  
- Useful for understanding or fine-tuning pretrained models.  

### 7. Extension Commands
| Command | Action |
|----------|--------|
| `NetSmith: New Model` | Opens visual builder in a new WebView. |
| `NetSmith: Open Project` | Prompts for or auto-detects `.netsmith/` folder. |
| `NetSmith: Run Model` | Executes the selected model in the local Python environment. |
| `NetSmith: Stop Training` | Gracefully terminates running process. |
| `NetSmith: Ask AI for Architecture` | Opens chat input for AI-generated design. |

### 8. Integration with VS Code APIs
- **Python Extension:** reuse interpreter selection and debugging features.  
- **Terminal API:** run Python commands in background.  
- **WebView API:** custom React dashboard.  
- **FileSystem API:** read/write to workspace.  

---

## Architecture

```
┌──────────────────────────────────┐
│          VS Code Shell           │
│ (User selects workspace folder)  │
└────────────────┬─────────────────┘
                 │
     VS Code Extension Host (TypeScript)
                 │
     ┌───────────┴────────────┐
     │        WebView UI       │
     │ (React + React Flow)    │
     │ - Layer graph builder   │
     │ - Metric charts         │
     │ - AI assistant panel    │
     └───────────┬────────────┘
                 │ message passing (JSON)
     ┌───────────┴────────────┐
     │   Python Backend (local)│
     │ - PyTorch / TF training │
     │ - Metric streaming      │
     │ - Weight I/O            │
     └───────────┬────────────┘
                 │ stdout / WebSocket
     ┌───────────┴────────────┐
     │      Run Tracker        │
     │   (.netsmith/ files)    │
     └─────────────────────────┘
```

---

## Tech Stack
| Layer | Technology |
|--------|-------------|
| **UI Framework** | React + React Flow + Tailwind |
| **Charts** | Recharts / Chart.js |
| **Extension Runtime** | TypeScript (VS Code API) |
| **AI Assistant** | OpenAI GPT-5 API |
| **Local Training** | Python 3.10+ with PyTorch or TensorFlow |
| **IPC / Messaging** | VS Code `postMessage` API or WebSocket |
| **Project Storage** | JSON files under `.netsmith/` |

---

## MVP Roadmap

| Phase | Deliverables |
|-------|---------------|
| **Phase 1: Core Extension** | Command palette commands, WebView, React builder skeleton, export to `.py`. |
| **Phase 2: Python Bridge** | Run model locally, stream stdout to WebView, show metrics. |
| **Phase 3: AI Assistant** | Integrate GPT-based architecture generator. |
| **Phase 4: Project Tracking** | Add `.netsmith/` manifest, run logs, weight management. |
| **Phase 5: Prebuilt Models & Import** | Load standard architectures and parse weight files. |
| **Phase 6: Polishing** | UX improvements, run comparison, publish to VS Code Marketplace. |

---

## Future Extensions
- Plugin system for custom layers or training routines.  
- Cloud sync option for model configs (without training).  
- Integration with Hugging Face model zoo.  
- ONNX export/import support.  
- Lightweight local inference preview (e.g., image classification tester).  

---

## Licensing
Open-source under MIT License.  
Optional Pro tier may include cloud-sync or commercial AI-assistant tokens.

---

## Repository Structure (Target)

```
netsmith/
├─ package.json
├─ src/
│   ├─ extension.ts
│   ├─ webview/
│   │   ├─ index.html
│   │   ├─ main.tsx
│   │   ├─ components/
│   │   └─ react-flow/
│   └─ python/
│       ├─ runner.py
│       ├─ utils/
│       └─ templates/
├─ media/
│   └─ icons/
├─ assets/
│   └─ logo.svg
├─ README.md
└─ SPEC.md
```

---

## Tagline
> **NetSmith** — Forge neural networks visually, train them locally.

---

## Development Setup

### Prerequisites
- Node.js 18+
- Python 3.10+ with PyTorch
- VS Code

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd NetSmith
```

2. Install dependencies:
```bash
# Install extension dependencies
npm install

# Install webview dependencies
cd webview
npm install
cd ..
```

3. Install Python dependencies:
```bash
pip install -r src/python/requirements.txt
```

4. Build the project:
```bash
npm run build
```

### Running the Extension

1. Open the project in VS Code
2. Press `F5` to launch the Extension Development Host
3. In the new VS Code window, open the Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`)
4. Run the command: `NetSmith: New Model`
5. Start building your neural network!

### Quick Start Guide

1. **Create a new model**: `Cmd/Ctrl+Shift+P` → `NetSmith: New Model`
2. **Add layers**: Click on layers in the left palette to add them to your canvas
3. **Connect layers**: Drag from the bottom handle of one layer to the top handle of another
4. **Configure layers**: Click on a layer and edit its properties in the right panel
5. **Configure training**: Click the ⚙️ button in the top-right panel to set dataset path and hyperparameters
6. **Run training**: Click the ▶ Run Model button

### Project Structure

```
NetSmith/
├── src/                    # Extension source code
│   ├── extension.ts        # Main extension entry point
│   ├── types.ts            # Type definitions
│   ├── project/            # Project management
│   ├── training/           # Training orchestration
│   ├── webview/            # WebView panel
│   └── python/             # Python training backend
│       ├── runner.py       # PyTorch training script
│       └── requirements.txt
├── webview/                # React UI
│   ├── src/
│   │   ├── App.tsx         # Main app component
│   │   ├── components/     # UI components
│   │   ├── store.ts        # Zustand state management
│   │   └── types.ts        # WebView types
│   └── dist/               # Built webview assets
├── dist/                   # Compiled extension code
└── package.json

Generated .netsmith/ project:
.netsmith/
├── config.json             # Project configuration
├── models/                 # Saved model architectures
├── runs/                   # Training runs
│   └── run_*/
│       ├── architecture.json
│       ├── config.json
│       ├── metrics.json
│       └── weights.pt
└── blocks/                 # Reusable blocks
```

### Available Commands

- `NetSmith: New Model` - Open the visual model builder
- `NetSmith: Open Project` - Initialize/open a NetSmith project
- `NetSmith: Run Model` - Start training the current model
- `NetSmith: Stop Training` - Stop the current training run

### Development Commands

```bash
# Watch mode (auto-rebuild on changes)
npm run watch              # Watch extension
npm run watch:webview      # Watch webview

# Build
npm run build              # Build everything
npm run build:extension    # Build extension only
npm run build:webview      # Build webview only

# Run both watches in parallel
npm run dev
```

---