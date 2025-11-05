import { useState, useEffect, useRef } from 'react';
import { useStore } from '../store';
import { vscode } from '../vscode';
import { OptimizerType, LossType, LayerNode, Edge } from '../types';

const optimizers: OptimizerType[] = ['adam', 'sgd', 'rmsprop', 'adamw', 'adagrad'];
const losses: LossType[] = ['cross_entropy', 'mse', 'mae', 'binary_cross_entropy', 'huber'];

interface TrainingPanelProps {
  onBeforeRun: () => void;
  nodes: LayerNode[];
  edges: Edge[];
}

export default function TrainingPanel({ onBeforeRun, nodes, edges }: TrainingPanelProps) {
  const { isTraining, trainingConfig, setTrainingConfig, clearMetrics, showTrainingConfig, setShowTrainingConfig } = useStore();
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [showSaveInput, setShowSaveInput] = useState(false);
  const [modelName, setModelName] = useState('my-model');
  const [showExportOptions, setShowExportOptions] = useState(false);
  const [showModelSelection, setShowModelSelection] = useState(false);
  const [availableRuns, setAvailableRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>('');
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const message = event.data;
      if (message.type === 'availableDatasets') {
        setAvailableDatasets(message.datasets);
      } else if (message.type === 'availableRuns') {
        setAvailableRuns(message.runs);
        if (message.runs.length > 0) {
          setShowModelSelection(true);
        } else {
          alert('No trained models found. Please complete training first.');
        }
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(event.target as Node)) {
        setShowTrainingConfig(false);
      }
    };

    if (showTrainingConfig) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showTrainingConfig, setShowTrainingConfig]);

  const handleRunModel = () => {
    if (nodes.length === 0) {
      alert('Please add layers to your model first');
      return;
    }

    if (!trainingConfig.datasetPath) {
      alert('Please configure dataset path');
      setShowTrainingConfig(true);
      return;
    }

    // Sync React Flow state to store before running
    onBeforeRun();

    clearMetrics();

    vscode.postMessage({
      type: 'runModel',
      architecture: { nodes, edges, blocks: [] },
      config: trainingConfig,
    });
  };

  const handleStopTraining = () => {
    vscode.postMessage({ type: 'stopTraining' });
  };

  const handleSaveModel = () => {
    setShowSaveInput(true);
  };

  const handleConfirmSave = () => {
    if (!modelName.trim()) {
      alert('Model name cannot be empty');
      return;
    }
    if (/[<>:"/\\|?*]/.test(modelName)) {
      alert('Model name contains invalid characters');
      return;
    }

    onBeforeRun(); // Sync state before saving
    vscode.postMessage({
      type: 'saveModel',
      data: { nodes, edges, blocks: [] },
      modelName: modelName.trim(),
    });
    setShowSaveInput(false);
  };

  const handleCancelSave = () => {
    setShowSaveInput(false);
    setModelName('my-model');
  };

  const handleExport = () => {
    if (nodes.length === 0) {
      alert('Please add layers to your model first');
      return;
    }
    setShowExportOptions(true);
  };

  const handleExportFormat = (format: 'pytorch' | 'onnx') => {
    if (format === 'pytorch') {
      // PyTorch export doesn't need model selection
      onBeforeRun();
      vscode.postMessage({
        type: 'exportModel',
        format: 'pytorch',
        data: { nodes, edges, blocks: [] },
      });
      setShowExportOptions(false);
    } else if (format === 'onnx') {
      // ONNX needs model selection, request available runs
      setShowExportOptions(false);
      vscode.postMessage({ type: 'requestAvailableRuns' });
    }
  };

  const handleCancelExport = () => {
    setShowExportOptions(false);
  };

  const handleConfirmOnnxExport = () => {
    if (!selectedRun) {
      alert('Please select a trained model');
      return;
    }

    onBeforeRun();
    vscode.postMessage({
      type: 'exportModel',
      format: 'onnx',
      data: { nodes, edges, blocks: [] },
      selectedRun: selectedRun,
    });
    setShowModelSelection(false);
    setSelectedRun('');
  };

  const handleCancelModelSelection = () => {
    setShowModelSelection(false);
    setSelectedRun('');
  };

  const handlePickDataset = () => {
    vscode.postMessage({ type: 'pickDatasetFile' });
  };

  return (
    <div ref={panelRef} className="min-w-[300px]">
      <div className="flex gap-2 mb-4">
        {!isTraining ? (
          <button
            onClick={handleRunModel}
            className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded font-medium"
          >
            ▶ Run Model
          </button>
        ) : (
          <button
            onClick={handleStopTraining}
            className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded font-medium"
          >
            ⏹ Stop Training
          </button>
        )}

        <button
          onClick={() => setShowTrainingConfig(!showTrainingConfig)}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded"
        >
          ⚙️
        </button>
      </div>

      <div className="flex gap-2">
        <button
          onClick={handleSaveModel}
          className="flex-1 px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
        >
          💾 Save
        </button>
        <button
          onClick={handleExport}
          className="flex-1 px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
        >
          📤 Export
        </button>
      </div>

      {showSaveInput && (
        <div className="mt-2 p-3 bg-gray-900 rounded-lg border border-blue-500">
          <label className="block text-xs font-medium text-gray-300 mb-2">
            Model Name
          </label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleConfirmSave();
              if (e.key === 'Escape') handleCancelSave();
            }}
            placeholder="my-model"
            autoFocus
            className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none mb-2"
          />
          <div className="flex gap-2">
            <button
              onClick={handleConfirmSave}
              className="flex-1 px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded"
            >
              ✓ Save
            </button>
            <button
              onClick={handleCancelSave}
              className="flex-1 px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
            >
              ✕ Cancel
            </button>
          </div>
        </div>
      )}

      {showExportOptions && (
        <div className="mt-2 p-3 bg-gray-900 rounded-lg border border-purple-500">
          <label className="block text-xs font-medium text-gray-300 mb-2">
            Export Format
          </label>
          <div className="space-y-2">
            <button
              onClick={() => handleExportFormat('pytorch')}
              className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded text-left"
            >
              <div className="font-medium">🐍 PyTorch (.py)</div>
              <div className="text-xs text-gray-300 mt-0.5">Standalone Python script</div>
            </button>
            <button
              onClick={() => handleExportFormat('onnx')}
              className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded text-left"
            >
              <div className="font-medium">🔄 ONNX (.onnx)</div>
              <div className="text-xs text-gray-300 mt-0.5">Cross-framework format (requires trained weights)</div>
            </button>
          </div>
          <button
            onClick={handleCancelExport}
            className="w-full mt-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
          >
            ✕ Cancel
          </button>
        </div>
      )}

      {showModelSelection && (
        <div className="mt-2 p-3 bg-gray-900 rounded-lg border border-purple-500">
          <label className="block text-xs font-medium text-gray-300 mb-2">
            Select Trained Model
          </label>
          <select
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-purple-500 focus:outline-none mb-2"
          >
            <option value="">Choose a model...</option>
            {availableRuns.map((run) => (
              <option key={run} value={run}>
                {run}
              </option>
            ))}
          </select>
          <div className="flex gap-2">
            <button
              onClick={handleConfirmOnnxExport}
              className="flex-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded"
            >
              ✓ Export ONNX
            </button>
            <button
              onClick={handleCancelModelSelection}
              className="flex-1 px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
            >
              ✕ Cancel
            </button>
          </div>
        </div>
      )}

      {showTrainingConfig && (
        <div className="mt-4 p-4 bg-gray-900 rounded-lg max-h-[400px] overflow-y-auto">
          <h3 className="font-bold text-white mb-3">Training Configuration</h3>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">
                Dataset Path
              </label>
              <div className="flex gap-2">
                {availableDatasets.length > 0 ? (
                  <select
                    value={trainingConfig.datasetPath}
                    onChange={(e) => setTrainingConfig({ datasetPath: e.target.value })}
                    className="flex-1 px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
                  >
                    <option value="">Select dataset...</option>
                    {availableDatasets.map((path) => (
                      <option key={path} value={path}>
                        {path}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={trainingConfig.datasetPath}
                    onChange={(e) => setTrainingConfig({ datasetPath: e.target.value })}
                    placeholder="data/mnist_decoded.npz"
                    className="flex-1 px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
                  />
                )}
                <button
                  onClick={handlePickDataset}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded"
                  title="Browse files"
                >
                  📁
                </button>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">
                Dataset Type
              </label>
              <select
                value={trainingConfig.datasetType}
                onChange={(e) =>
                  setTrainingConfig({ datasetType: e.target.value as 'full' | 'split' })
                }
                className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              >
                <option value="full">Full (auto-split)</option>
                <option value="split">Pre-split</option>
              </select>
            </div>

            {trainingConfig.datasetType === 'split' && (
              <>
                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-1">
                    Train Path
                  </label>
                  <input
                    type="text"
                    value={trainingConfig.trainPath || ''}
                    onChange={(e) => setTrainingConfig({ trainPath: e.target.value })}
                    className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-1">Val Path</label>
                  <input
                    type="text"
                    value={trainingConfig.valPath || ''}
                    onChange={(e) => setTrainingConfig({ valPath: e.target.value })}
                    className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-1">
                    Test Path
                  </label>
                  <input
                    type="text"
                    value={trainingConfig.testPath || ''}
                    onChange={(e) => setTrainingConfig({ testPath: e.target.value })}
                    className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
                  />
                </div>
              </>
            )}

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">Batch Size</label>
                <input
                  type="number"
                  value={trainingConfig.batchSize}
                  onChange={(e) => setTrainingConfig({ batchSize: parseInt(e.target.value) })}
                  className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">Epochs</label>
                <input
                  type="number"
                  value={trainingConfig.epochs}
                  onChange={(e) => setTrainingConfig({ epochs: parseInt(e.target.value) })}
                  className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                value={trainingConfig.learningRate}
                onChange={(e) => setTrainingConfig({ learningRate: parseFloat(e.target.value) })}
                className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Optimizer</label>
              <select
                value={trainingConfig.optimizer}
                onChange={(e) => setTrainingConfig({ optimizer: e.target.value as OptimizerType })}
                className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              >
                {optimizers.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Loss</label>
              <select
                value={trainingConfig.loss}
                onChange={(e) => setTrainingConfig({ loss: e.target.value as LossType })}
                className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              >
                {losses.map((loss) => (
                  <option key={loss} value={loss}>
                    {loss}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Device</label>
              <select
                value={trainingConfig.device}
                onChange={(e) =>
                  setTrainingConfig({
                    device: e.target.value as 'cpu' | 'cuda' | 'mps' | 'auto',
                  })
                }
                className="w-full px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              >
                <option value="auto">Auto</option>
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA (NVIDIA)</option>
                <option value="mps">MPS (Apple Silicon)</option>
              </select>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
