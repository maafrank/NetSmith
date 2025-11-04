import { useState } from 'react';
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
  const { isTraining, trainingConfig, setTrainingConfig, clearMetrics } = useStore();
  const [showConfig, setShowConfig] = useState(false);

  const handleRunModel = () => {
    if (nodes.length === 0) {
      alert('Please add layers to your model first');
      return;
    }

    if (!trainingConfig.datasetPath) {
      alert('Please configure dataset path');
      setShowConfig(true);
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
    onBeforeRun(); // Sync state before saving
    vscode.postMessage({
      type: 'saveModel',
      data: { nodes, edges, blocks: [] },
    });
  };

  const handleExport = () => {
    onBeforeRun(); // Sync state before exporting
    vscode.postMessage({
      type: 'exportModel',
      format: 'pytorch',
    });
  };

  const handlePickDataset = () => {
    vscode.postMessage({ type: 'pickDatasetFile' });
  };

  return (
    <div className="min-w-[300px]">
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
          onClick={() => setShowConfig(!showConfig)}
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

      {showConfig && (
        <div className="mt-4 p-4 bg-gray-900 rounded-lg max-h-[400px] overflow-y-auto">
          <h3 className="font-bold text-white mb-3">Training Configuration</h3>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">
                Dataset Path
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={trainingConfig.datasetPath}
                  onChange={(e) => setTrainingConfig({ datasetPath: e.target.value })}
                  placeholder="/path/to/dataset.npz"
                  className="flex-1 px-2 py-1 text-sm bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
                />
                <button
                  onClick={handlePickDataset}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded"
                  title="Browse files"
                >
                  📁
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Supported: .npz (NumPy arrays), .pt (PyTorch), folders with images
              </p>
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
