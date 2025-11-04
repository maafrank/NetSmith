import { useStore } from '../store';
import { ActivationType } from '../types';
import { useState, useEffect } from 'react';

const activationOptions: ActivationType[] = [
  'relu',
  'sigmoid',
  'tanh',
  'softmax',
  'leaky_relu',
  'elu',
  'selu',
  'gelu',
  'swish',
  'linear',
];

interface PropertiesPanelProps {
  onUpdateNode: (nodeId: string, newData: any) => void;
}

export default function PropertiesPanel({ onUpdateNode }: PropertiesPanelProps) {
  const { selectedNode, updateNode, deleteNode } = useStore();

  // Local state for all text inputs to allow free editing
  const [inputShapeText, setInputShapeText] = useState('');
  const [unitsText, setUnitsText] = useState('');
  const [filtersText, setFiltersText] = useState('');
  const [kernelSizeText, setKernelSizeText] = useState('');
  const [poolSizeText, setPoolSizeText] = useState('');
  const [dropoutText, setDropoutText] = useState('');
  const [momentumText, setMomentumText] = useState('');
  const [epsilonText, setEpsilonText] = useState('');
  const [outputUnitsText, setOutputUnitsText] = useState('');

  // Sync all text inputs with selectedNode params
  useEffect(() => {
    if (!selectedNode) return;

    const { layerType, params } = selectedNode.data;

    if (layerType === 'Input') {
      setInputShapeText(params.inputShape?.join(',') || '');
    }
    if (layerType === 'Dense') {
      setUnitsText(String(params.units || 128));
    }
    if (layerType === 'Conv2D' || layerType === 'Conv1D') {
      setFiltersText(String(params.filters || 32));
      const kernelSize = Array.isArray(params.kernelSize) ? params.kernelSize[0] : params.kernelSize || 3;
      setKernelSizeText(String(kernelSize));
    }
    if (layerType === 'MaxPool2D' || layerType === 'AvgPool2D') {
      const poolSize = Array.isArray(params.poolSize) ? params.poolSize[0] : params.poolSize || 2;
      setPoolSizeText(String(poolSize));
    }
    if (layerType === 'Dropout') {
      setDropoutText(String(params.rate || 0.5));
    }
    if (layerType === 'BatchNorm') {
      setMomentumText(String(params.momentum || 0.99));
      setEpsilonText(String(params.epsilon || 0.001));
    }
    if (layerType === 'Output') {
      setOutputUnitsText(String(params.units || 10));
    }
  }, [selectedNode?.id, selectedNode?.data]);

  if (!selectedNode) {
    return (
      <div className="p-4">
        <h2 className="text-lg font-bold text-white mb-4">Properties</h2>
        <p className="text-sm text-gray-400">Select a layer to edit its properties</p>
      </div>
    );
  }

  const { id, data } = selectedNode;
  const { layerType, params } = data;

  const handleParamChange = (key: string, value: any) => {
    const newData = {
      params: {
        ...params,
        [key]: value,
      },
    };

    // Update both store and React Flow
    updateNode(id, newData);
    onUpdateNode(id, newData);
  };

  const handleDelete = () => {
    deleteNode(id);
  };

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold text-white">Properties</h2>
        <button
          onClick={handleDelete}
          className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded"
        >
          Delete
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Layer Type</label>
          <div className="text-white font-bold">{layerType}</div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Name</label>
          <input
            type="text"
            value={params.name || ''}
            onChange={(e) => handleParamChange('name', e.target.value)}
            placeholder={`${layerType} layer`}
            className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
          />
        </div>

        {/* Layer-specific parameters */}
        {layerType === 'Input' && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Input Shape (e.g., 28,28,1)
            </label>
            <input
              type="text"
              value={inputShapeText}
              onChange={(e) => {
                const value = e.target.value;
                setInputShapeText(value);

                if (!value.trim()) {
                  handleParamChange('inputShape', []);
                  return;
                }

                // Parse and validate
                const parsed = value.split(',').map((v) => {
                  const trimmed = v.trim();
                  if (trimmed === '') return null;
                  const num = parseInt(trimmed);
                  return isNaN(num) ? null : num;
                });

                // Only update if all parts are valid positive numbers
                if (parsed.every(n => n !== null && n > 0)) {
                  handleParamChange('inputShape', parsed as number[]);
                }
              }}
              placeholder="28,28,1"
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {layerType === 'Dense' && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Units</label>
            <input
              type="text"
              value={unitsText}
              onChange={(e) => {
                const value = e.target.value;
                setUnitsText(value);
                const num = parseInt(value);
                if (!isNaN(num) && num > 0) {
                  handleParamChange('units', num);
                }
              }}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {(layerType === 'Conv2D' || layerType === 'Conv1D') && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Filters</label>
              <input
                type="text"
                value={filtersText}
                onChange={(e) => {
                  const value = e.target.value;
                  setFiltersText(value);
                  const num = parseInt(value);
                  if (!isNaN(num) && num > 0) {
                    handleParamChange('filters', num);
                  }
                }}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Kernel Size</label>
              <input
                type="text"
                value={kernelSizeText}
                onChange={(e) => {
                  const value = e.target.value;
                  setKernelSizeText(value);
                  const num = parseInt(value);
                  if (!isNaN(num) && num > 0) {
                    handleParamChange('kernelSize', num);
                  }
                }}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Padding</label>
              <select
                value={params.padding || 'same'}
                onChange={(e) => handleParamChange('padding', e.target.value)}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              >
                <option value="same">Same</option>
                <option value="valid">Valid</option>
              </select>
            </div>
          </>
        )}

        {(layerType === 'MaxPool2D' || layerType === 'AvgPool2D') && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Pool Size</label>
            <input
              type="text"
              value={poolSizeText}
              onChange={(e) => {
                const value = e.target.value;
                setPoolSizeText(value);
                const num = parseInt(value);
                if (!isNaN(num) && num > 0) {
                  handleParamChange('poolSize', num);
                }
              }}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {layerType === 'Dropout' && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Dropout Rate</label>
            <input
              type="text"
              value={dropoutText}
              onChange={(e) => {
                const value = e.target.value;
                setDropoutText(value);
                const num = parseFloat(value);
                if (!isNaN(num) && num >= 0 && num <= 1) {
                  handleParamChange('rate', num);
                }
              }}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {layerType === 'BatchNorm' && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Momentum</label>
              <input
                type="text"
                value={momentumText}
                onChange={(e) => {
                  const value = e.target.value;
                  setMomentumText(value);
                  const num = parseFloat(value);
                  if (!isNaN(num) && num >= 0 && num <= 1) {
                    handleParamChange('momentum', num);
                  }
                }}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Epsilon</label>
              <input
                type="text"
                value={epsilonText}
                onChange={(e) => {
                  const value = e.target.value;
                  setEpsilonText(value);
                  const num = parseFloat(value);
                  if (!isNaN(num) && num >= 0) {
                    handleParamChange('epsilon', num);
                  }
                }}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </>
        )}

        {layerType === 'Activation' && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Activation</label>
            <select
              value={params.activation || 'relu'}
              onChange={(e) => handleParamChange('activation', e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            >
              {activationOptions.map((act) => (
                <option key={act} value={act}>
                  {act}
                </option>
              ))}
            </select>
          </div>
        )}

        {layerType === 'Output' && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Units (Classes)</label>
              <input
                type="text"
                value={outputUnitsText}
                onChange={(e) => {
                  const value = e.target.value;
                  setOutputUnitsText(value);
                  const num = parseInt(value);
                  if (!isNaN(num) && num > 0) {
                    handleParamChange('units', num);
                  }
                }}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
              <p className="text-xs text-gray-500 mt-1">Number of output classes</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Activation</label>
              <select
                value={params.activation || 'softmax'}
                onChange={(e) => handleParamChange('activation', e.target.value)}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              >
                {activationOptions.map((act) => (
                  <option key={act} value={act}>
                    {act}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">Typically softmax for classification, linear for regression</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
