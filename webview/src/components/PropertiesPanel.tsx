import { useStore } from '../store';
import { ActivationType } from '../types';

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

export default function PropertiesPanel() {
  const { selectedNode, updateNode, deleteNode } = useStore();

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
    updateNode(id, {
      params: {
        ...params,
        [key]: value,
      },
    });
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
              value={params.inputShape?.join(',') || ''}
              onChange={(e) =>
                handleParamChange(
                  'inputShape',
                  e.target.value.split(',').map((v) => parseInt(v.trim()))
                )
              }
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {layerType === 'Dense' && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Units</label>
            <input
              type="number"
              value={params.units || 128}
              onChange={(e) => handleParamChange('units', parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {(layerType === 'Conv2D' || layerType === 'Conv1D') && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Filters</label>
              <input
                type="number"
                value={params.filters || 32}
                onChange={(e) => handleParamChange('filters', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Kernel Size</label>
              <input
                type="number"
                value={
                  Array.isArray(params.kernelSize) ? params.kernelSize[0] : params.kernelSize || 3
                }
                onChange={(e) => handleParamChange('kernelSize', parseInt(e.target.value))}
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
              type="number"
              value={
                Array.isArray(params.poolSize) ? params.poolSize[0] : params.poolSize || 2
              }
              onChange={(e) => handleParamChange('poolSize', parseInt(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {layerType === 'Dropout' && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Dropout Rate</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={params.rate || 0.5}
              onChange={(e) => handleParamChange('rate', parseFloat(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {layerType === 'BatchNorm' && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Momentum</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={params.momentum || 0.99}
                onChange={(e) => handleParamChange('momentum', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Epsilon</label>
              <input
                type="number"
                step="0.0001"
                min="0"
                value={params.epsilon || 0.001}
                onChange={(e) => handleParamChange('epsilon', parseFloat(e.target.value))}
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
      </div>
    </div>
  );
}
