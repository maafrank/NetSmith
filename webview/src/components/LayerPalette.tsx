import { LayerType } from '../types';
import { useStore } from '../store';

const layerCategories = {
  'Core Layers': [
    { type: 'Input' as LayerType, icon: '⬇️', description: 'Input layer' },
    { type: 'Dense' as LayerType, icon: '🔲', description: 'Fully connected' },
    { type: 'Output' as LayerType, icon: '⬆️', description: 'Output layer' },
  ],
  'Convolutional': [
    { type: 'Conv2D' as LayerType, icon: '🔳', description: '2D convolution' },
    { type: 'Conv1D' as LayerType, icon: '📊', description: '1D convolution' },
  ],
  'Pooling': [
    { type: 'MaxPool2D' as LayerType, icon: '⬇️', description: 'Max pooling 2D' },
    { type: 'AvgPool2D' as LayerType, icon: '📉', description: 'Average pooling 2D' },
  ],
  'Regularization': [
    { type: 'Dropout' as LayerType, icon: '🎲', description: 'Dropout' },
    { type: 'BatchNorm' as LayerType, icon: '📏', description: 'Batch normalization' },
  ],
  'Activation': [
    { type: 'Activation' as LayerType, icon: '⚡', description: 'Activation function' },
  ],
  'Utility': [
    { type: 'Flatten' as LayerType, icon: '📄', description: 'Flatten' },
  ],
};

export default function LayerPalette() {
  const { addNode, nodes } = useStore();

  const handleAddLayer = (layerType: LayerType) => {
    const nodeId = `${layerType.toLowerCase()}_${Date.now()}`;
    const position = {
      x: Math.random() * 500 + 100,
      y: Math.random() * 500 + 100,
    };

    const defaultParams: any = {};

    // Set defaults based on layer type
    switch (layerType) {
      case 'Dense':
        defaultParams.units = 128;
        break;
      case 'Conv2D':
        defaultParams.filters = 32;
        defaultParams.kernelSize = 3;
        defaultParams.padding = 'same';
        break;
      case 'Conv1D':
        defaultParams.filters = 32;
        defaultParams.kernelSize = 3;
        defaultParams.padding = 'same';
        break;
      case 'Dropout':
        defaultParams.rate = 0.5;
        break;
      case 'Activation':
        defaultParams.activation = 'relu';
        break;
      case 'MaxPool2D':
      case 'AvgPool2D':
        defaultParams.poolSize = 2;
        break;
      case 'BatchNorm':
        defaultParams.momentum = 0.99;
        defaultParams.epsilon = 0.001;
        break;
      case 'Input':
        defaultParams.inputShape = [28, 28, 1];
        break;
    }

    addNode({
      id: nodeId,
      type: layerType,
      position,
      data: {
        label: layerType,
        layerType,
        params: defaultParams,
      },
    });
  };

  return (
    <div className="p-4">
      <h2 className="text-lg font-bold text-white mb-4">Layer Palette</h2>

      {Object.entries(layerCategories).map(([category, layers]) => (
        <div key={category} className="mb-6">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">{category}</h3>
          <div className="space-y-2">
            {layers.map((layer) => (
              <button
                key={layer.type}
                onClick={() => handleAddLayer(layer.type)}
                className="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-md transition-colors group"
              >
                <div className="flex items-center gap-2">
                  <span className="text-lg">{layer.icon}</span>
                  <div className="flex-1">
                    <div className="text-sm font-medium text-white">{layer.type}</div>
                    <div className="text-xs text-gray-400">{layer.description}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      ))}

      <div className="mt-6 pt-6 border-t border-gray-700">
        <div className="text-xs text-gray-500">
          Total layers: {nodes.length}
        </div>
      </div>
    </div>
  );
}
