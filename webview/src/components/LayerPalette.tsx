import { LayerType } from '../types';
import { useStore } from '../store';
import { getBlockTemplate } from '../blockTemplates';

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
    { type: 'GlobalAvgPool2D' as LayerType, icon: '🌐', description: 'Global average pooling' },
    { type: 'GlobalMaxPool2D' as LayerType, icon: '🔝', description: 'Global max pooling' },
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
    { type: 'Reshape' as LayerType, icon: '🔄', description: 'Reshape' },
    { type: 'UpSampling2D' as LayerType, icon: '⬆️', description: 'Upsample 2D' },
  ],
  'Merge Layers': [
    { type: 'Add' as LayerType, icon: '➕', description: 'Element-wise add' },
    { type: 'Concat' as LayerType, icon: '🔗', description: 'Concatenate' },
    { type: 'Multiply' as LayerType, icon: '✖️', description: 'Element-wise multiply' },
    { type: 'Subtract' as LayerType, icon: '➖', description: 'Element-wise subtract' },
    { type: 'Maximum' as LayerType, icon: '🔼', description: 'Element-wise maximum' },
    { type: 'Minimum' as LayerType, icon: '🔽', description: 'Element-wise minimum' },
  ],
  'Blocks - Basic': [
    { type: 'Block' as LayerType, icon: '🔷', description: 'Conv-BN-ReLU', blockType: 'ConvBNRelu' },
    { type: 'Block' as LayerType, icon: '🔄', description: 'Skip connection (ResNet)', blockType: 'SkipConnection' },
  ],
  'Blocks - Advanced': [
    { type: 'Block' as LayerType, icon: '🏺', description: 'Bottleneck (ResNet-50+)', blockType: 'Bottleneck' },
    { type: 'Block' as LayerType, icon: '📱', description: 'Depthwise separable', blockType: 'DepthwiseSeparable' },
    { type: 'Block' as LayerType, icon: '🔃', description: 'Inverted residual (MobileNetV2)', blockType: 'InvertedResidual' },
    { type: 'Block' as LayerType, icon: '🔥', description: 'Fire module (SqueezeNet)', blockType: 'Fire' },
    { type: 'Block' as LayerType, icon: '🎯', description: 'Squeeze-Excitation', blockType: 'SE' },
    { type: 'Block' as LayerType, icon: '🌲', description: 'Dense block (DenseNet)', blockType: 'Dense' },
    { type: 'Block' as LayerType, icon: '💎', description: 'Inception module', blockType: 'Inception' },
    { type: 'Block' as LayerType, icon: '🔬', description: 'SE-ResNet block', blockType: 'SEResNet' },
  ],
  'Blocks - U-Net': [
    { type: 'Block' as LayerType, icon: '⬇️', description: 'U-Net encoder', blockType: 'UNetEncoder' },
    { type: 'Block' as LayerType, icon: '⬆️', description: 'U-Net decoder', blockType: 'UNetDecoder' },
    { type: 'Block' as LayerType, icon: '🎯', description: 'Attention gate', blockType: 'AttentionGate' },
  ],
  'Blocks - Attention': [
    { type: 'Block' as LayerType, icon: '👁️', description: 'Self-attention', blockType: 'Attention' },
    { type: 'Block' as LayerType, icon: '🤖', description: 'Transformer (basic)', blockType: 'Transformer' },
    { type: 'Block' as LayerType, icon: '📥', description: 'Transformer encoder', blockType: 'TransformerEncoder' },
    { type: 'Block' as LayerType, icon: '📤', description: 'Transformer decoder', blockType: 'TransformerDecoder' },
  ],
};

export default function LayerPalette() {
  const { addNode, nodes } = useStore();

  const handleAddLayer = (layerType: LayerType, blockType?: string) => {
    const nodeId = `${layerType.toLowerCase()}_${Date.now()}`;
    const basePosition = {
      x: Math.random() * 500 + 100,
      y: Math.random() * 500 + 100,
    };

    // Handle blocks by generating all internal nodes directly
    if (layerType === 'Block' && blockType) {
      const template = getBlockTemplate(blockType as any, nodeId);
      if (template) {
        // Add all internal nodes with absolute positions
        const firstNodeId = template.internalNodes[0]?.id;
        const addNodeId = template.internalNodes.find(n => n.data.layerType === 'Add')?.id;

        template.internalNodes.forEach((internalNode) => {
          addNode({
            ...internalNode,
            position: {
              x: basePosition.x + internalNode.position.x,
              y: basePosition.y + internalNode.position.y,
            },
          });
        });

        // Add all internal edges AND skip connection
        const { setEdges } = useStore.getState();
        const currentEdges = useStore.getState().edges;
        const newEdges = [...currentEdges, ...template.internalEdges];

        // Add skip connection (from first node to Add node)
        if (firstNodeId && addNodeId) {
          newEdges.push({
            id: `${nodeId}_skip`,
            source: firstNodeId,
            target: addNodeId,
          });
        }

        setEdges(newEdges);
      }
      return; // Don't add a container node
    }

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
      case 'Output':
        defaultParams.units = 10;
        defaultParams.activation = 'softmax';
        break;
      case 'Concat':
        defaultParams.axis = -1;
        break;
      case 'Reshape':
        defaultParams.targetShape = [1, -1];
        break;
      case 'UpSampling2D':
        defaultParams.size = 2;
        defaultParams.interpolation = 'nearest';
        break;
      case 'Add':
      case 'Multiply':
      case 'Subtract':
      case 'Maximum':
      case 'Minimum':
      case 'GlobalAvgPool2D':
      case 'GlobalMaxPool2D':
        // No default params needed
        break;
    }

    addNode({
      id: nodeId,
      type: layerType,
      position: basePosition,
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
            {layers.map((layer: any) => (
              <button
                key={layer.type + (layer.blockType || '')}
                onClick={() => handleAddLayer(layer.type, layer.blockType)}
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
