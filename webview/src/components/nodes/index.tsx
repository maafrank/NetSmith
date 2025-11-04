import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { LayerNode } from '../../types';

const BaseNode = ({ data, selected }: NodeProps<LayerNode['data']>) => {
  const getNodeColor = (layerType: string) => {
    switch (layerType) {
      case 'Input':
        return 'bg-green-600';
      case 'Output':
        return 'bg-red-600';
      case 'Dense':
        return 'bg-blue-600';
      case 'Conv2D':
      case 'Conv1D':
        return 'bg-purple-600';
      case 'MaxPool2D':
      case 'AvgPool2D':
        return 'bg-indigo-600';
      case 'Activation':
        return 'bg-yellow-600';
      case 'Dropout':
        return 'bg-orange-600';
      case 'BatchNorm':
        return 'bg-pink-600';
      case 'Flatten':
        return 'bg-teal-600';
      default:
        return 'bg-gray-600';
    }
  };

  const color = getNodeColor(data.layerType);
  const borderClass = selected ? 'ring-2 ring-blue-400' : '';

  return (
    <div className={`${color} ${borderClass} rounded-lg shadow-lg px-4 py-3 min-w-[150px]`}>
      {data.layerType !== 'Input' && (
        <Handle type="target" position={Position.Top} className="w-3 h-3" />
      )}

      <div className="text-white">
        <div className="font-bold text-sm">{data.layerType}</div>
        <div className="text-xs mt-1 opacity-90">
          {data.params.name || data.label}
        </div>

        {/* Layer-specific info */}
        {data.layerType === 'Dense' && data.params.units && (
          <div className="text-xs mt-1">Units: {data.params.units}</div>
        )}
        {data.layerType === 'Conv2D' && data.params.filters && (
          <div className="text-xs mt-1">
            Filters: {data.params.filters}
            {data.params.kernelSize && ` | K: ${data.params.kernelSize}`}
          </div>
        )}
        {data.layerType === 'Dropout' && data.params.rate && (
          <div className="text-xs mt-1">Rate: {data.params.rate}</div>
        )}
        {data.layerType === 'Activation' && data.params.activation && (
          <div className="text-xs mt-1">{data.params.activation}</div>
        )}
        {data.layerType === 'Output' && data.params.units && (
          <div className="text-xs mt-1">
            Classes: {data.params.units}
            {data.params.activation && ` | ${data.params.activation}`}
          </div>
        )}
      </div>

      {data.layerType !== 'Output' && (
        <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
      )}
    </div>
  );
};

export const nodeTypes = {
  default: memo(BaseNode),
  Input: memo(BaseNode),
  Dense: memo(BaseNode),
  Conv2D: memo(BaseNode),
  Conv1D: memo(BaseNode),
  MaxPool2D: memo(BaseNode),
  AvgPool2D: memo(BaseNode),
  Flatten: memo(BaseNode),
  Dropout: memo(BaseNode),
  BatchNorm: memo(BaseNode),
  Activation: memo(BaseNode),
  Output: memo(BaseNode),
  Block: memo(BaseNode),
};
