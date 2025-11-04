import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { LayerNode } from '../../types';

interface BlockNodeProps extends NodeProps<LayerNode['data']> {}

const BlockNode = ({ data, selected }: BlockNodeProps) => {
  const expanded = data.params.expanded || false;
  const blockType = data.params.blockType || 'Custom';

  // Get block display name
  const getBlockDisplayName = (type: string) => {
    switch (type) {
      case 'SkipConnection':
        return 'Skip Connection';
      case 'Transformer':
        return 'Transformer';
      case 'Custom':
        return 'Custom Block';
      default:
        return type;
    }
  };

  const displayName = getBlockDisplayName(blockType);
  const borderClass = selected ? 'ring-2 ring-blue-400' : '';

  // Special styling for blocks - use cyan/teal gradient
  const blockColor = expanded
    ? 'bg-gradient-to-br from-cyan-600 to-teal-600 border-2 border-cyan-400'
    : 'bg-gradient-to-br from-cyan-700 to-teal-700';

  return (
    <div
      className={`${blockColor} ${borderClass} rounded-lg shadow-lg px-4 py-3 min-w-[180px] relative`}
      style={expanded ? { opacity: 0.9 } : {}}
    >
      {/* Input handle - always at top */}
      <Handle type="target" position={Position.Top} className="w-3 h-3" />

      <div className="text-white">
        {/* Header with expand/collapse button */}
        <div className="flex items-center justify-between mb-1">
          <div className="font-bold text-sm flex items-center gap-1">
            {/* Block icon */}
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <rect x="3" y="3" width="7" height="7" strokeWidth="2" rx="1"/>
              <rect x="14" y="3" width="7" height="7" strokeWidth="2" rx="1"/>
              <rect x="3" y="14" width="7" height="7" strokeWidth="2" rx="1"/>
              <rect x="14" y="14" width="7" height="7" strokeWidth="2" rx="1"/>
            </svg>
            Block
          </div>

          {/* Expand/collapse indicator */}
          <div className="text-xs opacity-75">
            {expanded ? '📂' : '📁'}
          </div>
        </div>

        {/* Block type */}
        <div className="text-xs mt-1 opacity-90 font-medium">
          {displayName}
        </div>

        {/* Block parameters (when collapsed) */}
        {!expanded && (
          <div className="text-xs mt-2 space-y-0.5 opacity-80">
            {data.params.filters && (
              <div>Filters: {data.params.filters}</div>
            )}
            {data.params.kernelSize && (
              <div>Kernel: {data.params.kernelSize}</div>
            )}
            {data.params.activation && (
              <div>Activation: {data.params.activation}</div>
            )}
          </div>
        )}

        {/* Expanded state indicator */}
        {expanded && (
          <div className="text-xs mt-2 opacity-75 italic">
            Click to collapse
          </div>
        )}
      </div>

      {/* Output handle - always at bottom */}
      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
};

export default memo(BlockNode);
