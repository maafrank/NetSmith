import { LayerNode, Edge, ActivationType } from './types';

/**
 * Block templates for creating reusable neural network patterns
 */

export interface BlockTemplate {
  internalNodes: LayerNode[];
  internalEdges: Edge[];
  defaultParams: Record<string, any>;
}

/**
 * Creates a skip connection (residual) block
 * Architecture: Conv2D → BatchNorm → ReLU → Add (with skip from input)
 */
export function createSkipConnectionBlock(blockId: string, params?: {
  filters?: number;
  kernelSize?: number | [number, number];
  activation?: ActivationType;
}): BlockTemplate {
  const {
    filters = 64,
    kernelSize = 3,
    activation = 'relu' as ActivationType
  } = params || {};

  // Internal node IDs (prefixed with block ID for uniqueness)
  const convId = `${blockId}_conv`;
  const bnId = `${blockId}_bn`;
  const actId = `${blockId}_act`;
  const addId = `${blockId}_add`;

  // Create internal nodes with relative positioning
  const internalNodes: LayerNode[] = [
    {
      id: convId,
      type: 'layer',
      position: { x: 0, y: 0 },
      data: {
        label: 'Conv2D',
        layerType: 'Conv2D',
        params: {
          filters,
          kernelSize,
          padding: 'same' as const,
          stride: 1
        }
      }
    },
    {
      id: bnId,
      type: 'layer',
      position: { x: 0, y: 100 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: {
          momentum: 0.99,
          epsilon: 0.001
        }
      }
    },
    {
      id: actId,
      type: 'layer',
      position: { x: 0, y: 200 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: {
          activation
        }
      }
    },
    {
      id: addId,
      type: 'layer',
      position: { x: 0, y: 300 },
      data: {
        label: 'Add',
        layerType: 'Add',
        params: {}
      }
    }
  ];

  // Create edges connecting the layers
  // Main path: Conv → BN → Act → Add
  // Skip connection: will be handled by the flattening logic (input → Add)
  const internalEdges: Edge[] = [
    {
      id: `${blockId}_edge_conv_bn`,
      source: convId,
      target: bnId
    },
    {
      id: `${blockId}_edge_bn_act`,
      source: bnId,
      target: actId
    },
    {
      id: `${blockId}_edge_act_add`,
      source: actId,
      target: addId
    }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: {
      filters,
      kernelSize,
      activation
    }
  };
}

/**
 * Registry of available block templates
 */
export const BLOCK_TEMPLATES: Record<string, (blockId: string, params?: any) => BlockTemplate> = {
  SkipConnection: createSkipConnectionBlock,
  // Future: Transformer, ResNetBlock, InceptionModule, etc.
};

/**
 * Get a block template by type
 */
export function getBlockTemplate(
  blockType: 'SkipConnection' | 'Transformer' | 'Custom',
  blockId: string,
  params?: Record<string, any>
): BlockTemplate | null {
  const templateFn = BLOCK_TEMPLATES[blockType as string];
  if (!templateFn) {
    return null;
  }
  return templateFn(blockId, params);
}
