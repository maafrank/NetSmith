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
 * Architecture: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → Add (with skip from input)
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
  const conv1Id = `${blockId}_conv1`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;
  const conv2Id = `${blockId}_conv2`;
  const bn2Id = `${blockId}_bn2`;
  const addId = `${blockId}_add`;
  const actFinalId = `${blockId}_act_final`;

  // Create internal nodes with relative positioning
  const internalNodes: LayerNode[] = [
    {
      id: conv1Id,
      type: 'Conv2D',
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
      id: bn1Id,
      type: 'BatchNorm',
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
      id: act1Id,
      type: 'Activation',
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
      id: conv2Id,
      type: 'Conv2D',
      position: { x: 0, y: 300 },
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
      id: bn2Id,
      type: 'BatchNorm',
      position: { x: 0, y: 400 },
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
      id: addId,
      type: 'Add',
      position: { x: 0, y: 500 },
      data: {
        label: 'Add',
        layerType: 'Add',
        params: {}
      }
    },
    {
      id: actFinalId,
      type: 'Activation',
      position: { x: 0, y: 600 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: {
          activation
        }
      }
    }
  ];

  // Create edges connecting the layers
  // Main path: Conv1 → BN1 → Act1 → Conv2 → BN2 → Add → ActFinal
  // Skip connection: will be handled by the flattening logic (input → Add)
  const internalEdges: Edge[] = [
    {
      id: `${blockId}_edge_conv1_bn1`,
      source: conv1Id,
      target: bn1Id
    },
    {
      id: `${blockId}_edge_bn1_act1`,
      source: bn1Id,
      target: act1Id
    },
    {
      id: `${blockId}_edge_act1_conv2`,
      source: act1Id,
      target: conv2Id
    },
    {
      id: `${blockId}_edge_conv2_bn2`,
      source: conv2Id,
      target: bn2Id
    },
    {
      id: `${blockId}_edge_bn2_add`,
      source: bn2Id,
      target: addId
    },
    {
      id: `${blockId}_edge_add_act_final`,
      source: addId,
      target: actFinalId
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
 * Creates a bottleneck block (ResNet-50+ style)
 * Architecture: 1x1 Conv (reduce) → 3x3 Conv → 1x1 Conv (expand) → Add (with skip)
 */
export function createBottleneckBlock(blockId: string, params?: {
  filters?: number;
  bottleneckRatio?: number;
  activation?: ActivationType;
}): BlockTemplate {
  const {
    filters = 64,
    bottleneckRatio = 4,
    activation = 'relu' as ActivationType
  } = params || {};

  const bottleneckFilters = Math.floor(filters / bottleneckRatio);

  const conv1Id = `${blockId}_conv1`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;
  const conv2Id = `${blockId}_conv2`;
  const bn2Id = `${blockId}_bn2`;
  const act2Id = `${blockId}_act2`;
  const conv3Id = `${blockId}_conv3`;
  const bn3Id = `${blockId}_bn3`;
  const addId = `${blockId}_add`;
  const actFinalId = `${blockId}_act_final`;

  const internalNodes: LayerNode[] = [
    {
      id: conv1Id,
      type: 'Conv2D',
      position: { x: 0, y: 0 },
      data: {
        label: 'Conv2D 1x1',
        layerType: 'Conv2D',
        params: { filters: bottleneckFilters, kernelSize: 1, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn1Id,
      type: 'BatchNorm',
      position: { x: 0, y: 100 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act1Id,
      type: 'Activation',
      position: { x: 0, y: 200 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: { activation }
      }
    },
    {
      id: conv2Id,
      type: 'Conv2D',
      position: { x: 0, y: 300 },
      data: {
        label: 'Conv2D 3x3',
        layerType: 'Conv2D',
        params: { filters: bottleneckFilters, kernelSize: 3, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn2Id,
      type: 'BatchNorm',
      position: { x: 0, y: 400 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act2Id,
      type: 'Activation',
      position: { x: 0, y: 500 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: { activation }
      }
    },
    {
      id: conv3Id,
      type: 'Conv2D',
      position: { x: 0, y: 600 },
      data: {
        label: 'Conv2D 1x1',
        layerType: 'Conv2D',
        params: { filters, kernelSize: 1, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn3Id,
      type: 'BatchNorm',
      position: { x: 0, y: 700 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: addId,
      type: 'Add',
      position: { x: 0, y: 800 },
      data: {
        label: 'Add',
        layerType: 'Add',
        params: {}
      }
    },
    {
      id: actFinalId,
      type: 'Activation',
      position: { x: 0, y: 900 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: { activation }
      }
    }
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_conv1_bn1`, source: conv1Id, target: bn1Id },
    { id: `${blockId}_edge_bn1_act1`, source: bn1Id, target: act1Id },
    { id: `${blockId}_edge_act1_conv2`, source: act1Id, target: conv2Id },
    { id: `${blockId}_edge_conv2_bn2`, source: conv2Id, target: bn2Id },
    { id: `${blockId}_edge_bn2_act2`, source: bn2Id, target: act2Id },
    { id: `${blockId}_edge_act2_conv3`, source: act2Id, target: conv3Id },
    { id: `${blockId}_edge_conv3_bn3`, source: conv3Id, target: bn3Id },
    { id: `${blockId}_edge_bn3_add`, source: bn3Id, target: addId },
    { id: `${blockId}_edge_add_act_final`, source: addId, target: actFinalId }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters, bottleneckRatio, activation }
  };
}

/**
 * Creates a depthwise separable convolution block
 * Architecture: Depthwise Conv → BatchNorm → ReLU → Pointwise Conv (1x1) → BatchNorm → ReLU
 * Note: Using regular Conv2D as placeholder since we don't have DepthwiseConv2D yet
 */
export function createDepthwiseSeparableBlock(blockId: string, params?: {
  filters?: number;
  kernelSize?: number;
  activation?: ActivationType;
}): BlockTemplate {
  const {
    filters = 64,
    kernelSize = 3,
    activation = 'relu' as ActivationType
  } = params || {};

  const dwConvId = `${blockId}_dw_conv`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;
  const pwConvId = `${blockId}_pw_conv`;
  const bn2Id = `${blockId}_bn2`;
  const act2Id = `${blockId}_act2`;

  const internalNodes: LayerNode[] = [
    {
      id: dwConvId,
      type: 'Conv2D',
      position: { x: 0, y: 0 },
      data: {
        label: `DW Conv ${kernelSize}x${kernelSize}`,
        layerType: 'Conv2D',
        params: { filters: 1, kernelSize, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn1Id,
      type: 'BatchNorm',
      position: { x: 0, y: 100 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act1Id,
      type: 'Activation',
      position: { x: 0, y: 200 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: { activation }
      }
    },
    {
      id: pwConvId,
      type: 'Conv2D',
      position: { x: 0, y: 300 },
      data: {
        label: 'PW Conv 1x1',
        layerType: 'Conv2D',
        params: { filters, kernelSize: 1, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn2Id,
      type: 'BatchNorm',
      position: { x: 0, y: 400 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act2Id,
      type: 'Activation',
      position: { x: 0, y: 500 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: { activation }
      }
    }
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_dw_bn1`, source: dwConvId, target: bn1Id },
    { id: `${blockId}_edge_bn1_act1`, source: bn1Id, target: act1Id },
    { id: `${blockId}_edge_act1_pw`, source: act1Id, target: pwConvId },
    { id: `${blockId}_edge_pw_bn2`, source: pwConvId, target: bn2Id },
    { id: `${blockId}_edge_bn2_act2`, source: bn2Id, target: act2Id }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters, kernelSize, activation }
  };
}

/**
 * Creates an inverted residual block (MobileNetV2)
 * Architecture: 1x1 Expand → DW Conv → 1x1 Project → Add (with skip)
 */
export function createInvertedResidualBlock(blockId: string, params?: {
  filters?: number;
  expandRatio?: number;
  activation?: ActivationType;
}): BlockTemplate {
  const {
    filters = 64,
    expandRatio = 6,
    activation = 'relu' as ActivationType
  } = params || {};

  const expandFilters = filters * expandRatio;

  const expandId = `${blockId}_expand`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;
  const dwId = `${blockId}_dw`;
  const bn2Id = `${blockId}_bn2`;
  const act2Id = `${blockId}_act2`;
  const projectId = `${blockId}_project`;
  const bn3Id = `${blockId}_bn3`;
  const addId = `${blockId}_add`;

  const internalNodes: LayerNode[] = [
    {
      id: expandId,
      type: 'Conv2D',
      position: { x: 0, y: 0 },
      data: {
        label: 'Expand 1x1',
        layerType: 'Conv2D',
        params: { filters: expandFilters, kernelSize: 1, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn1Id,
      type: 'BatchNorm',
      position: { x: 0, y: 100 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act1Id,
      type: 'Activation',
      position: { x: 0, y: 200 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: { activation }
      }
    },
    {
      id: dwId,
      type: 'Conv2D',
      position: { x: 0, y: 300 },
      data: {
        label: 'DW Conv 3x3',
        layerType: 'Conv2D',
        params: { filters: 1, kernelSize: 3, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn2Id,
      type: 'BatchNorm',
      position: { x: 0, y: 400 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act2Id,
      type: 'Activation',
      position: { x: 0, y: 500 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: { activation }
      }
    },
    {
      id: projectId,
      type: 'Conv2D',
      position: { x: 0, y: 600 },
      data: {
        label: 'Project 1x1',
        layerType: 'Conv2D',
        params: { filters, kernelSize: 1, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn3Id,
      type: 'BatchNorm',
      position: { x: 0, y: 700 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: addId,
      type: 'Add',
      position: { x: 0, y: 800 },
      data: {
        label: 'Add',
        layerType: 'Add',
        params: {}
      }
    }
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_expand_bn1`, source: expandId, target: bn1Id },
    { id: `${blockId}_edge_bn1_act1`, source: bn1Id, target: act1Id },
    { id: `${blockId}_edge_act1_dw`, source: act1Id, target: dwId },
    { id: `${blockId}_edge_dw_bn2`, source: dwId, target: bn2Id },
    { id: `${blockId}_edge_bn2_act2`, source: bn2Id, target: act2Id },
    { id: `${blockId}_edge_act2_project`, source: act2Id, target: projectId },
    { id: `${blockId}_edge_project_bn3`, source: projectId, target: bn3Id },
    { id: `${blockId}_edge_bn3_add`, source: bn3Id, target: addId }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters, expandRatio, activation }
  };
}

/**
 * Creates a basic Conv-BN-ReLU block
 * Architecture: Conv2D → BatchNorm → Activation
 */
export function createConvBNReluBlock(blockId: string, params?: {
  filters?: number;
  kernelSize?: number;
  activation?: ActivationType;
}): BlockTemplate {
  const {
    filters = 64,
    kernelSize = 3,
    activation = 'relu' as ActivationType
  } = params || {};

  const convId = `${blockId}_conv`;
  const bnId = `${blockId}_bn`;
  const actId = `${blockId}_act`;

  const internalNodes: LayerNode[] = [
    {
      id: convId,
      type: 'Conv2D',
      position: { x: 0, y: 0 },
      data: {
        label: 'Conv2D',
        layerType: 'Conv2D',
        params: { filters, kernelSize, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bnId,
      type: 'BatchNorm',
      position: { x: 0, y: 100 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: actId,
      type: 'Activation',
      position: { x: 0, y: 200 },
      data: {
        label: activation.toUpperCase(),
        layerType: 'Activation',
        params: { activation }
      }
    }
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_conv_bn`, source: convId, target: bnId },
    { id: `${blockId}_edge_bn_act`, source: bnId, target: actId }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters, kernelSize, activation }
  };
}

/**
 * Creates a Squeeze-and-Excitation (SE) block
 * Architecture: GlobalAvgPool → Dense (squeeze) → ReLU → Dense (excite) → Sigmoid
 * Note: Simplified version using Dense layers since we don't have GlobalAvgPool
 */
export function createSEBlock(blockId: string, params?: {
  filters?: number;
  reduction?: number;
}): BlockTemplate {
  const {
    filters = 64,
    reduction = 16
  } = params || {};

  const squeezeFilters = Math.max(1, Math.floor(filters / reduction));

  const flattenId = `${blockId}_flatten`;
  const squeezeId = `${blockId}_squeeze`;
  const act1Id = `${blockId}_act1`;
  const exciteId = `${blockId}_excite`;
  const sigmoidId = `${blockId}_sigmoid`;

  const internalNodes: LayerNode[] = [
    {
      id: flattenId,
      type: 'Flatten',
      position: { x: 0, y: 0 },
      data: {
        label: 'Flatten',
        layerType: 'Flatten',
        params: {}
      }
    },
    {
      id: squeezeId,
      type: 'Dense',
      position: { x: 0, y: 100 },
      data: {
        label: 'Squeeze',
        layerType: 'Dense',
        params: { units: squeezeFilters }
      }
    },
    {
      id: act1Id,
      type: 'Activation',
      position: { x: 0, y: 200 },
      data: {
        label: 'RELU',
        layerType: 'Activation',
        params: { activation: 'relu' as ActivationType }
      }
    },
    {
      id: exciteId,
      type: 'Dense',
      position: { x: 0, y: 300 },
      data: {
        label: 'Excite',
        layerType: 'Dense',
        params: { units: filters }
      }
    },
    {
      id: sigmoidId,
      type: 'Activation',
      position: { x: 0, y: 400 },
      data: {
        label: 'SIGMOID',
        layerType: 'Activation',
        params: { activation: 'sigmoid' as ActivationType }
      }
    }
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_flatten_squeeze`, source: flattenId, target: squeezeId },
    { id: `${blockId}_edge_squeeze_act`, source: squeezeId, target: act1Id },
    { id: `${blockId}_edge_act_excite`, source: act1Id, target: exciteId },
    { id: `${blockId}_edge_excite_sigmoid`, source: exciteId, target: sigmoidId }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters, reduction }
  };
}

/**
 * Creates a Fire Module (SqueezeNet)
 * Architecture: Squeeze (1x1) → Split → [Expand 1x1, Expand 3x3] → Concat
 * Note: Simplified sequential version since we need Concat layer
 */
export function createFireModule(blockId: string, params?: {
  squeezeFilters?: number;
  expandFilters?: number;
}): BlockTemplate {
  const {
    squeezeFilters = 16,
    expandFilters = 64
  } = params || {};

  const squeezeId = `${blockId}_squeeze`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;
  const expand1x1Id = `${blockId}_expand1x1`;
  const bn2Id = `${blockId}_bn2`;
  const act2Id = `${blockId}_act2`;
  const expand3x3Id = `${blockId}_expand3x3`;
  const bn3Id = `${blockId}_bn3`;
  const act3Id = `${blockId}_act3`;

  const internalNodes: LayerNode[] = [
    {
      id: squeezeId,
      type: 'Conv2D',
      position: { x: 0, y: 0 },
      data: {
        label: 'Squeeze 1x1',
        layerType: 'Conv2D',
        params: { filters: squeezeFilters, kernelSize: 1, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn1Id,
      type: 'BatchNorm',
      position: { x: 0, y: 100 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act1Id,
      type: 'Activation',
      position: { x: 0, y: 200 },
      data: {
        label: 'RELU',
        layerType: 'Activation',
        params: { activation: 'relu' as ActivationType }
      }
    },
    {
      id: expand1x1Id,
      type: 'Conv2D',
      position: { x: 0, y: 300 },
      data: {
        label: 'Expand 1x1',
        layerType: 'Conv2D',
        params: { filters: expandFilters / 2, kernelSize: 1, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn2Id,
      type: 'BatchNorm',
      position: { x: 0, y: 400 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act2Id,
      type: 'Activation',
      position: { x: 0, y: 500 },
      data: {
        label: 'RELU',
        layerType: 'Activation',
        params: { activation: 'relu' as ActivationType }
      }
    },
    {
      id: expand3x3Id,
      type: 'Conv2D',
      position: { x: 0, y: 600 },
      data: {
        label: 'Expand 3x3',
        layerType: 'Conv2D',
        params: { filters: expandFilters / 2, kernelSize: 3, padding: 'same' as const, stride: 1 }
      }
    },
    {
      id: bn3Id,
      type: 'BatchNorm',
      position: { x: 0, y: 700 },
      data: {
        label: 'BatchNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: act3Id,
      type: 'Activation',
      position: { x: 0, y: 800 },
      data: {
        label: 'RELU',
        layerType: 'Activation',
        params: { activation: 'relu' as ActivationType }
      }
    }
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_squeeze_bn1`, source: squeezeId, target: bn1Id },
    { id: `${blockId}_edge_bn1_act1`, source: bn1Id, target: act1Id },
    { id: `${blockId}_edge_act1_expand1`, source: act1Id, target: expand1x1Id },
    { id: `${blockId}_edge_expand1_bn2`, source: expand1x1Id, target: bn2Id },
    { id: `${blockId}_edge_bn2_act2`, source: bn2Id, target: act2Id },
    { id: `${blockId}_edge_act2_expand3`, source: act2Id, target: expand3x3Id },
    { id: `${blockId}_edge_expand3_bn3`, source: expand3x3Id, target: bn3Id },
    { id: `${blockId}_edge_bn3_act3`, source: bn3Id, target: act3Id }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { squeezeFilters, expandFilters }
  };
}

/**
 * Creates a Dense Block (DenseNet)
 * Architecture: Multiple Conv-BN-ReLU layers where each connects to all subsequent layers
 * Simplified to 3 layers for clarity
 */
export function createDenseBlock(blockId: string, params?: {
  growthRate?: number;
  numLayers?: number;
}): BlockTemplate {
  const {
    growthRate = 32,
    numLayers = 3
  } = params || {};

  const layers: LayerNode[] = [];
  const edges: Edge[] = [];

  // Create 3 dense layers
  for (let i = 0; i < 3; i++) {
    const convId = `${blockId}_conv${i}`;
    const bnId = `${blockId}_bn${i}`;
    const actId = `${blockId}_act${i}`;

    layers.push(
      {
        id: bnId,
        type: 'BatchNorm',
        position: { x: 0, y: i * 300 },
        data: {
          label: 'BatchNorm',
          layerType: 'BatchNorm',
          params: { momentum: 0.99, epsilon: 0.001 }
        }
      },
      {
        id: actId,
        type: 'Activation',
        position: { x: 0, y: i * 300 + 100 },
        data: {
          label: 'RELU',
          layerType: 'Activation',
          params: { activation: 'relu' as ActivationType }
        }
      },
      {
        id: convId,
        type: 'Conv2D',
        position: { x: 0, y: i * 300 + 200 },
        data: {
          label: `Conv ${i + 1}`,
          layerType: 'Conv2D',
          params: { filters: growthRate, kernelSize: 3, padding: 'same' as const, stride: 1 }
        }
      }
    );

    edges.push(
      { id: `${blockId}_edge_bn${i}_act${i}`, source: bnId, target: actId },
      { id: `${blockId}_edge_act${i}_conv${i}`, source: actId, target: convId }
    );

    // Connect to next layer if not last
    if (i < 2) {
      edges.push({
        id: `${blockId}_edge_conv${i}_bn${i + 1}`,
        source: convId,
        target: `${blockId}_bn${i + 1}`
      });
    }
  }

  return {
    internalNodes: layers,
    internalEdges: edges,
    defaultParams: { growthRate, numLayers }
  };
}

/**
 * Creates a simple Self-Attention block
 * Architecture: Dense (query) + Dense (key) + Dense (value) → Dense (output)
 * Simplified version without actual attention mechanism
 */
export function createAttentionBlock(blockId: string, params?: {
  units?: number;
  heads?: number;
}): BlockTemplate {
  const {
    units = 64,
    heads = 4
  } = params || {};

  const queryId = `${blockId}_query`;
  const keyId = `${blockId}_key`;
  const valueId = `${blockId}_value`;
  const outputId = `${blockId}_output`;
  const addId = `${blockId}_add`;

  const internalNodes: LayerNode[] = [
    {
      id: queryId,
      type: 'Dense',
      position: { x: 0, y: 0 },
      data: {
        label: 'Query',
        layerType: 'Dense',
        params: { units }
      }
    },
    {
      id: keyId,
      type: 'Dense',
      position: { x: 0, y: 100 },
      data: {
        label: 'Key',
        layerType: 'Dense',
        params: { units }
      }
    },
    {
      id: valueId,
      type: 'Dense',
      position: { x: 0, y: 200 },
      data: {
        label: 'Value',
        layerType: 'Dense',
        params: { units }
      }
    },
    {
      id: outputId,
      type: 'Dense',
      position: { x: 0, y: 300 },
      data: {
        label: 'Output',
        layerType: 'Dense',
        params: { units }
      }
    },
    {
      id: addId,
      type: 'Add',
      position: { x: 0, y: 400 },
      data: {
        label: 'Add (Residual)',
        layerType: 'Add',
        params: {}
      }
    }
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_q_k`, source: queryId, target: keyId },
    { id: `${blockId}_edge_k_v`, source: keyId, target: valueId },
    { id: `${blockId}_edge_v_out`, source: valueId, target: outputId },
    { id: `${blockId}_edge_out_add`, source: outputId, target: addId }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { units, heads }
  };
}

/**
 * Creates a Transformer Block
 * Architecture: Attention → Add → LayerNorm → FFN → Add → LayerNorm
 * Simplified version
 */
export function createTransformerBlock(blockId: string, params?: {
  units?: number;
  ffnUnits?: number;
}): BlockTemplate {
  const {
    units = 128,
    ffnUnits = 512
  } = params || {};

  const attnId = `${blockId}_attn`;
  const add1Id = `${blockId}_add1`;
  const norm1Id = `${blockId}_norm1`;
  const ffn1Id = `${blockId}_ffn1`;
  const actId = `${blockId}_act`;
  const ffn2Id = `${blockId}_ffn2`;
  const add2Id = `${blockId}_add2`;
  const norm2Id = `${blockId}_norm2`;

  const internalNodes: LayerNode[] = [
    {
      id: attnId,
      type: 'Dense',
      position: { x: 0, y: 0 },
      data: {
        label: 'Attention',
        layerType: 'Dense',
        params: { units }
      }
    },
    {
      id: add1Id,
      type: 'Add',
      position: { x: 0, y: 100 },
      data: {
        label: 'Add',
        layerType: 'Add',
        params: {}
      }
    },
    {
      id: norm1Id,
      type: 'BatchNorm',
      position: { x: 0, y: 200 },
      data: {
        label: 'LayerNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    },
    {
      id: ffn1Id,
      type: 'Dense',
      position: { x: 0, y: 300 },
      data: {
        label: 'FFN',
        layerType: 'Dense',
        params: { units: ffnUnits }
      }
    },
    {
      id: actId,
      type: 'Activation',
      position: { x: 0, y: 400 },
      data: {
        label: 'RELU',
        layerType: 'Activation',
        params: { activation: 'relu' as ActivationType }
      }
    },
    {
      id: ffn2Id,
      type: 'Dense',
      position: { x: 0, y: 500 },
      data: {
        label: 'FFN Out',
        layerType: 'Dense',
        params: { units }
      }
    },
    {
      id: add2Id,
      type: 'Add',
      position: { x: 0, y: 600 },
      data: {
        label: 'Add',
        layerType: 'Add',
        params: {}
      }
    },
    {
      id: norm2Id,
      type: 'BatchNorm',
      position: { x: 0, y: 700 },
      data: {
        label: 'LayerNorm',
        layerType: 'BatchNorm',
        params: { momentum: 0.99, epsilon: 0.001 }
      }
    }
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_attn_add1`, source: attnId, target: add1Id },
    { id: `${blockId}_edge_add1_norm1`, source: add1Id, target: norm1Id },
    { id: `${blockId}_edge_norm1_ffn1`, source: norm1Id, target: ffn1Id },
    { id: `${blockId}_edge_ffn1_act`, source: ffn1Id, target: actId },
    { id: `${blockId}_edge_act_ffn2`, source: actId, target: ffn2Id },
    { id: `${blockId}_edge_ffn2_add2`, source: ffn2Id, target: add2Id },
    { id: `${blockId}_edge_add2_norm2`, source: add2Id, target: norm2Id }
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { units, ffnUnits }
  };
}

/**
 * Creates an Inception module with parallel paths
 * Architecture: 4 parallel paths (1x1, 3x3, 5x5, maxpool) → Concat
 */
export function createInceptionBlock(blockId: string, params?: {
  filters1x1?: number;
  filters3x3?: number;
  filters5x5?: number;
}): BlockTemplate {
  const {
    filters1x1 = 64,
    filters3x3 = 96,
    filters5x5 = 48
  } = params || {};

  // Path 1: 1x1 conv
  const conv1x1Id = `${blockId}_conv1x1`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;

  // Path 2: 1x1 → 3x3 conv
  const conv1x1_3Id = `${blockId}_conv1x1_3`;
  const bn2Id = `${blockId}_bn2`;
  const act2Id = `${blockId}_act2`;
  const conv3x3Id = `${blockId}_conv3x3`;
  const bn3Id = `${blockId}_bn3`;
  const act3Id = `${blockId}_act3`;

  // Path 3: 1x1 → 5x5 conv
  const conv1x1_5Id = `${blockId}_conv1x1_5`;
  const bn4Id = `${blockId}_bn4`;
  const act4Id = `${blockId}_act4`;
  const conv5x5Id = `${blockId}_conv5x5`;
  const bn5Id = `${blockId}_bn5`;
  const act5Id = `${blockId}_act5`;

  // Path 4: maxpool → 1x1
  const poolId = `${blockId}_pool`;
  const convPoolId = `${blockId}_conv_pool`;
  const bn6Id = `${blockId}_bn6`;
  const act6Id = `${blockId}_act6`;

  // Concat all paths
  const concat1Id = `${blockId}_concat1`;
  const concat2Id = `${blockId}_concat2`;
  const concat3Id = `${blockId}_concat3`;

  const internalNodes: LayerNode[] = [
    // Path 1: 1x1
    { id: conv1x1Id, type: 'Conv2D', position: { x: -300, y: 0 }, data: { label: 'Conv 1x1', layerType: 'Conv2D', params: { filters: filters1x1, kernelSize: 1, padding: 'same' as const } } },
    { id: bn1Id, type: 'BatchNorm', position: { x: -300, y: 100 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act1Id, type: 'Activation', position: { x: -300, y: 200 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },

    // Path 2: 1x1 → 3x3
    { id: conv1x1_3Id, type: 'Conv2D', position: { x: -100, y: 0 }, data: { label: 'Conv 1x1', layerType: 'Conv2D', params: { filters: filters3x3 / 2, kernelSize: 1, padding: 'same' as const } } },
    { id: bn2Id, type: 'BatchNorm', position: { x: -100, y: 100 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act2Id, type: 'Activation', position: { x: -100, y: 200 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
    { id: conv3x3Id, type: 'Conv2D', position: { x: -100, y: 300 }, data: { label: 'Conv 3x3', layerType: 'Conv2D', params: { filters: filters3x3, kernelSize: 3, padding: 'same' as const } } },
    { id: bn3Id, type: 'BatchNorm', position: { x: -100, y: 400 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act3Id, type: 'Activation', position: { x: -100, y: 500 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },

    // Path 3: 1x1 → 5x5
    { id: conv1x1_5Id, type: 'Conv2D', position: { x: 100, y: 0 }, data: { label: 'Conv 1x1', layerType: 'Conv2D', params: { filters: filters5x5 / 2, kernelSize: 1, padding: 'same' as const } } },
    { id: bn4Id, type: 'BatchNorm', position: { x: 100, y: 100 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act4Id, type: 'Activation', position: { x: 100, y: 200 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
    { id: conv5x5Id, type: 'Conv2D', position: { x: 100, y: 300 }, data: { label: 'Conv 5x5', layerType: 'Conv2D', params: { filters: filters5x5, kernelSize: 5, padding: 'same' as const } } },
    { id: bn5Id, type: 'BatchNorm', position: { x: 100, y: 400 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act5Id, type: 'Activation', position: { x: 100, y: 500 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },

    // Path 4: pool → 1x1
    { id: poolId, type: 'MaxPool2D', position: { x: 300, y: 0 }, data: { label: 'MaxPool', layerType: 'MaxPool2D', params: { poolSize: 3 } } },
    { id: convPoolId, type: 'Conv2D', position: { x: 300, y: 100 }, data: { label: 'Conv 1x1', layerType: 'Conv2D', params: { filters: 32, kernelSize: 1, padding: 'same' as const } } },
    { id: bn6Id, type: 'BatchNorm', position: { x: 300, y: 200 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act6Id, type: 'Activation', position: { x: 300, y: 300 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },

    // Concatenate all paths
    { id: concat1Id, type: 'Concat', position: { x: 0, y: 600 }, data: { label: 'Concat', layerType: 'Concat', params: { axis: 1 } } },
    { id: concat2Id, type: 'Concat', position: { x: 0, y: 700 }, data: { label: 'Concat', layerType: 'Concat', params: { axis: 1 } } },
    { id: concat3Id, type: 'Concat', position: { x: 0, y: 800 }, data: { label: 'Concat', layerType: 'Concat', params: { axis: 1 } } },
  ];

  const internalEdges: Edge[] = [
    // Path 1
    { id: `${blockId}_edge_conv1_bn1`, source: conv1x1Id, target: bn1Id },
    { id: `${blockId}_edge_bn1_act1`, source: bn1Id, target: act1Id },

    // Path 2
    { id: `${blockId}_edge_conv1_3_bn2`, source: conv1x1_3Id, target: bn2Id },
    { id: `${blockId}_edge_bn2_act2`, source: bn2Id, target: act2Id },
    { id: `${blockId}_edge_act2_conv3`, source: act2Id, target: conv3x3Id },
    { id: `${blockId}_edge_conv3_bn3`, source: conv3x3Id, target: bn3Id },
    { id: `${blockId}_edge_bn3_act3`, source: bn3Id, target: act3Id },

    // Path 3
    { id: `${blockId}_edge_conv1_5_bn4`, source: conv1x1_5Id, target: bn4Id },
    { id: `${blockId}_edge_bn4_act4`, source: bn4Id, target: act4Id },
    { id: `${blockId}_edge_act4_conv5`, source: act4Id, target: conv5x5Id },
    { id: `${blockId}_edge_conv5_bn5`, source: conv5x5Id, target: bn5Id },
    { id: `${blockId}_edge_bn5_act5`, source: bn5Id, target: act5Id },

    // Path 4
    { id: `${blockId}_edge_pool_conv`, source: poolId, target: convPoolId },
    { id: `${blockId}_edge_conv_bn6`, source: convPoolId, target: bn6Id },
    { id: `${blockId}_edge_bn6_act6`, source: bn6Id, target: act6Id },

    // Concatenation (chain the concats)
    { id: `${blockId}_edge_act1_concat1`, source: act1Id, target: concat1Id },
    { id: `${blockId}_edge_act3_concat1`, source: act3Id, target: concat1Id },
    { id: `${blockId}_edge_concat1_concat2`, source: concat1Id, target: concat2Id },
    { id: `${blockId}_edge_act5_concat2`, source: act5Id, target: concat2Id },
    { id: `${blockId}_edge_concat2_concat3`, source: concat2Id, target: concat3Id },
    { id: `${blockId}_edge_act6_concat3`, source: act6Id, target: concat3Id },
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters1x1, filters3x3, filters5x5 }
  };
}

/**
 * Creates a U-Net encoder block
 * Architecture: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool (with skip output before pool)
 */
export function createUNetEncoderBlock(blockId: string, params?: {
  filters?: number;
}): BlockTemplate {
  const { filters = 64 } = params || {};

  const conv1Id = `${blockId}_conv1`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;
  const conv2Id = `${blockId}_conv2`;
  const bn2Id = `${blockId}_bn2`;
  const act2Id = `${blockId}_act2`;
  const poolId = `${blockId}_pool`;

  const internalNodes: LayerNode[] = [
    { id: conv1Id, type: 'Conv2D', position: { x: 0, y: 0 }, data: { label: 'Conv2D', layerType: 'Conv2D', params: { filters, kernelSize: 3, padding: 'same' as const } } },
    { id: bn1Id, type: 'BatchNorm', position: { x: 0, y: 100 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act1Id, type: 'Activation', position: { x: 0, y: 200 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
    { id: conv2Id, type: 'Conv2D', position: { x: 0, y: 300 }, data: { label: 'Conv2D', layerType: 'Conv2D', params: { filters, kernelSize: 3, padding: 'same' as const } } },
    { id: bn2Id, type: 'BatchNorm', position: { x: 0, y: 400 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act2Id, type: 'Activation', position: { x: 0, y: 500 }, data: { label: 'RELU (skip)', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
    { id: poolId, type: 'MaxPool2D', position: { x: 0, y: 600 }, data: { label: 'MaxPool', layerType: 'MaxPool2D', params: { poolSize: 2 } } },
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_conv1_bn1`, source: conv1Id, target: bn1Id },
    { id: `${blockId}_edge_bn1_act1`, source: bn1Id, target: act1Id },
    { id: `${blockId}_edge_act1_conv2`, source: act1Id, target: conv2Id },
    { id: `${blockId}_edge_conv2_bn2`, source: conv2Id, target: bn2Id },
    { id: `${blockId}_edge_bn2_act2`, source: bn2Id, target: act2Id },
    { id: `${blockId}_edge_act2_pool`, source: act2Id, target: poolId },
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters }
  };
}

/**
 * Creates a U-Net decoder block
 * Architecture: UpSample → Conv → BN → ReLU → Concat (with skip) → Conv → BN → ReLU
 */
export function createUNetDecoderBlock(blockId: string, params?: {
  filters?: number;
}): BlockTemplate {
  const { filters = 64 } = params || {};

  const upsampleId = `${blockId}_upsample`;
  const conv1Id = `${blockId}_conv1`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;
  const concatId = `${blockId}_concat`;
  const conv2Id = `${blockId}_conv2`;
  const bn2Id = `${blockId}_bn2`;
  const act2Id = `${blockId}_act2`;

  const internalNodes: LayerNode[] = [
    { id: upsampleId, type: 'UpSampling2D', position: { x: 0, y: 0 }, data: { label: 'UpSample', layerType: 'UpSampling2D', params: { size: 2, interpolation: 'nearest' as const } } },
    { id: conv1Id, type: 'Conv2D', position: { x: 0, y: 100 }, data: { label: 'Conv2D', layerType: 'Conv2D', params: { filters, kernelSize: 3, padding: 'same' as const } } },
    { id: bn1Id, type: 'BatchNorm', position: { x: 0, y: 200 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act1Id, type: 'Activation', position: { x: 0, y: 300 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
    { id: concatId, type: 'Concat', position: { x: 0, y: 400 }, data: { label: 'Concat (skip)', layerType: 'Concat', params: { axis: 1 } } },
    { id: conv2Id, type: 'Conv2D', position: { x: 0, y: 500 }, data: { label: 'Conv2D', layerType: 'Conv2D', params: { filters, kernelSize: 3, padding: 'same' as const } } },
    { id: bn2Id, type: 'BatchNorm', position: { x: 0, y: 600 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act2Id, type: 'Activation', position: { x: 0, y: 700 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_up_conv1`, source: upsampleId, target: conv1Id },
    { id: `${blockId}_edge_conv1_bn1`, source: conv1Id, target: bn1Id },
    { id: `${blockId}_edge_bn1_act1`, source: bn1Id, target: act1Id },
    { id: `${blockId}_edge_act1_concat`, source: act1Id, target: concatId },
    { id: `${blockId}_edge_concat_conv2`, source: concatId, target: conv2Id },
    { id: `${blockId}_edge_conv2_bn2`, source: conv2Id, target: bn2Id },
    { id: `${blockId}_edge_bn2_act2`, source: bn2Id, target: act2Id },
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters }
  };
}

/**
 * Creates a Squeeze-Excitation ResNet block
 * Architecture: Residual path + SE path with GlobalAvgPool → Dense → Dense → Multiply
 */
export function createSEResNetBlock(blockId: string, params?: {
  filters?: number;
  reduction?: number;
}): BlockTemplate {
  const { filters = 64, reduction = 16 } = params || {};

  const conv1Id = `${blockId}_conv1`;
  const bn1Id = `${blockId}_bn1`;
  const act1Id = `${blockId}_act1`;
  const conv2Id = `${blockId}_conv2`;
  const bn2Id = `${blockId}_bn2`;

  // SE path
  const gapId = `${blockId}_gap`;
  const flattenId = `${blockId}_flatten`;
  const fc1Id = `${blockId}_fc1`;
  const actSeId = `${blockId}_act_se`;
  const fc2Id = `${blockId}_fc2`;
  const sigmoidId = `${blockId}_sigmoid`;

  const multiplyId = `${blockId}_multiply`;
  const addId = `${blockId}_add`;
  const actFinalId = `${blockId}_act_final`;

  const internalNodes: LayerNode[] = [
    // Main path
    { id: conv1Id, type: 'Conv2D', position: { x: -100, y: 0 }, data: { label: 'Conv2D', layerType: 'Conv2D', params: { filters, kernelSize: 3, padding: 'same' as const } } },
    { id: bn1Id, type: 'BatchNorm', position: { x: -100, y: 100 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },
    { id: act1Id, type: 'Activation', position: { x: -100, y: 200 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
    { id: conv2Id, type: 'Conv2D', position: { x: -100, y: 300 }, data: { label: 'Conv2D', layerType: 'Conv2D', params: { filters, kernelSize: 3, padding: 'same' as const } } },
    { id: bn2Id, type: 'BatchNorm', position: { x: -100, y: 400 }, data: { label: 'BN', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 0.001 } } },

    // SE path
    { id: gapId, type: 'GlobalAvgPool2D', position: { x: 100, y: 400 }, data: { label: 'GAP', layerType: 'GlobalAvgPool2D', params: {} } },
    { id: flattenId, type: 'Flatten', position: { x: 100, y: 500 }, data: { label: 'Flatten', layerType: 'Flatten', params: {} } },
    { id: fc1Id, type: 'Dense', position: { x: 100, y: 600 }, data: { label: 'FC', layerType: 'Dense', params: { units: Math.max(1, Math.floor(filters / reduction)) } } },
    { id: actSeId, type: 'Activation', position: { x: 100, y: 700 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
    { id: fc2Id, type: 'Dense', position: { x: 100, y: 800 }, data: { label: 'FC', layerType: 'Dense', params: { units: filters } } },
    { id: sigmoidId, type: 'Activation', position: { x: 100, y: 900 }, data: { label: 'SIGMOID', layerType: 'Activation', params: { activation: 'sigmoid' as ActivationType } } },

    // Merge
    { id: multiplyId, type: 'Multiply', position: { x: 0, y: 1000 }, data: { label: 'Multiply', layerType: 'Multiply', params: {} } },
    { id: addId, type: 'Add', position: { x: 0, y: 1100 }, data: { label: 'Add', layerType: 'Add', params: {} } },
    { id: actFinalId, type: 'Activation', position: { x: 0, y: 1200 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
  ];

  const internalEdges: Edge[] = [
    // Main path
    { id: `${blockId}_edge_conv1_bn1`, source: conv1Id, target: bn1Id },
    { id: `${blockId}_edge_bn1_act1`, source: bn1Id, target: act1Id },
    { id: `${blockId}_edge_act1_conv2`, source: act1Id, target: conv2Id },
    { id: `${blockId}_edge_conv2_bn2`, source: conv2Id, target: bn2Id },

    // SE path
    { id: `${blockId}_edge_bn2_gap`, source: bn2Id, target: gapId },
    { id: `${blockId}_edge_gap_flatten`, source: gapId, target: flattenId },
    { id: `${blockId}_edge_flatten_fc1`, source: flattenId, target: fc1Id },
    { id: `${blockId}_edge_fc1_act`, source: fc1Id, target: actSeId },
    { id: `${blockId}_edge_act_fc2`, source: actSeId, target: fc2Id },
    { id: `${blockId}_edge_fc2_sigmoid`, source: fc2Id, target: sigmoidId },

    // Apply SE and residual
    { id: `${blockId}_edge_bn2_multiply`, source: bn2Id, target: multiplyId },
    { id: `${blockId}_edge_sigmoid_multiply`, source: sigmoidId, target: multiplyId },
    { id: `${blockId}_edge_multiply_add`, source: multiplyId, target: addId },
    { id: `${blockId}_edge_add_final`, source: addId, target: actFinalId },
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters, reduction }
  };
}

/**
 * Creates an Attention Gate block (for U-Net)
 * Architecture: Two inputs (skip + gating) → process → Multiply with skip
 */
export function createAttentionGateBlock(blockId: string, params?: {
  filters?: number;
}): BlockTemplate {
  const { filters = 64 } = params || {};

  const convSkipId = `${blockId}_conv_skip`;
  const convGateId = `${blockId}_conv_gate`;
  const addId = `${blockId}_add`;
  const actId = `${blockId}_act`;
  const convAttId = `${blockId}_conv_att`;
  const sigmoidId = `${blockId}_sigmoid`;
  const multiplyId = `${blockId}_multiply`;

  const internalNodes: LayerNode[] = [
    { id: convSkipId, type: 'Conv2D', position: { x: -100, y: 0 }, data: { label: 'Conv Skip', layerType: 'Conv2D', params: { filters, kernelSize: 1, padding: 'same' as const } } },
    { id: convGateId, type: 'Conv2D', position: { x: 100, y: 0 }, data: { label: 'Conv Gate', layerType: 'Conv2D', params: { filters, kernelSize: 1, padding: 'same' as const } } },
    { id: addId, type: 'Add', position: { x: 0, y: 100 }, data: { label: 'Add', layerType: 'Add', params: {} } },
    { id: actId, type: 'Activation', position: { x: 0, y: 200 }, data: { label: 'RELU', layerType: 'Activation', params: { activation: 'relu' as ActivationType } } },
    { id: convAttId, type: 'Conv2D', position: { x: 0, y: 300 }, data: { label: 'Conv Att', layerType: 'Conv2D', params: { filters: 1, kernelSize: 1, padding: 'same' as const } } },
    { id: sigmoidId, type: 'Activation', position: { x: 0, y: 400 }, data: { label: 'SIGMOID', layerType: 'Activation', params: { activation: 'sigmoid' as ActivationType } } },
    { id: multiplyId, type: 'Multiply', position: { x: 0, y: 500 }, data: { label: 'Multiply', layerType: 'Multiply', params: {} } },
  ];

  const internalEdges: Edge[] = [
    { id: `${blockId}_edge_skip_add`, source: convSkipId, target: addId },
    { id: `${blockId}_edge_gate_add`, source: convGateId, target: addId },
    { id: `${blockId}_edge_add_act`, source: addId, target: actId },
    { id: `${blockId}_edge_act_conv`, source: actId, target: convAttId },
    { id: `${blockId}_edge_conv_sigmoid`, source: convAttId, target: sigmoidId },
    { id: `${blockId}_edge_sigmoid_multiply`, source: sigmoidId, target: multiplyId },
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { filters }
  };
}

/**
 * Creates a Transformer Encoder block
 * Architecture: Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
 */
export function createTransformerEncoderBlock(blockId: string, params?: {
  units?: number;
  ffnUnits?: number;
  heads?: number;
}): BlockTemplate {
  const {
    units = 128,
    ffnUnits = 512,
    heads = 8
  } = params || {};

  // Multi-head attention (simplified as single Dense layers)
  const qId = `${blockId}_query`;
  const kId = `${blockId}_key`;
  const vId = `${blockId}_value`;
  const attnOutId = `${blockId}_attn_out`;

  // Add & Norm 1
  const add1Id = `${blockId}_add1`;
  const norm1Id = `${blockId}_norm1`;

  // FFN
  const ffn1Id = `${blockId}_ffn1`;
  const actId = `${blockId}_act`;
  const dropoutId = `${blockId}_dropout`;
  const ffn2Id = `${blockId}_ffn2`;

  // Add & Norm 2
  const add2Id = `${blockId}_add2`;
  const norm2Id = `${blockId}_norm2`;

  const internalNodes: LayerNode[] = [
    // Multi-head self-attention
    { id: qId, type: 'Dense', position: { x: -100, y: 0 }, data: { label: 'Query', layerType: 'Dense', params: { units } } },
    { id: kId, type: 'Dense', position: { x: 0, y: 0 }, data: { label: 'Key', layerType: 'Dense', params: { units } } },
    { id: vId, type: 'Dense', position: { x: 100, y: 0 }, data: { label: 'Value', layerType: 'Dense', params: { units } } },
    { id: attnOutId, type: 'Dense', position: { x: 0, y: 100 }, data: { label: 'Attn Output', layerType: 'Dense', params: { units } } },

    // Add & Norm 1 (residual connection)
    { id: add1Id, type: 'Add', position: { x: 0, y: 200 }, data: { label: 'Add (residual)', layerType: 'Add', params: {} } },
    { id: norm1Id, type: 'BatchNorm', position: { x: 0, y: 300 }, data: { label: 'LayerNorm', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 1e-6 } } },

    // Feed-forward network
    { id: ffn1Id, type: 'Dense', position: { x: 0, y: 400 }, data: { label: 'FFN', layerType: 'Dense', params: { units: ffnUnits } } },
    { id: actId, type: 'Activation', position: { x: 0, y: 500 }, data: { label: 'GELU', layerType: 'Activation', params: { activation: 'gelu' as ActivationType } } },
    { id: dropoutId, type: 'Dropout', position: { x: 0, y: 600 }, data: { label: 'Dropout', layerType: 'Dropout', params: { rate: 0.1 } } },
    { id: ffn2Id, type: 'Dense', position: { x: 0, y: 700 }, data: { label: 'FFN Out', layerType: 'Dense', params: { units } } },

    // Add & Norm 2 (residual connection)
    { id: add2Id, type: 'Add', position: { x: 0, y: 800 }, data: { label: 'Add (residual)', layerType: 'Add', params: {} } },
    { id: norm2Id, type: 'BatchNorm', position: { x: 0, y: 900 }, data: { label: 'LayerNorm', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 1e-6 } } },
  ];

  const internalEdges: Edge[] = [
    // Attention path (simplified - in reality Q, K, V interact via attention mechanism)
    { id: `${blockId}_edge_q_k`, source: qId, target: kId },
    { id: `${blockId}_edge_k_v`, source: kId, target: vId },
    { id: `${blockId}_edge_v_attn`, source: vId, target: attnOutId },

    // First residual connection
    { id: `${blockId}_edge_attn_add1`, source: attnOutId, target: add1Id },
    { id: `${blockId}_edge_add1_norm1`, source: add1Id, target: norm1Id },

    // FFN path
    { id: `${blockId}_edge_norm1_ffn1`, source: norm1Id, target: ffn1Id },
    { id: `${blockId}_edge_ffn1_act`, source: ffn1Id, target: actId },
    { id: `${blockId}_edge_act_dropout`, source: actId, target: dropoutId },
    { id: `${blockId}_edge_dropout_ffn2`, source: dropoutId, target: ffn2Id },

    // Second residual connection
    { id: `${blockId}_edge_ffn2_add2`, source: ffn2Id, target: add2Id },
    { id: `${blockId}_edge_add2_norm2`, source: add2Id, target: norm2Id },
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { units, ffnUnits, heads }
  };
}

/**
 * Creates a Transformer Decoder block
 * Architecture: Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm
 */
export function createTransformerDecoderBlock(blockId: string, params?: {
  units?: number;
  ffnUnits?: number;
  heads?: number;
}): BlockTemplate {
  const {
    units = 128,
    ffnUnits = 512,
    heads = 8
  } = params || {};

  // Masked self-attention
  const qSelfId = `${blockId}_q_self`;
  const kSelfId = `${blockId}_k_self`;
  const vSelfId = `${blockId}_v_self`;
  const attnSelfOutId = `${blockId}_attn_self_out`;

  // Add & Norm 1
  const add1Id = `${blockId}_add1`;
  const norm1Id = `${blockId}_norm1`;

  // Cross-attention (with encoder output)
  const qCrossId = `${blockId}_q_cross`;
  const kCrossId = `${blockId}_k_cross`;
  const vCrossId = `${blockId}_v_cross`;
  const attnCrossOutId = `${blockId}_attn_cross_out`;

  // Add & Norm 2
  const add2Id = `${blockId}_add2`;
  const norm2Id = `${blockId}_norm2`;

  // FFN
  const ffn1Id = `${blockId}_ffn1`;
  const actId = `${blockId}_act`;
  const dropoutId = `${blockId}_dropout`;
  const ffn2Id = `${blockId}_ffn2`;

  // Add & Norm 3
  const add3Id = `${blockId}_add3`;
  const norm3Id = `${blockId}_norm3`;

  const internalNodes: LayerNode[] = [
    // Masked self-attention
    { id: qSelfId, type: 'Dense', position: { x: -100, y: 0 }, data: { label: 'Q (self)', layerType: 'Dense', params: { units } } },
    { id: kSelfId, type: 'Dense', position: { x: 0, y: 0 }, data: { label: 'K (self)', layerType: 'Dense', params: { units } } },
    { id: vSelfId, type: 'Dense', position: { x: 100, y: 0 }, data: { label: 'V (self)', layerType: 'Dense', params: { units } } },
    { id: attnSelfOutId, type: 'Dense', position: { x: 0, y: 100 }, data: { label: 'Self-Attn Out', layerType: 'Dense', params: { units } } },

    // Add & Norm 1
    { id: add1Id, type: 'Add', position: { x: 0, y: 200 }, data: { label: 'Add', layerType: 'Add', params: {} } },
    { id: norm1Id, type: 'BatchNorm', position: { x: 0, y: 300 }, data: { label: 'LayerNorm', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 1e-6 } } },

    // Cross-attention
    { id: qCrossId, type: 'Dense', position: { x: -100, y: 400 }, data: { label: 'Q (cross)', layerType: 'Dense', params: { units } } },
    { id: kCrossId, type: 'Dense', position: { x: 0, y: 400 }, data: { label: 'K (enc)', layerType: 'Dense', params: { units } } },
    { id: vCrossId, type: 'Dense', position: { x: 100, y: 400 }, data: { label: 'V (enc)', layerType: 'Dense', params: { units } } },
    { id: attnCrossOutId, type: 'Dense', position: { x: 0, y: 500 }, data: { label: 'Cross-Attn Out', layerType: 'Dense', params: { units } } },

    // Add & Norm 2
    { id: add2Id, type: 'Add', position: { x: 0, y: 600 }, data: { label: 'Add', layerType: 'Add', params: {} } },
    { id: norm2Id, type: 'BatchNorm', position: { x: 0, y: 700 }, data: { label: 'LayerNorm', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 1e-6 } } },

    // Feed-forward network
    { id: ffn1Id, type: 'Dense', position: { x: 0, y: 800 }, data: { label: 'FFN', layerType: 'Dense', params: { units: ffnUnits } } },
    { id: actId, type: 'Activation', position: { x: 0, y: 900 }, data: { label: 'GELU', layerType: 'Activation', params: { activation: 'gelu' as ActivationType } } },
    { id: dropoutId, type: 'Dropout', position: { x: 0, y: 1000 }, data: { label: 'Dropout', layerType: 'Dropout', params: { rate: 0.1 } } },
    { id: ffn2Id, type: 'Dense', position: { x: 0, y: 1100 }, data: { label: 'FFN Out', layerType: 'Dense', params: { units } } },

    // Add & Norm 3
    { id: add3Id, type: 'Add', position: { x: 0, y: 1200 }, data: { label: 'Add', layerType: 'Add', params: {} } },
    { id: norm3Id, type: 'BatchNorm', position: { x: 0, y: 1300 }, data: { label: 'LayerNorm', layerType: 'BatchNorm', params: { momentum: 0.99, epsilon: 1e-6 } } },
  ];

  const internalEdges: Edge[] = [
    // Masked self-attention path
    { id: `${blockId}_edge_q_k_self`, source: qSelfId, target: kSelfId },
    { id: `${blockId}_edge_k_v_self`, source: kSelfId, target: vSelfId },
    { id: `${blockId}_edge_v_attn_self`, source: vSelfId, target: attnSelfOutId },

    // First residual
    { id: `${blockId}_edge_attn_add1`, source: attnSelfOutId, target: add1Id },
    { id: `${blockId}_edge_add1_norm1`, source: add1Id, target: norm1Id },

    // Cross-attention path
    { id: `${blockId}_edge_norm1_q_cross`, source: norm1Id, target: qCrossId },
    { id: `${blockId}_edge_q_k_cross`, source: qCrossId, target: kCrossId },
    { id: `${blockId}_edge_k_v_cross`, source: kCrossId, target: vCrossId },
    { id: `${blockId}_edge_v_attn_cross`, source: vCrossId, target: attnCrossOutId },

    // Second residual
    { id: `${blockId}_edge_attn_add2`, source: attnCrossOutId, target: add2Id },
    { id: `${blockId}_edge_add2_norm2`, source: add2Id, target: norm2Id },

    // FFN path
    { id: `${blockId}_edge_norm2_ffn1`, source: norm2Id, target: ffn1Id },
    { id: `${blockId}_edge_ffn1_act`, source: ffn1Id, target: actId },
    { id: `${blockId}_edge_act_dropout`, source: actId, target: dropoutId },
    { id: `${blockId}_edge_dropout_ffn2`, source: dropoutId, target: ffn2Id },

    // Third residual
    { id: `${blockId}_edge_ffn2_add3`, source: ffn2Id, target: add3Id },
    { id: `${blockId}_edge_add3_norm3`, source: add3Id, target: norm3Id },
  ];

  return {
    internalNodes,
    internalEdges,
    defaultParams: { units, ffnUnits, heads }
  };
}

/**
 * Registry of available block templates
 */
export const BLOCK_TEMPLATES: Record<string, (blockId: string, params?: any) => BlockTemplate> = {
  SkipConnection: createSkipConnectionBlock,
  Bottleneck: createBottleneckBlock,
  DepthwiseSeparable: createDepthwiseSeparableBlock,
  InvertedResidual: createInvertedResidualBlock,
  ConvBNRelu: createConvBNReluBlock,
  SE: createSEBlock,
  Fire: createFireModule,
  Dense: createDenseBlock,
  Attention: createAttentionBlock,
  Transformer: createTransformerBlock,
  TransformerEncoder: createTransformerEncoderBlock,
  TransformerDecoder: createTransformerDecoderBlock,
  Inception: createInceptionBlock,
  UNetEncoder: createUNetEncoderBlock,
  UNetDecoder: createUNetDecoderBlock,
  SEResNet: createSEResNetBlock,
  AttentionGate: createAttentionGateBlock,
};

/**
 * Get a block template by type
 */
export function getBlockTemplate(
  blockType: string,
  blockId: string,
  params?: Record<string, any>
): BlockTemplate | null {
  const templateFn = BLOCK_TEMPLATES[blockType as string];
  if (!templateFn) {
    return null;
  }
  return templateFn(blockId, params);
}
