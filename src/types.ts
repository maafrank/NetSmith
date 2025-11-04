// Core types for NetSmith

export interface LayerNode {
    id: string;
    type: string;
    position: { x: number; y: number };
    data: LayerData;
}

export interface LayerData {
    label: string;
    layerType: LayerType;
    params: LayerParams;
}

export type LayerType =
    | 'Input'
    | 'Dense'
    | 'Conv2D'
    | 'Conv1D'
    | 'MaxPool2D'
    | 'AvgPool2D'
    | 'Flatten'
    | 'Dropout'
    | 'BatchNorm'
    | 'Activation'
    | 'Output'
    | 'Block';

export interface LayerParams {
    // Common params
    name?: string;

    // Input layer
    inputShape?: number[];

    // Dense layer
    units?: number;

    // Conv2D layer
    filters?: number;
    kernelSize?: number | [number, number];
    stride?: number | [number, number];
    padding?: 'valid' | 'same';

    // Pooling
    poolSize?: number | [number, number];

    // Dropout
    rate?: number;

    // BatchNorm
    momentum?: number;
    epsilon?: number;

    // Activation
    activation?: ActivationType;

    // Block reference
    blockId?: string;
}

export type ActivationType =
    | 'relu'
    | 'sigmoid'
    | 'tanh'
    | 'softmax'
    | 'leaky_relu'
    | 'elu'
    | 'selu'
    | 'gelu'
    | 'swish'
    | 'linear';

export interface Edge {
    id: string;
    source: string;
    target: string;
    sourceHandle?: string;
    targetHandle?: string;
}

export interface ModelArchitecture {
    nodes: LayerNode[];
    edges: Edge[];
    blocks: BlockDefinition[];
}

export interface BlockDefinition {
    id: string;
    name: string;
    nodes: LayerNode[];
    edges: Edge[];
    inputs: string[];  // Node IDs for input points
    outputs: string[]; // Node IDs for output points
}

export interface TrainingConfig {
    // Data
    datasetPath: string;
    datasetType: 'full' | 'split';
    trainPath?: string;
    valPath?: string;
    testPath?: string;
    splitRatio?: [number, number, number]; // [train, val, test]
    batchSize: number;

    // Training
    epochs: number;
    learningRate: number;
    optimizer: OptimizerType;
    loss: LossType;
    metrics: MetricType[];

    // Device
    device: 'cpu' | 'cuda' | 'mps' | 'auto';

    // Checkpointing
    saveCheckpoints: boolean;
    checkpointFrequency?: number;
}

export type OptimizerType =
    | 'adam'
    | 'sgd'
    | 'rmsprop'
    | 'adamw'
    | 'adagrad';

export type LossType =
    | 'cross_entropy'
    | 'mse'
    | 'mae'
    | 'binary_cross_entropy'
    | 'huber';

export type MetricType =
    | 'accuracy'
    | 'precision'
    | 'recall'
    | 'f1';

export interface TrainingMetrics {
    epoch: number;
    batch?: number;
    totalBatches?: number;
    loss: number;
    valLoss?: number;
    metrics: { [key: string]: number };
    timestamp: number;
}

export interface ProjectConfig {
    version: string;
    pythonPath?: string;
    defaultDevice: 'cpu' | 'cuda' | 'mps' | 'auto';
    models: ModelReference[];
    runs: RunReference[];
}

export interface ModelReference {
    id: string;
    name: string;
    path: string;
    createdAt: string;
    updatedAt: string;
}

export interface RunReference {
    id: string;
    modelId: string;
    name: string;
    path: string;
    status: 'running' | 'completed' | 'failed' | 'stopped';
    createdAt: string;
    completedAt?: string;
}

// Messages between extension and webview
export type MessageToWebview =
    | { type: 'loadModel'; data: ModelArchitecture }
    | { type: 'trainingMetrics'; data: TrainingMetrics }
    | { type: 'trainingStarted'; runId: string }
    | { type: 'trainingStopped' }
    | { type: 'trainingCompleted' }
    | { type: 'trainingError'; error: string }
    | { type: 'loadBlocks'; blocks: BlockDefinition[] }
    | { type: 'datasetPathSelected'; path: string }
    | { type: 'availableDatasets'; datasets: string[] };

export type MessageFromWebview =
    | { type: 'saveModel'; data: ModelArchitecture; modelName?: string }
    | { type: 'runModel'; config: TrainingConfig; architecture: ModelArchitecture }
    | { type: 'stopTraining' }
    | { type: 'exportModel'; format: 'pytorch' | 'onnx' }
    | { type: 'saveBlock'; block: BlockDefinition }
    | { type: 'loadBlocks' }
    | { type: 'ready' }
    | { type: 'pickDatasetFile' }
    | { type: 'scanForDatasets' };
