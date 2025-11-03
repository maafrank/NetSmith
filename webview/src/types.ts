// Mirror types from extension
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
  name?: string;
  inputShape?: number[];
  units?: number;
  filters?: number;
  kernelSize?: number | [number, number];
  stride?: number | [number, number];
  padding?: 'valid' | 'same';
  poolSize?: number | [number, number];
  rate?: number;
  momentum?: number;
  epsilon?: number;
  activation?: ActivationType;
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

export interface TrainingConfig {
  datasetPath: string;
  datasetType: 'full' | 'split';
  trainPath?: string;
  valPath?: string;
  testPath?: string;
  splitRatio?: [number, number, number];
  batchSize: number;
  epochs: number;
  learningRate: number;
  optimizer: OptimizerType;
  loss: LossType;
  metrics: MetricType[];
  device: 'cpu' | 'cuda' | 'mps' | 'auto';
  saveCheckpoints: boolean;
  checkpointFrequency?: number;
}

export type OptimizerType = 'adam' | 'sgd' | 'rmsprop' | 'adamw' | 'adagrad';
export type LossType = 'cross_entropy' | 'mse' | 'mae' | 'binary_cross_entropy' | 'huber';
export type MetricType = 'accuracy' | 'precision' | 'recall' | 'f1';

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  valLoss?: number;
  metrics: { [key: string]: number };
  timestamp: number;
}

export interface MessageToWebview {
  type: string;
  [key: string]: any;
}

export interface MessageFromWebview {
  type: string;
  [key: string]: any;
}
