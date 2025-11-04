import { create } from 'zustand';
import { LayerNode, Edge, TrainingMetrics, TrainingConfig } from './types';

interface AppState {
  // Model
  nodes: LayerNode[];
  edges: Edge[];
  selectedNode: LayerNode | null;

  // Training
  isTraining: boolean;
  trainingMetrics: TrainingMetrics[];
  trainingConfig: TrainingConfig;
  trainingError: string | null;
  showTrainingConfig: boolean;

  // Actions
  setNodes: (nodes: LayerNode[]) => void;
  setEdges: (edges: Edge[]) => void;
  setSelectedNode: (node: LayerNode | null) => void;
  addNode: (node: LayerNode) => void;
  updateNode: (id: string, data: Partial<LayerNode['data']>) => void;
  deleteNode: (id: string) => void;
  setTrainingConfig: (config: Partial<TrainingConfig>) => void;
  addTrainingMetrics: (metrics: TrainingMetrics) => void;
  setIsTraining: (isTraining: boolean) => void;
  setTrainingError: (error: string | null) => void;
  clearMetrics: () => void;
  setShowTrainingConfig: (show: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  nodes: [],
  edges: [],
  selectedNode: null,
  isTraining: false,
  trainingMetrics: [],
  trainingError: null,
  showTrainingConfig: false,
  trainingConfig: {
    datasetPath: '',
    datasetType: 'full',
    splitRatio: [0.7, 0.15, 0.15],
    batchSize: 32,
    epochs: 10,
    learningRate: 0.001,
    optimizer: 'adam',
    loss: 'cross_entropy',
    metrics: ['accuracy'],
    device: 'auto',
    saveCheckpoints: true,
    checkpointFrequency: 5,
  },

  // Actions
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  setSelectedNode: (node) => set({ selectedNode: node }),

  addNode: (node) =>
    set((state) => ({
      nodes: [...state.nodes, node],
    })),

  updateNode: (id, data) =>
    set((state) => {
      const updatedNodes = state.nodes.map((node) =>
        node.id === id ? { ...node, data: { ...node.data, ...data } } : node
      );
      const updatedSelectedNode = state.selectedNode?.id === id
        ? { ...state.selectedNode, data: { ...state.selectedNode.data, ...data } }
        : state.selectedNode;

      return {
        nodes: updatedNodes,
        selectedNode: updatedSelectedNode,
      };
    }),

  deleteNode: (id) =>
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== id),
      edges: state.edges.filter((edge) => edge.source !== id && edge.target !== id),
    })),

  setTrainingConfig: (config) =>
    set((state) => ({
      trainingConfig: { ...state.trainingConfig, ...config },
    })),

  addTrainingMetrics: (metrics) =>
    set((state) => ({
      trainingMetrics: [...state.trainingMetrics, metrics],
    })),

  setIsTraining: (isTraining) => set({ isTraining }),

  setTrainingError: (error) => set({ trainingError: error }),

  clearMetrics: () => set({ trainingMetrics: [], trainingError: null }),

  setShowTrainingConfig: (show) => set({ showTrainingConfig: show }),
}));
