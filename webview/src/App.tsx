import { useEffect, useCallback } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  BackgroundVariant,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { useStore } from './store';
import { vscode } from './vscode';
import LayerPalette from './components/LayerPalette';
import PropertiesPanel from './components/PropertiesPanel';
import TrainingPanel from './components/TrainingPanel';
import MetricsPanel from './components/MetricsPanel';
import { nodeTypes } from './components/nodes';

function App() {
  const { nodes, edges, setNodes, setEdges, setSelectedNode, addTrainingMetrics, setIsTraining } = useStore();
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState(nodes);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState(edges);

  // Sync store with React Flow
  useEffect(() => {
    setRfNodes(nodes);
  }, [nodes, setRfNodes]);

  useEffect(() => {
    setRfEdges(edges);
  }, [edges, setRfEdges]);

  useEffect(() => {
    setNodes(rfNodes as any);
  }, [rfNodes, setNodes]);

  useEffect(() => {
    setEdges(rfEdges as any);
  }, [rfEdges, setEdges]);

  // Handle connections
  const onConnect = useCallback(
    (params: Connection) => setRfEdges((eds) => addEdge(params, eds)),
    [setRfEdges]
  );

  // Handle node selection
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: any) => {
      setSelectedNode(node);
    },
    [setSelectedNode]
  );

  // Handle messages from extension
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const message = event.data;

      switch (message.type) {
        case 'trainingMetrics':
          addTrainingMetrics(message.data);
          break;

        case 'trainingStarted':
          setIsTraining(true);
          break;

        case 'trainingStopped':
        case 'trainingCompleted':
          setIsTraining(false);
          break;

        case 'trainingError':
          setIsTraining(false);
          console.error('Training error:', message.error);
          break;
      }
    };

    window.addEventListener('message', handleMessage);

    // Notify extension that webview is ready
    vscode.postMessage({ type: 'ready' });

    return () => window.removeEventListener('message', handleMessage);
  }, [addTrainingMetrics, setIsTraining]);

  return (
    <div className="w-full h-full flex">
      {/* Left sidebar - Layer Palette */}
      <div className="w-64 border-r border-gray-700 bg-gray-900 overflow-y-auto">
        <LayerPalette />
      </div>

      {/* Main canvas */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          fitView
          attributionPosition="bottom-left"
        >
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          <Controls />
          <MiniMap />

          <Panel position="top-right" className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <TrainingPanel />
          </Panel>
        </ReactFlow>

        {/* Metrics overlay when training */}
        <MetricsPanel />
      </div>

      {/* Right sidebar - Properties Panel */}
      <div className="w-80 border-l border-gray-700 bg-gray-900 overflow-y-auto">
        <PropertiesPanel />
      </div>
    </div>
  );
}

export default App;
