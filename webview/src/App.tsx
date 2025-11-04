import { useEffect, useCallback, useState } from 'react';
import ReactFlow, {
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  BackgroundVariant,
  Panel,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from 'dagre';

import { useStore } from './store';
import { vscode } from './vscode';
import LayerPalette from './components/LayerPalette';
import PropertiesPanel from './components/PropertiesPanel';
import TrainingPanel from './components/TrainingPanel';
import MetricsPanel from './components/MetricsPanel';
import { nodeTypes } from './components/nodes';

function App() {
  const { nodes, edges, setNodes, setEdges, setSelectedNode, addTrainingMetrics, setIsTraining, setTrainingConfig } = useStore();
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState([]);

  // Panel widths
  const [leftPanelWidth, setLeftPanelWidth] = useState(256); // 16rem = 256px (w-64)
  const [rightPanelWidth, setRightPanelWidth] = useState(320); // 20rem = 320px (w-80)
  const [isResizingLeft, setIsResizingLeft] = useState(false);
  const [isResizingRight, setIsResizingRight] = useState(false);

  // Only sync from store to React Flow when store adds new nodes
  useEffect(() => {
    if (nodes.length > rfNodes.length) {
      setRfNodes(nodes as any);
    }
  }, [nodes.length]);

  useEffect(() => {
    if (edges.length !== rfEdges.length) {
      setRfEdges(edges as any);
    }
  }, [edges.length]);

  // Sync React Flow changes back to store (for saves/exports)
  const syncToStore = useCallback(() => {
    setNodes(rfNodes as any);
    setEdges(rfEdges as any);
  }, [rfNodes, rfEdges, setNodes, setEdges]);

  // Update a node's data in React Flow
  const updateNodeData = useCallback((nodeId: string, newData: any) => {
    setRfNodes((nodes) =>
      nodes.map((node) =>
        node.id === nodeId ? { ...node, data: { ...node.data, ...newData } } : node
      )
    );
  }, [setRfNodes]);

  // Auto-layout using Dagre
  const autoLayout = useCallback(() => {
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    dagreGraph.setGraph({ rankdir: 'TB', nodesep: 100, ranksep: 150 });

    // Add nodes to dagre
    rfNodes.forEach((node) => {
      dagreGraph.setNode(node.id, { width: 200, height: 100 });
    });

    // Add edges to dagre
    rfEdges.forEach((edge) => {
      dagreGraph.setEdge(edge.source, edge.target);
    });

    // Calculate layout
    dagre.layout(dagreGraph);

    // Apply positions
    const layoutedNodes = rfNodes.map((node) => {
      const nodeWithPosition = dagreGraph.node(node.id);
      return {
        ...node,
        position: {
          x: nodeWithPosition.x - 100,
          y: nodeWithPosition.y - 50,
        },
      };
    });

    setRfNodes(layoutedNodes);
    syncToStore();
  }, [rfNodes, rfEdges, setRfNodes, syncToStore]);

  // Handle connections
  const onConnect = useCallback(
    (params: Connection) => setRfEdges((eds) => addEdge({ ...params, markerEnd: { type: MarkerType.ArrowClosed } }, eds)),
    [setRfEdges]
  );

  // Default edge options
  const defaultEdgeOptions = {
    markerEnd: { type: MarkerType.ArrowClosed },
  };

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

        case 'datasetPathSelected':
          setTrainingConfig({ datasetPath: message.path });
          break;
      }
    };

    window.addEventListener('message', handleMessage);

    // Notify extension that webview is ready
    vscode.postMessage({ type: 'ready' });

    return () => window.removeEventListener('message', handleMessage);
  }, [addTrainingMetrics, setIsTraining, setTrainingConfig]);

  // Handle left panel resize
  const handleLeftMouseDown = useCallback(() => {
    setIsResizingLeft(true);
  }, []);

  const handleLeftMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isResizingLeft) {
        const newWidth = Math.max(50, Math.min(600, e.clientX));
        setLeftPanelWidth(newWidth);
      }
    },
    [isResizingLeft]
  );

  const handleLeftMouseUp = useCallback(() => {
    setIsResizingLeft(false);
  }, []);

  // Handle right panel resize
  const handleRightMouseDown = useCallback(() => {
    setIsResizingRight(true);
  }, []);

  const handleRightMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isResizingRight) {
        const newWidth = Math.max(50, Math.min(600, window.innerWidth - e.clientX));
        setRightPanelWidth(newWidth);
      }
    },
    [isResizingRight]
  );

  const handleRightMouseUp = useCallback(() => {
    setIsResizingRight(false);
  }, []);

  // Add/remove event listeners for resizing
  useEffect(() => {
    if (isResizingLeft) {
      document.addEventListener('mousemove', handleLeftMouseMove);
      document.addEventListener('mouseup', handleLeftMouseUp);
      document.body.style.cursor = 'ew-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleLeftMouseMove);
      document.removeEventListener('mouseup', handleLeftMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleLeftMouseMove);
      document.removeEventListener('mouseup', handleLeftMouseUp);
    };
  }, [isResizingLeft, handleLeftMouseMove, handleLeftMouseUp]);

  useEffect(() => {
    if (isResizingRight) {
      document.addEventListener('mousemove', handleRightMouseMove);
      document.addEventListener('mouseup', handleRightMouseUp);
      document.body.style.cursor = 'ew-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleRightMouseMove);
      document.removeEventListener('mouseup', handleRightMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleRightMouseMove);
      document.removeEventListener('mouseup', handleRightMouseUp);
    };
  }, [isResizingRight, handleRightMouseMove, handleRightMouseUp]);

  return (
    <div className="w-full h-full flex">
      {/* Left sidebar - Layer Palette */}
      <div
        style={{ width: `${leftPanelWidth}px` }}
        className="border-r border-gray-700 bg-gray-900 overflow-y-auto flex-shrink-0"
      >
        <LayerPalette />
      </div>

      {/* Left resize handle */}
      <div
        onMouseDown={handleLeftMouseDown}
        className="w-1 bg-gray-700 hover:bg-blue-500 cursor-ew-resize flex-shrink-0 transition-colors"
        style={{ cursor: 'ew-resize' }}
      />

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
          defaultEdgeOptions={defaultEdgeOptions}
          fitView
          attributionPosition="bottom-left"
        >
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          <Controls />

          <Panel position="top-left" className="m-2">
            <button
              onClick={autoLayout}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg shadow-lg transition-colors font-medium"
              title="Auto-arrange nodes"
            >
              🔄 Auto Layout
            </button>
          </Panel>

          <Panel position="top-right" className="bg-gray-800 p-4 rounded-lg shadow-lg">
            <TrainingPanel onBeforeRun={syncToStore} nodes={rfNodes as any} edges={rfEdges as any} />
          </Panel>
        </ReactFlow>

        {/* Metrics overlay when training */}
        <MetricsPanel />
      </div>

      {/* Right resize handle */}
      <div
        onMouseDown={handleRightMouseDown}
        className="w-1 bg-gray-700 hover:bg-blue-500 cursor-ew-resize flex-shrink-0 transition-colors"
        style={{ cursor: 'ew-resize' }}
      />

      {/* Right sidebar - Properties Panel */}
      <div
        style={{ width: `${rightPanelWidth}px` }}
        className="border-l border-gray-700 bg-gray-900 overflow-y-auto flex-shrink-0"
      >
        <PropertiesPanel onUpdateNode={updateNodeData} />
      </div>
    </div>
  );
}

export default App;
