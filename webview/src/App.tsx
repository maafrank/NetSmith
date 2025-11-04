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
  const { nodes, edges, setNodes, setEdges, setSelectedNode, addTrainingMetrics, setIsTraining, setTrainingConfig, setTrainingError, setShowTrainingConfig, deleteNode, toggleBlockExpansion } = useStore();
  const [rfNodes, setRfNodes, onNodesChangeInternal] = useNodesState([]);
  const [rfEdges, setRfEdges, onEdgesChangeInternal] = useEdgesState([]);

  // Wrap onNodesChange to sync deletions back to store
  const onNodesChange = useCallback((changes: any) => {
    onNodesChangeInternal(changes);

    // Check if any nodes were removed
    const removedNodes = changes.filter((change: any) => change.type === 'remove');
    if (removedNodes.length > 0) {
      // Update store to match React Flow state
      removedNodes.forEach((change: any) => {
        deleteNode(change.id);
      });
    }
  }, [onNodesChangeInternal, deleteNode]);

  // Wrap onEdgesChange to sync back to store when edges are deleted
  const onEdgesChange = useCallback((changes: any) => {
    onEdgesChangeInternal(changes);

    // Check if any edges were removed
    const hasRemoval = changes.some((change: any) => change.type === 'remove');
    if (hasRemoval) {
      setTimeout(() => {
        setRfEdges((currentEdges) => {
          setEdges(currentEdges as any);
          return currentEdges;
        });
      }, 0);
    }
  }, [onEdgesChangeInternal, setEdges, setRfEdges]);

  // Panel widths
  const [leftPanelWidth, setLeftPanelWidth] = useState(200); // 12.5rem = 200px
  const [rightPanelWidth, setRightPanelWidth] = useState(200); // 12.5rem = 200px
  const [isResizingLeft, setIsResizingLeft] = useState(false);
  const [isResizingRight, setIsResizingRight] = useState(false);

  // Only sync from store to React Flow when store adds new nodes
  // But preserve React Flow's positions for existing nodes
  useEffect(() => {
    if (nodes.length > rfNodes.length) {
      // New node(s) added - merge with React Flow state
      const rfNodeIds = new Set(rfNodes.map(n => n.id));
      const newNodes = nodes.filter(n => !rfNodeIds.has(n.id));
      setRfNodes([...rfNodes, ...newNodes as any]);
    }
  }, [nodes.length, nodes, rfNodes, setRfNodes]);

  useEffect(() => {
    // Only sync edges from store to React Flow when the count differs significantly
    // This prevents the "undo" effect where React Flow deletions get overwritten
    const edgeDiff = Math.abs(edges.length - rfEdges.length);
    if (edgeDiff > 0) {
      setRfEdges(edges as any);
    }
  }, [edges.length, edges, rfEdges.length, setRfEdges]);

  // Handle block expansion/collapse - add/remove internal nodes and edges
  useEffect(() => {
    let updatedNodes = [...rfNodes];
    let updatedEdges = [...rfEdges];
    let hasChanges = false;

    // Process each node to check for block expansion changes
    nodes.forEach((node) => {
      if (node.data.layerType === 'Block' && node.data.params.internalNodes) {
        const isExpanded = node.data.params.expanded;
        const blockNode = rfNodes.find((n) => n.id === node.id);

        if (!blockNode) return;

        // Get internal node IDs for this block
        const internalNodeIds = new Set(
          node.data.params.internalNodes?.map((n: any) => n.id) || []
        );

        // Check if internal nodes are currently in rfNodes
        const hasInternalNodes = rfNodes.some((n) => internalNodeIds.has(n.id));

        if (isExpanded && !hasInternalNodes) {
          // Block was just expanded - add internal nodes and edges
          const blockPosition = blockNode.position;

          // Calculate bounds for the internal nodes
          const internalNodeCount = node.data.params.internalNodes.length;
          const nodeSpacing = 120;
          const containerPadding = 40;
          const nodeWidth = 200;

          // Position internal nodes in a vertical layout centered relative to block
          const positionedInternalNodes = node.data.params.internalNodes.map(
            (internalNode: any, index: number) => ({
              ...internalNode,
              position: {
                x: blockPosition.x,
                y: blockPosition.y + 120 + (index * nodeSpacing), // Start below block with spacing
              },
              style: {
                ...internalNode.style,
                // Add visual indicator that these are internal to a block
                boxShadow: '0 0 0 2px rgba(6, 182, 212, 0.3)',
              }
            })
          );

          // Add a background container node for visual grouping
          const containerHeight = (internalNodeCount * nodeSpacing) + containerPadding * 2 - 20;
          const containerNode = {
            id: `${node.id}_container`,
            type: 'default',
            position: {
              x: blockPosition.x - containerPadding,
              y: blockPosition.y + 100,
            },
            data: {
              label: '',
              layerType: 'BlockContainer' as any,
              params: {},
            },
            style: {
              width: nodeWidth + containerPadding * 2,
              height: containerHeight,
              backgroundColor: 'rgba(6, 182, 212, 0.08)',
              border: '2px dashed rgba(6, 182, 212, 0.3)',
              borderRadius: '12px',
              zIndex: -1,
              pointerEvents: 'none',
            },
            draggable: false,
            selectable: false,
          };

          updatedNodes = [...updatedNodes, containerNode, ...positionedInternalNodes];

          // Add internal edges
          if (node.data.params.internalEdges) {
            const internalEdgesWithMarkers = node.data.params.internalEdges.map(
              (edge: any) => ({
                ...edge,
                markerEnd: { type: MarkerType.ArrowClosed },
              })
            );
            updatedEdges = [...updatedEdges, ...internalEdgesWithMarkers];
          }

          // Add skip connection edge (from block input to Add node)
          const addNode = node.data.params.internalNodes.find(
            (n: any) => n.data.layerType === 'Add'
          );
          if (addNode) {
            // Find edges connecting TO the block node
            const edgesToBlock = rfEdges.filter((e) => e.target === node.id);
            edgesToBlock.forEach((edgeToBlock) => {
              // Create skip edge from the block's input source to the Add node
              const skipEdge = {
                id: `${node.id}_skip_${edgeToBlock.source}_${addNode.id}`,
                source: edgeToBlock.source,
                target: addNode.id,
                markerEnd: { type: MarkerType.ArrowClosed },
                style: { stroke: '#f59e0b', strokeWidth: 2, strokeDasharray: '5,5' }, // Dashed orange for skip
              };
              updatedEdges.push(skipEdge);
            });

            // Redirect edges TO block to point to first internal node (Conv)
            const firstInternalNode = node.data.params.internalNodes[0];
            updatedEdges = updatedEdges.map((edge) =>
              edge.target === node.id
                ? { ...edge, target: firstInternalNode.id }
                : edge
            );

            // Redirect edges FROM block to come from last internal node (Add)
            const lastInternalNode =
              node.data.params.internalNodes[node.data.params.internalNodes.length - 1];
            updatedEdges = updatedEdges.map((edge) =>
              edge.source === node.id
                ? { ...edge, source: lastInternalNode.id }
                : edge
            );
          }

          hasChanges = true;
        } else if (!isExpanded && hasInternalNodes) {
          // Block was just collapsed - remove internal nodes, container, and edges
          updatedNodes = updatedNodes.filter(
            (n) => !internalNodeIds.has(n.id) && n.id !== `${node.id}_container`
          );

          // Remove internal edges and skip connection edges
          updatedEdges = updatedEdges.filter(
            (edge) =>
              !internalNodeIds.has(edge.source) &&
              !internalNodeIds.has(edge.target) &&
              !edge.id.includes(`${node.id}_skip_`)
          );

          // Restore edges to/from block node
          const firstInternalNode = node.data.params.internalNodes[0];
          const lastInternalNode =
            node.data.params.internalNodes[node.data.params.internalNodes.length - 1];

          updatedEdges = updatedEdges.map((edge) => {
            if (edge.target === firstInternalNode.id) {
              return { ...edge, target: node.id };
            }
            if (edge.source === lastInternalNode.id) {
              return { ...edge, source: node.id };
            }
            return edge;
          });

          hasChanges = true;
        }
      }
    });

    if (hasChanges) {
      setRfNodes(updatedNodes);
      setRfEdges(updatedEdges);
    }
  }, [nodes, rfNodes, rfEdges, setRfNodes, setRfEdges]);

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

  // Delete a node from React Flow
  const deleteNodeFromRF = useCallback((nodeId: string) => {
    setRfNodes((nodes) => nodes.filter((node) => node.id !== nodeId));
    setRfEdges((edges) => edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
  }, [setRfNodes, setRfEdges]);

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
    (params: Connection) => {
      const newEdges = addEdge({ ...params, markerEnd: { type: MarkerType.ArrowClosed } }, rfEdges);
      setRfEdges(newEdges);
      setEdges(newEdges as any);
    },
    [setRfEdges, setEdges, rfEdges]
  );

  // Default edge options
  const defaultEdgeOptions = {
    markerEnd: { type: MarkerType.ArrowClosed },
  };

  // Handle node selection and block expansion
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: any) => {
      setSelectedNode(node);

      // Toggle block expansion if it's a block node
      if (node.data?.layerType === 'Block') {
        toggleBlockExpansion(node.id);
      }
    },
    [setSelectedNode, toggleBlockExpansion]
  );

  // Handle pane click (clicking on the canvas background)
  const onPaneClick = useCallback(() => {
    setShowTrainingConfig(false);
  }, [setShowTrainingConfig]);

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
          setTrainingError(message.error);
          console.error('Training error:', message.error);
          break;

        case 'datasetPathSelected':
          console.log('Dataset path selected:', message.path);
          setTrainingConfig({ datasetPath: message.path });
          break;
      }
    };

    window.addEventListener('message', handleMessage);

    // Notify extension that webview is ready
    vscode.postMessage({ type: 'ready' });

    return () => window.removeEventListener('message', handleMessage);
  }, [addTrainingMetrics, setIsTraining, setTrainingConfig, setTrainingError]);

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
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes}
          defaultEdgeOptions={defaultEdgeOptions}
          defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
          minZoom={0.1}
          maxZoom={2}
          proOptions={{ hideAttribution: true }}
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
        <PropertiesPanel onUpdateNode={updateNodeData} onDeleteNode={deleteNodeFromRF} />
      </div>
    </div>
  );
}

export default App;
