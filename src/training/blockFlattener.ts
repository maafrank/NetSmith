import { LayerNode, Edge } from '../types';

/**
 * Flattens blocks into their constituent layers for training.
 * This replaces Block nodes with their internal nodes and rewires edges appropriately.
 */

export interface FlattenedArchitecture {
  nodes: LayerNode[];
  edges: Edge[];
}

export function flattenBlocks(nodes: LayerNode[], edges: Edge[]): FlattenedArchitecture {
  let flattenedNodes: LayerNode[] = [];
  let flattenedEdges: Edge[] = [];

  // Map to track block ID -> first and last internal node IDs
  const blockMapping = new Map<string, { first: string; last: string; internal: LayerNode[] }>();

  // First pass: collect all blocks and their internal structure
  nodes.forEach((node) => {
    if (node.data.layerType === 'Block' && node.data.params.internalNodes) {
      const internalNodes = node.data.params.internalNodes;
      const internalEdges = node.data.params.internalEdges || [];

      if (internalNodes.length > 0) {
        // Store mapping for edge rewiring
        blockMapping.set(node.id, {
          first: internalNodes[0].id,
          last: internalNodes[internalNodes.length - 1].id,
          internal: internalNodes,
        });

        // Add internal nodes to flattened list
        flattenedNodes.push(...internalNodes);

        // Add internal edges to flattened list
        flattenedEdges.push(...internalEdges);
      }
    } else {
      // Regular node - add as-is
      flattenedNodes.push(node);
    }
  });

  // Second pass: rewire edges
  edges.forEach((edge) => {
    const sourceIsBlock = blockMapping.has(edge.source);
    const targetIsBlock = blockMapping.has(edge.target);

    if (sourceIsBlock && targetIsBlock) {
      // Both ends are blocks - connect last of source to first of target
      const sourceBlock = blockMapping.get(edge.source)!;
      const targetBlock = blockMapping.get(edge.target)!;

      flattenedEdges.push({
        ...edge,
        id: `${edge.id}_flattened`,
        source: sourceBlock.last,
        target: targetBlock.first,
      });
    } else if (sourceIsBlock) {
      // Source is block - use last internal node as source
      const sourceBlock = blockMapping.get(edge.source)!;

      flattenedEdges.push({
        ...edge,
        id: `${edge.id}_flattened`,
        source: sourceBlock.last,
      });
    } else if (targetIsBlock) {
      // Target is block - use first internal node as target
      const targetBlock = blockMapping.get(edge.target)!;

      flattenedEdges.push({
        ...edge,
        id: `${edge.id}_flattened`,
        target: targetBlock.first,
      });

      // Also add skip connection if block has Add layer
      const addNode = targetBlock.internal.find((n) => n.data.layerType === 'Add');
      if (addNode) {
        flattenedEdges.push({
          id: `${edge.id}_skip`,
          source: edge.source,
          target: addNode.id,
        });
      }
    } else {
      // Neither end is a block - add edge as-is
      flattenedEdges.push(edge);
    }
  });

  return {
    nodes: flattenedNodes,
    edges: flattenedEdges,
  };
}

/**
 * Validates that the flattened architecture is valid for training.
 * Returns an error message if invalid, or null if valid.
 */
export function validateFlattenedArchitecture(
  nodes: LayerNode[],
  edges: Edge[]
): string | null {
  // Check for Input layer
  const hasInput = nodes.some((n) => n.data.layerType === 'Input');
  if (!hasInput) {
    return 'Model must have an Input layer';
  }

  // Check for Output layer
  const hasOutput = nodes.some((n) => n.data.layerType === 'Output');
  if (!hasOutput) {
    return 'Model must have an Output layer';
  }

  // Check that there's at least one trainable layer
  const trainableLayers = nodes.filter(
    (n) =>
      n.data.layerType !== 'Input' &&
      n.data.layerType !== 'Output' &&
      n.data.layerType !== 'Flatten' &&
      n.data.layerType !== 'Activation'
  );

  if (trainableLayers.length === 0) {
    return 'Model must have at least one trainable layer (Dense, Conv2D, etc.)';
  }

  return null;
}
