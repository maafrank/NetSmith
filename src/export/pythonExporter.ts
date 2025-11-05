import { LayerNode, Edge, LayerType, LayerParams } from '../types';

/**
 * Generates a standalone PyTorch Python script from a model architecture
 */
export class PythonExporter {
    private nodes: LayerNode[];
    private edges: Edge[];
    private layerOrder: string[];

    constructor(nodes: LayerNode[], edges: Edge[]) {
        this.nodes = nodes;
        this.edges = edges;
        this.layerOrder = this.topologicalSort();
    }

    /**
     * Generate the complete Python script
     */
    generate(): string {
        const sections = [
            this.generateImports(),
            this.generateModelClass(),
            this.generateUsageExample(),
        ];

        return sections.join('\n\n');
    }

    private generateImports(): string {
        return `import torch
import torch.nn as nn
import torch.nn.functional as F`;
    }

    private generateModelClass(): string {
        const className = 'ExportedModel';
        const initMethod = this.generateInitMethod();
        const forwardMethod = this.generateForwardMethod();

        return `class ${className}(nn.Module):
    """
    PyTorch model exported from NetSmith
    Architecture: ${this.nodes.length} layers, ${this.edges.length} connections
    """

    def __init__(self):
        super(${className}, self).__init__()
        ${initMethod}

    def forward(self, x):
        ${forwardMethod}
        return x`;
    }

    private generateInitMethod(): string {
        const layerDefs: string[] = [];

        for (const nodeId of this.layerOrder) {
            const node = this.nodes.find(n => n.id === nodeId);
            if (!node) continue;

            const layerType = node.data.layerType;
            const params = node.data.params;
            const varName = this.sanitizeVarName(nodeId);

            // Skip Input/Output layers (no trainable params)
            if (layerType === 'Input' || layerType === 'Output') {
                continue;
            }

            const layerCode = this.generateLayerInit(layerType, params, varName);
            if (layerCode) {
                layerDefs.push(`self.${varName} = ${layerCode}`);
            }
        }

        return layerDefs.join('\n        ');
    }

    private generateLayerInit(layerType: LayerType, params: LayerParams, varName: string): string | null {
        switch (layerType) {
            case 'Dense':
                return `nn.LazyLinear(${params.units || 128})`;

            case 'Conv2D':
                const filters = params.filters || 32;
                const kernelSize = Array.isArray(params.kernelSize)
                    ? `(${params.kernelSize[0]}, ${params.kernelSize[1]})`
                    : params.kernelSize || 3;
                const stride = Array.isArray(params.stride)
                    ? `(${params.stride[0]}, ${params.stride[1]})`
                    : params.stride || 1;
                const padding = params.padding === 'same' ? `'same'` : 0;
                return `nn.LazyConv2d(${filters}, kernel_size=${kernelSize}, stride=${stride}, padding=${padding})`;

            case 'Conv1D':
                return `nn.LazyConv1d(${params.filters || 32}, kernel_size=${params.kernelSize || 3}, stride=${params.stride || 1})`;

            case 'BatchNorm':
                return `nn.LazyBatchNorm2d(momentum=${params.momentum || 0.1}, eps=${params.epsilon || 1e-5})`;

            case 'Dropout':
                return `nn.Dropout(p=${params.rate || 0.5})`;

            case 'MaxPool2D':
            case 'AvgPool2D':
                const poolSize = Array.isArray(params.poolSize)
                    ? `(${params.poolSize[0]}, ${params.poolSize[1]})`
                    : params.poolSize || 2;
                const poolType = layerType === 'MaxPool2D' ? 'MaxPool2d' : 'AvgPool2d';
                return `nn.${poolType}(kernel_size=${poolSize})`;

            case 'Flatten':
                // No parameters needed, handled in forward
                return null;

            case 'GlobalAvgPool2D':
            case 'GlobalMaxPool2D':
                // No parameters needed, handled in forward
                return null;

            case 'Activation':
                // Most activations are functional, no params needed
                return null;

            case 'Add':
            case 'Concat':
            case 'Multiply':
            case 'Subtract':
            case 'Maximum':
            case 'Minimum':
                // Merge layers handled in forward
                return null;

            case 'Reshape':
            case 'UpSampling2D':
                // Handled in forward
                return null;

            default:
                return `# TODO: Implement ${layerType} layer`;
        }
    }

    private generateForwardMethod(): string {
        const forwardLines: string[] = [];
        const outputs: Map<string, string> = new Map();

        // Build adjacency map
        const inputMap = this.buildInputMap();

        for (const nodeId of this.layerOrder) {
            const node = this.nodes.find(n => n.id === nodeId);
            if (!node) continue;

            const layerType = node.data.layerType;
            const params = node.data.params;
            const varName = this.sanitizeVarName(nodeId);
            const inputNodes = inputMap.get(nodeId) || [];

            let outputVar: string;

            if (layerType === 'Input') {
                outputs.set(nodeId, 'x');
                continue;
            }

            // Determine input variable
            let inputVar = 'x';
            if (inputNodes.length > 0) {
                inputVar = outputs.get(inputNodes[0]) || 'x';
            }

            // Generate layer forward pass
            const forwardCode = this.generateLayerForward(layerType, params, varName, inputVar, inputNodes, outputs);

            if (forwardCode) {
                outputVar = `x_${varName}`;
                forwardLines.push(`${outputVar} = ${forwardCode}`);
                outputs.set(nodeId, outputVar);
            } else {
                outputs.set(nodeId, inputVar);
            }
        }

        // Update final output to 'x' for return
        if (forwardLines.length > 0) {
            const lastLine = forwardLines[forwardLines.length - 1];
            forwardLines[forwardLines.length - 1] = lastLine.replace(/^x_\w+/, 'x');
        }

        return forwardLines.join('\n        ');
    }

    private generateLayerForward(
        layerType: LayerType,
        params: LayerParams,
        varName: string,
        inputVar: string,
        inputNodes: string[],
        outputs: Map<string, string>
    ): string | null {
        switch (layerType) {
            case 'Dense':
            case 'Conv2D':
            case 'Conv1D':
            case 'BatchNorm':
            case 'Dropout':
            case 'MaxPool2D':
            case 'AvgPool2D':
                return `self.${varName}(${inputVar})`;

            case 'Flatten':
                return `torch.flatten(${inputVar}, start_dim=1)`;

            case 'GlobalAvgPool2D':
                return `F.adaptive_avg_pool2d(${inputVar}, (1, 1)).squeeze(-1).squeeze(-1)`;

            case 'GlobalMaxPool2D':
                return `F.adaptive_max_pool2d(${inputVar}, (1, 1)).squeeze(-1).squeeze(-1)`;

            case 'Activation':
                const activation = params.activation || 'relu';
                return this.generateActivation(activation, inputVar);

            case 'Add':
                if (inputNodes.length >= 2) {
                    const input1 = outputs.get(inputNodes[0]) || inputVar;
                    const input2 = outputs.get(inputNodes[1]) || inputVar;
                    return `${input1} + ${input2}`;
                }
                return inputVar;

            case 'Concat':
                if (inputNodes.length >= 2) {
                    const inputs = inputNodes.map(id => outputs.get(id) || 'x').join(', ');
                    return `torch.cat([${inputs}], dim=1)`;
                }
                return inputVar;

            case 'Multiply':
                if (inputNodes.length >= 2) {
                    const input1 = outputs.get(inputNodes[0]) || inputVar;
                    const input2 = outputs.get(inputNodes[1]) || inputVar;
                    return `${input1} * ${input2}`;
                }
                return inputVar;

            case 'Subtract':
                if (inputNodes.length >= 2) {
                    const input1 = outputs.get(inputNodes[0]) || inputVar;
                    const input2 = outputs.get(inputNodes[1]) || inputVar;
                    return `${input1} - ${input2}`;
                }
                return inputVar;

            case 'Maximum':
                if (inputNodes.length >= 2) {
                    const input1 = outputs.get(inputNodes[0]) || inputVar;
                    const input2 = outputs.get(inputNodes[1]) || inputVar;
                    return `torch.maximum(${input1}, ${input2})`;
                }
                return inputVar;

            case 'Minimum':
                if (inputNodes.length >= 2) {
                    const input1 = outputs.get(inputNodes[0]) || inputVar;
                    const input2 = outputs.get(inputNodes[1]) || inputVar;
                    return `torch.minimum(${input1}, ${input2})`;
                }
                return inputVar;

            case 'Reshape':
                // TODO: Implement reshape
                return `${inputVar}  # TODO: Add reshape logic`;

            case 'UpSampling2D':
                return `F.interpolate(${inputVar}, scale_factor=2, mode='nearest')`;

            case 'Output':
                return null;

            default:
                return `${inputVar}  # TODO: Implement ${layerType}`;
        }
    }

    private generateActivation(activation: string, inputVar: string): string {
        switch (activation) {
            case 'relu':
                return `F.relu(${inputVar})`;
            case 'sigmoid':
                return `torch.sigmoid(${inputVar})`;
            case 'tanh':
                return `torch.tanh(${inputVar})`;
            case 'softmax':
                return `F.softmax(${inputVar}, dim=1)`;
            case 'leaky_relu':
                return `F.leaky_relu(${inputVar})`;
            case 'elu':
                return `F.elu(${inputVar})`;
            case 'selu':
                return `F.selu(${inputVar})`;
            case 'gelu':
                return `F.gelu(${inputVar})`;
            case 'swish':
                return `${inputVar} * torch.sigmoid(${inputVar})`;
            case 'linear':
                return inputVar;
            default:
                return `F.relu(${inputVar})`;
        }
    }

    private generateUsageExample(): string {
        return `# Usage example:
# model = ExportedModel()
#
# # Initialize lazy modules with a forward pass
# dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape as needed
# _ = model(dummy_input)
#
# # Now you can use the model
# output = model(your_input)
# print(f"Parameters: {sum(p.numel() for p in model.parameters())}")`;
    }

    /**
     * Topological sort to determine layer execution order
     */
    private topologicalSort(): string[] {
        const visited = new Set<string>();
        const result: string[] = [];
        const adjList = new Map<string, string[]>();

        // Build adjacency list
        for (const edge of this.edges) {
            if (!adjList.has(edge.source)) {
                adjList.set(edge.source, []);
            }
            adjList.get(edge.source)!.push(edge.target);
        }

        const dfs = (nodeId: string) => {
            if (visited.has(nodeId)) return;
            visited.add(nodeId);

            const neighbors = adjList.get(nodeId) || [];
            for (const neighbor of neighbors) {
                dfs(neighbor);
            }

            result.unshift(nodeId);
        };

        // Find all nodes with no incoming edges (start nodes)
        const allTargets = new Set(this.edges.map(e => e.target));
        const startNodes = this.nodes.filter(n => !allTargets.has(n.id));

        for (const node of startNodes) {
            dfs(node.id);
        }

        return result;
    }

    /**
     * Build a map of nodeId -> [input node IDs]
     */
    private buildInputMap(): Map<string, string[]> {
        const inputMap = new Map<string, string[]>();

        for (const edge of this.edges) {
            if (!inputMap.has(edge.target)) {
                inputMap.set(edge.target, []);
            }
            inputMap.get(edge.target)!.push(edge.source);
        }

        return inputMap;
    }

    /**
     * Sanitize node ID to valid Python variable name
     */
    private sanitizeVarName(nodeId: string): string {
        return nodeId.replace(/[^a-zA-Z0-9_]/g, '_').toLowerCase();
    }
}
