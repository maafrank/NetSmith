import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';
import { spawn, ChildProcess } from 'child_process';
import { TrainingConfig, ModelArchitecture, TrainingMetrics } from '../types';

export class TrainingManager {
    private currentProcess: ChildProcess | null = null;
    private currentRunId: string | null = null;
    private metricsCallback: ((metrics: TrainingMetrics) => void) | null = null;

    constructor(private context: vscode.ExtensionContext) {}

    async startTraining(
        workspaceUri: vscode.Uri,
        runPath: string,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        onMetrics: (metrics: TrainingMetrics) => void,
        onComplete: () => void,
        onError: (error: string) => void
    ): Promise<void> {
        this.metricsCallback = onMetrics;
        let stderrBuffer = '';

        // Save architecture and config to run directory
        await fs.writeFile(
            path.join(runPath, 'architecture.json'),
            JSON.stringify(architecture, null, 2)
        );
        await fs.writeFile(
            path.join(runPath, 'config.json'),
            JSON.stringify(config, null, 2)
        );

        // Get Python path
        const pythonPath = await this.getPythonPath();
        const runnerPath = path.join(this.context.extensionPath, 'src', 'python', 'runner.py');

        // Spawn Python process
        this.currentProcess = spawn(pythonPath, [runnerPath, runPath], {
            cwd: workspaceUri.fsPath,
            env: { ...process.env }
        });

        // Handle stdout (metrics)
        this.currentProcess.stdout?.on('data', (data) => {
            const lines = data.toString().split('\n');
            for (const line of lines) {
                if (line.trim().startsWith('METRICS:')) {
                    try {
                        const metricsJson = line.substring('METRICS:'.length).trim();
                        const metrics: TrainingMetrics = JSON.parse(metricsJson);
                        onMetrics(metrics);
                    } catch (e) {
                        console.error('Failed to parse metrics:', e);
                    }
                } else if (line.trim()) {
                    console.log('Training:', line);
                }
            }
        });

        // Handle stderr
        this.currentProcess.stderr?.on('data', (data) => {
            const errorText = data.toString();
            stderrBuffer += errorText;
            console.error('Training error:', errorText);
        });

        // Handle completion
        this.currentProcess.on('close', (code) => {
            if (code === 0) {
                onComplete();
            } else {
                // Send the captured stderr as the error message
                const errorMessage = stderrBuffer.trim() || `Training process exited with code ${code}`;
                onError(errorMessage);
            }
            this.currentProcess = null;
        });

        // Handle errors
        this.currentProcess.on('error', (err) => {
            onError(`Failed to start training: ${err.message}`);
            this.currentProcess = null;
        });
    }

    stopTraining(): void {
        if (this.currentProcess) {
            this.currentProcess.kill('SIGTERM');
            this.currentProcess = null;
        }
    }

    isTraining(): boolean {
        return this.currentProcess !== null;
    }

    private async getPythonPath(): Promise<string> {
        // Try to use Python extension API
        try {
            const pythonExtension = vscode.extensions.getExtension('ms-python.python');
            if (pythonExtension) {
                await pythonExtension.activate();
                const pythonApi = pythonExtension.exports;

                // Get active interpreter path
                const activeInterpreter = pythonApi.settings.getExecutionDetails?.();
                if (activeInterpreter?.execCommand?.[0]) {
                    return activeInterpreter.execCommand[0];
                }
            }
        } catch (e) {
            console.error('Failed to get Python path from extension:', e);
        }

        // Fallback to 'python3'
        return 'python3';
    }

    async exportModelToPyTorch(
        architecture: ModelArchitecture,
        outputPath: string
    ): Promise<void> {
        const code = this.generatePyTorchCode(architecture);
        await fs.writeFile(outputPath, code);
    }

    private generatePyTorchCode(architecture: ModelArchitecture): string {
        const lines: string[] = [
            'import torch',
            'import torch.nn as nn',
            'import torch.nn.functional as F',
            '',
            '',
            'class GeneratedModel(nn.Module):',
            '    def __init__(self):',
            '        super(GeneratedModel, self).__init__()',
            ''
        ];

        // Sort nodes topologically
        const sortedNodes = this.topologicalSort(architecture.nodes, architecture.edges);

        // Generate layer definitions
        const layerDefs: string[] = [];
        for (const node of sortedNodes) {
            if (node.data.layerType === 'Input' || node.data.layerType === 'Output') {
                continue;
            }

            const layerCode = this.generateLayerCode(node);
            if (layerCode) {
                layerDefs.push(`        self.${node.id} = ${layerCode}`);
            }
        }

        lines.push(...layerDefs);
        lines.push('');
        lines.push('    def forward(self, x):');

        // Generate forward pass
        const forwardPass: string[] = [];
        for (const node of sortedNodes) {
            if (node.data.layerType === 'Input') {
                continue;
            }

            const forwardCode = this.generateForwardCode(node, architecture.edges);
            if (forwardCode) {
                forwardPass.push(`        ${forwardCode}`);
            }
        }

        lines.push(...forwardPass);
        lines.push('        return x');
        lines.push('');

        return lines.join('\n');
    }

    private generateLayerCode(node: any): string | null {
        const params = node.data.params;

        switch (node.data.layerType) {
            case 'Dense':
                return `nn.Linear(${params.units || 128}, ${params.units || 128})`;

            case 'Conv2D':
                const kernel = Array.isArray(params.kernelSize)
                    ? params.kernelSize
                    : [params.kernelSize || 3, params.kernelSize || 3];
                const padding = params.padding === 'same' ? 'same' : 0;
                return `nn.Conv2d(in_channels=3, out_channels=${params.filters || 32}, kernel_size=${kernel[0]}, padding='${padding}')`;

            case 'MaxPool2D':
                const poolSize = Array.isArray(params.poolSize) ? params.poolSize[0] : (params.poolSize || 2);
                return `nn.MaxPool2d(kernel_size=${poolSize})`;

            case 'Dropout':
                return `nn.Dropout(p=${params.rate || 0.5})`;

            case 'BatchNorm':
                return `nn.BatchNorm2d(num_features=64)`;  // This needs to be dynamic

            case 'Flatten':
                return `nn.Flatten()`;

            default:
                return null;
        }
    }

    private generateForwardCode(node: any, edges: any[]): string | null {
        const params = node.data.params;

        switch (node.data.layerType) {
            case 'Activation':
                const activation = params.activation || 'relu';
                if (activation === 'relu') {
                    return `x = F.relu(x)`;
                } else if (activation === 'sigmoid') {
                    return `x = torch.sigmoid(x)`;
                } else if (activation === 'softmax') {
                    return `x = F.softmax(x, dim=1)`;
                } else {
                    return `x = F.${activation}(x)`;
                }

            case 'Flatten':
                return `x = self.${node.id}(x)`;

            case 'Output':
                return null;

            default:
                return `x = self.${node.id}(x)`;
        }
    }

    private topologicalSort(nodes: any[], edges: any[]): any[] {
        // Simple topological sort
        const sorted: any[] = [];
        const visited = new Set<string>();
        const adjacency = new Map<string, string[]>();

        // Build adjacency list
        for (const edge of edges) {
            if (!adjacency.has(edge.source)) {
                adjacency.set(edge.source, []);
            }
            adjacency.get(edge.source)!.push(edge.target);
        }

        // Find input node
        const inputNode = nodes.find(n => n.data.layerType === 'Input');
        if (!inputNode) {
            return nodes;
        }

        // DFS
        const visit = (nodeId: string) => {
            if (visited.has(nodeId)) {
                return;
            }

            visited.add(nodeId);
            const node = nodes.find(n => n.id === nodeId);
            if (node) {
                sorted.push(node);
            }

            const neighbors = adjacency.get(nodeId) || [];
            for (const neighbor of neighbors) {
                visit(neighbor);
            }
        };

        visit(inputNode.id);

        return sorted;
    }
}
