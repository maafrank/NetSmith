import * as vscode from 'vscode';
import * as path from 'path';
import { spawn } from 'child_process';

/**
 * Handles ONNX export by invoking the Python onnx_exporter.py script
 */
export class ONNXExporter {
    constructor(private context: vscode.ExtensionContext) {}

    /**
     * Export a trained model to ONNX format
     * @param runPath Path to the training run directory (contains architecture.json and weights.pt)
     * @param outputPath Path where the ONNX file should be saved
     * @returns Promise that resolves when export completes
     */
    async export(runPath: string, outputPath: string): Promise<void> {
        return new Promise((resolve, reject) => {
            // Get Python interpreter path
            const pythonExtension = vscode.extensions.getExtension('ms-python.python');
            let pythonPath = 'python3';

            if (pythonExtension?.isActive) {
                const pythonApi = pythonExtension.exports;
                const activeInterpreter = pythonApi.settings.getExecutionDetails?.(vscode.workspace.workspaceFolders?.[0]?.uri);
                if (activeInterpreter?.execCommand?.[0]) {
                    pythonPath = activeInterpreter.execCommand[0];
                }
            }

            // Path to the Python exporter script
            const scriptPath = path.join(
                this.context.extensionPath,
                'src',
                'python',
                'onnx_exporter.py'
            );

            // Spawn Python process
            const process = spawn(pythonPath, [scriptPath, runPath, outputPath]);

            let stdout = '';
            let stderr = '';

            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            process.on('close', (code) => {
                if (code === 0) {
                    try {
                        // Parse JSON output
                        const result = JSON.parse(stdout.trim());
                        if (result.success) {
                            resolve();
                        } else {
                            reject(new Error(result.error || 'Unknown error during ONNX export'));
                        }
                    } catch (e) {
                        reject(new Error(`Failed to parse export result: ${stdout}\n${stderr}`));
                    }
                } else {
                    reject(new Error(`ONNX export failed:\n${stderr}\n${stdout}`));
                }
            });

            process.on('error', (error) => {
                reject(new Error(`Failed to start Python process: ${error.message}`));
            });
        });
    }

    /**
     * Check if a run has trained weights available
     */
    async hasTrainedWeights(runPath: string): Promise<boolean> {
        try {
            const weightsPath = path.join(runPath, 'weights.pt');
            await vscode.workspace.fs.stat(vscode.Uri.file(weightsPath));
            return true;
        } catch {
            return false;
        }
    }
}
