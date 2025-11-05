import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';
import { ProjectManager } from '../project/ProjectManager';
import { TrainingManager } from '../training/TrainingManager';
import { PythonExporter } from '../export/pythonExporter';
import { ONNXExporter } from '../export/onnxExporter';
import { MessageFromWebview, MessageToWebview, ModelArchitecture, TrainingConfig } from '../types';

export class ModelBuilderPanel {
    public static currentPanel: ModelBuilderPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private _disposables: vscode.Disposable[] = [];
    private currentModelId: string = `model_${Date.now()}`;
    private onnxExporter: ONNXExporter;

    private constructor(
        panel: vscode.WebviewPanel,
        extensionUri: vscode.Uri,
        private projectManager: ProjectManager,
        private trainingManager: TrainingManager,
        private context: vscode.ExtensionContext
    ) {
        this._panel = panel;
        this.onnxExporter = new ONNXExporter(context);

        this._panel.webview.html = this._getHtmlForWebview(this._panel.webview, extensionUri);

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        this._panel.webview.onDidReceiveMessage(
            async (message: MessageFromWebview) => {
                await this._handleMessage(message);
            },
            null,
            this._disposables
        );
    }

    public static createOrShow(
        extensionUri: vscode.Uri,
        projectManager: ProjectManager,
        trainingManager: TrainingManager,
        context: vscode.ExtensionContext
    ) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        if (ModelBuilderPanel.currentPanel) {
            ModelBuilderPanel.currentPanel._panel.reveal(column);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'netsmithModelBuilder',
            'NetSmith Model Builder',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(extensionUri, 'dist'),
                    vscode.Uri.joinPath(extensionUri, 'webview', 'dist')
                ],
                retainContextWhenHidden: true
            }
        );

        ModelBuilderPanel.currentPanel = new ModelBuilderPanel(
            panel,
            extensionUri,
            projectManager,
            trainingManager,
            context
        );
    }

    public async runModel() {
        this._panel.webview.postMessage({ type: 'requestRunModel' } as any);
    }

    private async _handleMessage(message: MessageFromWebview) {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder open');
            return;
        }

        switch (message.type) {
            case 'ready':
                // Webview is ready, scan for datasets
                await this._scanForDatasets(workspaceFolder.uri);
                break;

            case 'saveModel':
                await this._saveModel(workspaceFolder.uri, message.data, message.modelName);
                break;

            case 'runModel':
                await this._runModel(workspaceFolder.uri, message.architecture, message.config);
                break;

            case 'stopTraining':
                this.trainingManager.stopTraining();
                this._sendMessage({ type: 'trainingStopped' });
                break;

            case 'exportModel':
                await this._exportModel(message.format, workspaceFolder.uri, message.data, message.selectedRun);
                break;

            case 'requestAvailableRuns':
                await this._sendAvailableRuns(workspaceFolder.uri);
                break;

            case 'saveBlock':
                // TODO: Implement block saving
                break;

            case 'loadBlocks':
                // TODO: Implement block loading
                this._sendMessage({ type: 'loadBlocks', blocks: [] });
                break;

            case 'pickDatasetFile':
                console.log('Received pickDatasetFile message');
                await this._pickDatasetFile();
                break;

            case 'scanForDatasets':
                await this._scanForDatasets(workspaceFolder.uri);
                break;
        }
    }

    private async _scanForDatasets(workspaceUri: vscode.Uri) {
        try {
            console.log('=== SCANNING FOR DATASETS ===');
            console.log('Workspace URI:', workspaceUri.fsPath);

            // Find dataset files in workspace
            const patterns = ['**/*.npz', '**/*.pt', '**/*.pth', '**/*.h5', '**/*.hdf5'];
            const datasets: string[] = [];

            for (const pattern of patterns) {
                console.log('Searching pattern:', pattern);
                // Use null as exclude pattern to search everything (respects .gitignore and files.exclude)
                const files = await vscode.workspace.findFiles(pattern, null, 100);
                console.log(`Found ${files.length} files for pattern ${pattern}`);
                for (const file of files) {
                    // Convert to relative path from workspace root
                    const relativePath = vscode.workspace.asRelativePath(file);
                    datasets.push(relativePath);
                }
            }

            console.log('Total datasets found:', datasets.length);
            console.log('Dataset paths:', datasets);

            this._sendMessage({
                type: 'availableDatasets',
                datasets
            } as any);
        } catch (error) {
            console.error('Error scanning for datasets:', error);
            this._sendMessage({
                type: 'availableDatasets',
                datasets: []
            } as any);
        }
    }

    private async _pickDatasetFile() {
        console.log('Opening file picker dialog...');

        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        const options: vscode.OpenDialogOptions = {
            canSelectMany: false,
            openLabel: 'Select Dataset',
            defaultUri: workspaceFolder?.uri,
            filters: {
                'Data Files': ['npz', 'pt', 'pth', 'h5', 'hdf5'],
                'All Files': ['*']
            }
        };

        const fileUri = await vscode.window.showOpenDialog(options);
        console.log('File picker result:', fileUri);
        if (fileUri && fileUri[0]) {
            // Convert to relative path if within workspace
            const relativePath = workspaceFolder
                ? vscode.workspace.asRelativePath(fileUri[0])
                : fileUri[0].fsPath;

            console.log('Sending dataset path to webview:', relativePath);
            this._sendMessage({
                type: 'datasetPathSelected',
                path: relativePath
            } as any);
        } else {
            console.log('No file selected');
        }
    }

    private async _runModel(
        workspaceUri: vscode.Uri,
        architecture: ModelArchitecture,
        config: TrainingConfig
    ) {
        if (this.trainingManager.isTraining()) {
            vscode.window.showWarningMessage('Training is already in progress');
            return;
        }

        // Create run
        const runId = await this.projectManager.createRun(
            workspaceUri,
            this.currentModelId,
            `Run ${new Date().toLocaleString()}`
        );

        const runPath = this.projectManager.getRunPath(workspaceUri, runId);

        this._sendMessage({ type: 'trainingStarted', runId });

        await this.trainingManager.startTraining(
            workspaceUri,
            runPath,
            architecture,
            config,
            (metrics) => {
                this._sendMessage({ type: 'trainingMetrics', data: metrics });
            },
            async () => {
                this._sendMessage({ type: 'trainingCompleted' });
                await this.projectManager.updateRunStatus(workspaceUri, runId, 'completed');
                vscode.window.showInformationMessage('Training completed successfully');
            },
            async (error) => {
                this._sendMessage({ type: 'trainingError', error });
                await this.projectManager.updateRunStatus(workspaceUri, runId, 'failed');
                vscode.window.showErrorMessage(`Training failed: ${error}`);
            }
        );
    }

    private async _saveModel(workspaceUri: vscode.Uri, architecture: ModelArchitecture, modelName?: string) {
        // Prompt for model name if not provided
        let name = modelName;
        if (!name) {
            name = await vscode.window.showInputBox({
                prompt: 'Enter a name for your model',
                placeHolder: 'my-model',
                value: 'my-model',
                validateInput: (value) => {
                    if (!value || value.trim().length === 0) {
                        return 'Model name cannot be empty';
                    }
                    // Check for invalid filename characters
                    if (/[<>:"/\\|?*]/.test(value)) {
                        return 'Model name contains invalid characters';
                    }
                    return null;
                }
            });
        }

        if (!name) {
            // User cancelled
            return;
        }

        // Use the name as the model ID (sanitized)
        const modelId = name.toLowerCase().replace(/\s+/g, '-');

        try {
            await this.projectManager.saveModel(
                workspaceUri,
                modelId,
                name,
                architecture
            );
            vscode.window.showInformationMessage(`Model "${name}" saved successfully`);

            // Update current model ID so subsequent saves overwrite
            this.currentModelId = modelId;
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to save model: ${error}`);
        }
    }

    private async _exportModel(format: 'pytorch' | 'onnx', workspaceUri: vscode.Uri, architecture?: ModelArchitecture, selectedRun?: string) {
        if (!architecture) {
            vscode.window.showErrorMessage('No architecture data provided for export');
            return;
        }

        try {
            if (format === 'pytorch') {
                await this._exportPyTorch(workspaceUri, architecture);
            } else if (format === 'onnx') {
                await this._exportONNX(workspaceUri, selectedRun);
            }
        } catch (error) {
            console.error('Export error:', error);
            vscode.window.showErrorMessage(`Failed to export model: ${error}`);
        }
    }

    private async _exportPyTorch(workspaceUri: vscode.Uri, architecture: ModelArchitecture) {
        // Generate Python code
        const exporter = new PythonExporter(architecture.nodes, architecture.edges);
        const pythonCode = exporter.generate();

        // Prompt for filename
        const defaultFilename = `exported_model_${Date.now()}.py`;
        const fileUri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.joinPath(workspaceUri, defaultFilename),
            filters: {
                'Python Files': ['py']
            },
            saveLabel: 'Export Model'
        });

        if (!fileUri) {
            // User cancelled
            return;
        }

        // Write file
        await fs.writeFile(fileUri.fsPath, pythonCode, 'utf-8');

        vscode.window.showInformationMessage(`Model exported to ${path.basename(fileUri.fsPath)}`);
    }

    private async _sendAvailableRuns(workspaceUri: vscode.Uri) {
        const netsmithPath = path.join(workspaceUri.fsPath, '.netsmith', 'runs');

        try {
            await fs.access(netsmithPath);
        } catch {
            this._sendMessage({ type: 'availableRuns', runs: [] });
            return;
        }

        // List available runs
        const runs = await fs.readdir(netsmithPath);
        if (runs.length === 0) {
            this._sendMessage({ type: 'availableRuns', runs: [] });
            return;
        }

        // Filter runs that have weights
        const runsWithWeights: string[] = [];
        for (const run of runs) {
            const runPath = path.join(netsmithPath, run);
            if (await this.onnxExporter.hasTrainedWeights(runPath)) {
                runsWithWeights.push(run);
            }
        }

        this._sendMessage({ type: 'availableRuns', runs: runsWithWeights });
    }

    private async _exportONNX(workspaceUri: vscode.Uri, selectedRun?: string) {
        if (!selectedRun) {
            vscode.window.showErrorMessage('No model selected for export');
            return;
        }

        const netsmithPath = path.join(workspaceUri.fsPath, '.netsmith', 'runs');
        const runPath = path.join(netsmithPath, selectedRun);

        // Prompt for output filename
        const defaultFilename = `${selectedRun}.onnx`;
        const fileUri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.joinPath(workspaceUri, defaultFilename),
            filters: {
                'ONNX Files': ['onnx']
            },
            saveLabel: 'Export to ONNX'
        });

        if (!fileUri) {
            return;
        }

        // Show progress
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Exporting model to ONNX...',
                cancellable: false
            },
            async (progress) => {
                progress.report({ increment: 0 });

                try {
                    await this.onnxExporter.export(runPath, fileUri.fsPath);
                    progress.report({ increment: 100 });
                    vscode.window.showInformationMessage(
                        `Model exported to ${path.basename(fileUri.fsPath)}`
                    );
                } catch (error) {
                    throw error;
                }
            }
        );
    }

    private _sendMessage(message: MessageToWebview) {
        this._panel.webview.postMessage(message);
    }

    private _getHtmlForWebview(webview: vscode.Webview, extensionUri: vscode.Uri) {
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(extensionUri, 'webview', 'dist', 'assets', 'index.js')
        );
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(extensionUri, 'webview', 'dist', 'assets', 'index.css')
        );

        const nonce = getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <link href="${styleUri}" rel="stylesheet">
    <title>NetSmith Model Builder</title>
</head>
<body>
    <div id="root"></div>
    <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
    }

    public dispose() {
        ModelBuilderPanel.currentPanel = undefined;

        this._panel.dispose();

        while (this._disposables.length) {
            const disposable = this._disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }
}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
