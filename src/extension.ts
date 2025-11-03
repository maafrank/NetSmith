import * as vscode from 'vscode';
import { ModelBuilderPanel } from './webview/ModelBuilderPanel';
import { ProjectManager } from './project/ProjectManager';
import { TrainingManager } from './training/TrainingManager';

export function activate(context: vscode.ExtensionContext) {
    console.log('NetSmith extension activated');

    const projectManager = new ProjectManager(context);
    const trainingManager = new TrainingManager(context);

    // Command: New Model
    const newModelCommand = vscode.commands.registerCommand('netsmith.newModel', async () => {
        ModelBuilderPanel.createOrShow(context.extensionUri, projectManager, trainingManager);
    });

    // Command: Open Project
    const openProjectCommand = vscode.commands.registerCommand('netsmith.openProject', async () => {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('Please open a workspace folder first');
            return;
        }

        await projectManager.initializeProject(workspaceFolder.uri);
        vscode.window.showInformationMessage('NetSmith project initialized');
    });

    // Command: Run Model
    const runModelCommand = vscode.commands.registerCommand('netsmith.runModel', async () => {
        const panel = ModelBuilderPanel.currentPanel;
        if (!panel) {
            vscode.window.showErrorMessage('No model is currently open');
            return;
        }

        panel.runModel();
    });

    // Command: Stop Training
    const stopTrainingCommand = vscode.commands.registerCommand('netsmith.stopTraining', async () => {
        trainingManager.stopTraining();
        vscode.window.showInformationMessage('Training stopped');
    });

    context.subscriptions.push(
        newModelCommand,
        openProjectCommand,
        runModelCommand,
        stopTrainingCommand
    );

    // Auto-detect .netsmith folder on activation
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (workspaceFolder) {
        projectManager.detectProject(workspaceFolder.uri);
    }
}

export function deactivate() {
    console.log('NetSmith extension deactivated');
}
