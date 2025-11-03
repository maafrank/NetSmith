import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs/promises';
import { ProjectConfig, ModelReference, RunReference } from '../types';

export class ProjectManager {
    private static readonly PROJECT_FOLDER = '.netsmith';
    private config: ProjectConfig | null = null;

    constructor(private context: vscode.ExtensionContext) {}

    async initializeProject(workspaceUri: vscode.Uri): Promise<void> {
        const projectPath = path.join(workspaceUri.fsPath, ProjectManager.PROJECT_FOLDER);

        // Create project structure
        await fs.mkdir(projectPath, { recursive: true });
        await fs.mkdir(path.join(projectPath, 'models'), { recursive: true });
        await fs.mkdir(path.join(projectPath, 'runs'), { recursive: true });
        await fs.mkdir(path.join(projectPath, 'blocks'), { recursive: true });

        // Initialize config
        const config: ProjectConfig = {
            version: '0.1.0',
            defaultDevice: 'auto',
            models: [],
            runs: []
        };

        await this.saveConfig(workspaceUri, config);
        this.config = config;
    }

    async detectProject(workspaceUri: vscode.Uri): Promise<boolean> {
        const projectPath = path.join(workspaceUri.fsPath, ProjectManager.PROJECT_FOLDER);

        try {
            await fs.access(projectPath);
            this.config = await this.loadConfig(workspaceUri);
            return true;
        } catch {
            return false;
        }
    }

    async loadConfig(workspaceUri: vscode.Uri): Promise<ProjectConfig> {
        const configPath = path.join(
            workspaceUri.fsPath,
            ProjectManager.PROJECT_FOLDER,
            'config.json'
        );

        const content = await fs.readFile(configPath, 'utf-8');
        return JSON.parse(content);
    }

    async saveConfig(workspaceUri: vscode.Uri, config: ProjectConfig): Promise<void> {
        const configPath = path.join(
            workspaceUri.fsPath,
            ProjectManager.PROJECT_FOLDER,
            'config.json'
        );

        await fs.writeFile(configPath, JSON.stringify(config, null, 2));
        this.config = config;
    }

    async saveModel(workspaceUri: vscode.Uri, modelId: string, name: string, architecture: any): Promise<void> {
        if (!this.config) {
            await this.initializeProject(workspaceUri);
        }

        const modelPath = path.join(
            workspaceUri.fsPath,
            ProjectManager.PROJECT_FOLDER,
            'models',
            `${modelId}.json`
        );

        await fs.writeFile(modelPath, JSON.stringify(architecture, null, 2));

        // Update config
        const existingModel = this.config!.models.find(m => m.id === modelId);
        if (existingModel) {
            existingModel.updatedAt = new Date().toISOString();
        } else {
            const modelRef: ModelReference = {
                id: modelId,
                name,
                path: `models/${modelId}.json`,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString()
            };
            this.config!.models.push(modelRef);
        }

        await this.saveConfig(workspaceUri, this.config!);
    }

    async loadModel(workspaceUri: vscode.Uri, modelId: string): Promise<any> {
        const modelPath = path.join(
            workspaceUri.fsPath,
            ProjectManager.PROJECT_FOLDER,
            'models',
            `${modelId}.json`
        );

        const content = await fs.readFile(modelPath, 'utf-8');
        return JSON.parse(content);
    }

    async createRun(workspaceUri: vscode.Uri, modelId: string, runName: string): Promise<string> {
        if (!this.config) {
            throw new Error('Project not initialized');
        }

        const runId = `run_${Date.now()}`;
        const runPath = path.join(
            workspaceUri.fsPath,
            ProjectManager.PROJECT_FOLDER,
            'runs',
            runId
        );

        await fs.mkdir(runPath, { recursive: true });

        const runRef: RunReference = {
            id: runId,
            modelId,
            name: runName,
            path: `runs/${runId}`,
            status: 'running',
            createdAt: new Date().toISOString()
        };

        this.config.runs.push(runRef);
        await this.saveConfig(workspaceUri, this.config);

        return runId;
    }

    async updateRunStatus(
        workspaceUri: vscode.Uri,
        runId: string,
        status: 'running' | 'completed' | 'failed' | 'stopped'
    ): Promise<void> {
        if (!this.config) {
            return;
        }

        const run = this.config.runs.find(r => r.id === runId);
        if (run) {
            run.status = status;
            if (status !== 'running') {
                run.completedAt = new Date().toISOString();
            }
            await this.saveConfig(workspaceUri, this.config);
        }
    }

    getProjectPath(workspaceUri: vscode.Uri): string {
        return path.join(workspaceUri.fsPath, ProjectManager.PROJECT_FOLDER);
    }

    getRunPath(workspaceUri: vscode.Uri, runId: string): string {
        return path.join(this.getProjectPath(workspaceUri), 'runs', runId);
    }
}
