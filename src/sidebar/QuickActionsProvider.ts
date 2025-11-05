import * as vscode from 'vscode';

export class QuickActionsProvider implements vscode.TreeDataProvider<QuickActionItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<QuickActionItem | undefined | null | void> = new vscode.EventEmitter<QuickActionItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<QuickActionItem | undefined | null | void> = this._onDidChangeTreeData.event;

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: QuickActionItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: QuickActionItem): Thenable<QuickActionItem[]> {
        if (element) {
            return Promise.resolve([]);
        }

        return Promise.resolve([
            new QuickActionItem(
                'New Model',
                'Create a new neural network model',
                vscode.TreeItemCollapsibleState.None,
                'netsmith.newModel',
                new vscode.ThemeIcon('add')
            ),
            new QuickActionItem(
                'Open Project',
                'Initialize NetSmith in this workspace',
                vscode.TreeItemCollapsibleState.None,
                'netsmith.openProject',
                new vscode.ThemeIcon('folder-opened')
            )
        ]);
    }
}

class QuickActionItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly description: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly commandId: string,
        public readonly icon: vscode.ThemeIcon
    ) {
        super(label, collapsibleState);
        this.tooltip = description;
        this.iconPath = icon;
        this.command = {
            command: commandId,
            title: label
        };
    }
}
