// VSCode API for webview communication
declare global {
  interface Window {
    acquireVsCodeApi: () => VSCodeAPI;
  }
}

interface VSCodeAPI {
  postMessage(message: any): void;
  getState(): any;
  setState(state: any): void;
}

class VSCodeAPIWrapper {
  private readonly api: VSCodeAPI | undefined;

  constructor() {
    if (typeof window.acquireVsCodeApi === 'function') {
      this.api = window.acquireVsCodeApi();
    }
  }

  public postMessage(message: any): void {
    if (this.api) {
      this.api.postMessage(message);
    } else {
      console.log('VSCode API not available, message:', message);
    }
  }

  public getState(): any {
    if (this.api) {
      return this.api.getState();
    }
    return undefined;
  }

  public setState(state: any): void {
    if (this.api) {
      this.api.setState(state);
    }
  }
}

export const vscode = new VSCodeAPIWrapper();
