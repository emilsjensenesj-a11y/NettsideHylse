import { contextBridge, ipcRenderer } from 'electron';

interface DesktopExportFilePayload {
  filename: string;
  mimeType: string;
  data: ArrayBuffer;
}

const desktopApi = {
  platform: process.platform,
  electronVersion: process.versions.electron,
  saveExportedFiles: (request: { files: DesktopExportFilePayload[] }) => ipcRenderer.invoke('noura:save-exported-files', request),
};

contextBridge.exposeInMainWorld('nouraDesktop', desktopApi);

declare global {
  interface Window {
    nouraDesktop?: typeof desktopApi;
  }
}
