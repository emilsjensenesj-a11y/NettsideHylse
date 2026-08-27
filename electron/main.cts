import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron';
import fs from 'node:fs';
import path from 'node:path';

const rendererDevUrl = process.env.ELECTRON_RENDERER_URL ?? 'http://127.0.0.1:5173';
let mainWindow: BrowserWindow | null = null;

interface DesktopExportFilePayload {
  filename: string;
  mimeType?: string;
  data: ArrayBuffer | Uint8Array | number[];
}

function writeStartupLog(message: string, detail?: unknown): void {
  try {
    const logPath = path.join(app.getPath('userData'), 'desktop-startup.log');
    const serializedDetail = detail instanceof Error ? `${detail.stack ?? detail.message}` : detail ? JSON.stringify(detail) : '';
    fs.appendFileSync(logPath, `[${new Date().toISOString()}] ${message}${serializedDetail ? ` ${serializedDetail}` : ''}\n`);
  } catch {
    // Startup logging must never block the app from opening.
  }
}

function getPreloadPath(): string {
  return path.join(__dirname, 'preload.cjs');
}

function getRendererEntryPath(): string {
  return path.join(__dirname, '..', '..', 'dist', 'index.html');
}

function getWindowIconPath(): string {
  return app.isPackaged ? path.join(process.resourcesPath, 'icon.ico') : path.join(process.cwd(), 'build', 'icon.ico');
}

async function createMainWindow(): Promise<void> {
  writeStartupLog('creating main window', {
    packaged: app.isPackaged,
    dirname: __dirname,
    rendererEntry: getRendererEntryPath(),
    preload: getPreloadPath(),
  });

  mainWindow = new BrowserWindow({
    title: 'NouraSoft',
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: '#00091c',
    icon: getWindowIconPath(),
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: getPreloadPath(),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.once('ready-to-show', () => {
    writeStartupLog('main window ready to show');
    mainWindow?.maximize();
    mainWindow?.show();
  });

  mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedUrl) => {
    writeStartupLog('renderer failed to load', { errorCode, errorDescription, validatedUrl });
  });

  mainWindow.webContents.on('render-process-gone', (_event, details) => {
    writeStartupLog('renderer process gone', details);
  });

  mainWindow.webContents.on('did-finish-load', () => {
    void mainWindow?.webContents
      .executeJavaScript('Boolean(window.nouraDesktop && window.nouraDesktop.saveExportedFiles)', true)
      .then((available) => writeStartupLog('desktop export bridge availability', { available }))
      .catch((error) => writeStartupLog('desktop export bridge check failed', error));
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: 'deny' };
  });

  if (app.isPackaged) {
    await mainWindow.loadFile(getRendererEntryPath());
  } else {
    await mainWindow.loadURL(rendererDevUrl);
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  mainWindow.on('closed', () => {
    writeStartupLog('main window closed');
    mainWindow = null;
  });
}

ipcMain.handle('noura:save-exported-files', async (_event, request: { files?: DesktopExportFilePayload[] }) => {
  writeStartupLog('save exported files requested', { count: request.files?.length ?? 0 });
  const files = Array.isArray(request.files) ? request.files : [];
  if (files.length === 0) {
    throw new Error('No export files were provided.');
  }

  if (files.length === 1) {
    const file = files[0];
    const saveOptions: Electron.SaveDialogOptions = {
      title: 'Export Mesh',
      defaultPath: sanitizeExportFilename(file.filename),
      filters: getSaveFilters(file.filename),
    };
    const { canceled, filePath } = mainWindow
      ? await dialog.showSaveDialog(mainWindow, saveOptions)
      : await dialog.showSaveDialog(saveOptions);
    if (canceled || !filePath) {
      return { canceled: true };
    }

    fs.writeFileSync(filePath, toBuffer(file.data));
    return { canceled: false, filePaths: [filePath] };
  }

  const primaryFile = files.find((file) => path.extname(file.filename).toLowerCase() === '.obj') ?? files[0];
  const originalBaseName = path.basename(primaryFile.filename, path.extname(primaryFile.filename));
  const saveOptions: Electron.SaveDialogOptions = {
    title: 'Export OBJ',
    defaultPath: sanitizeExportFilename(primaryFile.filename),
    filters: getSaveFilters(primaryFile.filename),
  };
  const { canceled, filePath } = mainWindow
    ? await dialog.showSaveDialog(mainWindow, saveOptions)
    : await dialog.showSaveDialog(saveOptions);
  if (canceled || !filePath) {
    return { canceled: true };
  }

  const selectedBaseName = sanitizeExportBasename(path.basename(filePath, path.extname(filePath)));
  const outputDirectory = path.join(path.dirname(filePath), selectedBaseName);
  fs.mkdirSync(outputDirectory, { recursive: true });
  const outputPaths: string[] = [];
  for (const file of files) {
    const outputFilename = getMultiFileExportFilename(file.filename, originalBaseName, selectedBaseName);
    const outputPath = path.join(outputDirectory, outputFilename);
    const outputData = rewriteLinkedObjExportFile(file, originalBaseName, selectedBaseName);
    fs.writeFileSync(outputPath, outputData);
    outputPaths.push(outputPath);
  }

  return { canceled: false, filePaths: outputPaths };
});

function toBuffer(data: DesktopExportFilePayload['data']): Buffer {
  if (data instanceof ArrayBuffer) {
    return Buffer.from(data);
  }
  if (ArrayBuffer.isView(data)) {
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  }
  return Buffer.from(data);
}

function sanitizeExportFilename(filename: string): string {
  const sanitized = path.basename(filename).replace(/[<>:"/\\|?*\u0000-\u001f]+/g, '_').trim();
  return sanitized || 'NouraSoft_export';
}

function sanitizeExportBasename(filename: string): string {
  const sanitized = filename.replace(/[<>:"/\\|?*\u0000-\u001f]+/g, '_').trim();
  return sanitized || 'NouraSoft_export';
}

function getMultiFileExportFilename(filename: string, originalBaseName: string, selectedBaseName: string): string {
  const extension = path.extname(filename).toLowerCase();
  const currentBaseName = path.basename(filename, path.extname(filename));
  if ((extension === '.obj' || extension === '.mtl') && currentBaseName === originalBaseName) {
    return `${selectedBaseName}${extension}`;
  }
  if (extension === '.png' && currentBaseName === `${originalBaseName}_texture`) {
    return `${selectedBaseName}_texture.png`;
  }
  return sanitizeExportFilename(filename);
}

function rewriteLinkedObjExportFile(
  file: DesktopExportFilePayload,
  originalBaseName: string,
  selectedBaseName: string,
): Buffer {
  const extension = path.extname(file.filename).toLowerCase();
  const buffer = toBuffer(file.data);
  if (extension === '.obj') {
    const objText = buffer
      .toString('utf8')
      .replace(new RegExp(`^mtllib ${escapeRegExp(originalBaseName)}\\.mtl$`, 'm'), () => `mtllib ${selectedBaseName}.mtl`)
      .replace(new RegExp(`^o ${escapeRegExp(originalBaseName)}$`, 'm'), () => `o ${selectedBaseName}`);
    return Buffer.from(objText, 'utf8');
  }
  if (extension === '.mtl') {
    const mtlText = buffer
      .toString('utf8')
      .replace(
        new RegExp(`^map_Kd ${escapeRegExp(originalBaseName)}_texture\\.png$`, 'm'),
        () => `map_Kd ${selectedBaseName}_texture.png`,
      );
    return Buffer.from(mtlText, 'utf8');
  }
  return buffer;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getSaveFilters(filename: string): Electron.FileFilter[] {
  const extension = path.extname(filename).toLowerCase();
  if (extension === '.stl') {
    return [{ name: 'STL Mesh', extensions: ['stl'] }];
  }
  if (extension === '.obj') {
    return [{ name: 'OBJ Mesh', extensions: ['obj'] }];
  }
  return [{ name: 'All Files', extensions: ['*'] }];
}

process.on('uncaughtException', (error) => {
  writeStartupLog('uncaught exception', error);
});

process.on('unhandledRejection', (reason) => {
  writeStartupLog('unhandled rejection', reason instanceof Error ? reason : { reason: String(reason) });
});

app.setAppUserModelId('com.nourasoft.desktop');

app.whenReady().then(() => {
  writeStartupLog('app ready');
  void createMainWindow().catch((error) => {
    writeStartupLog('create main window failed', error);
    app.quit();
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createMainWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
