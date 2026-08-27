# NouraSoft Desktop Build

The desktop app keeps the current Vite/Three.js UI as the renderer and uses Electron only as the Windows shell. This keeps the browser version intact while giving us a path to add native Windows file dialogs later.

## Commands

```bash
npm run desktop:dev
```

Runs Vite and opens the Electron desktop window for development.

```bash
npm run desktop:pack
```

Builds an unpacked Windows app at `release/win-unpacked/NouraSoft.exe`.

```bash
npm run desktop:dist
```

Builds the installer and zip:

- `release/NouraSoft Setup 0.1.1.exe`
- `release/NouraSoft-0.1.1-win.zip`

## Safety Notes

- The renderer keeps browser security defaults.
- `nodeIntegration` is disabled.
- `contextIsolation` is enabled.
- The preload bridge is intentionally minimal.
- Native file access should be added through explicit IPC handlers instead of exposing Node directly to the UI.
