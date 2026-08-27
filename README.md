# NouraSoft

Browser and Windows desktop mesh editing focused on responsive local workflows. NouraSoft loads local `STL` and `OBJ` meshes and provides sculpting, selection editing, hole filling, remeshing, thickening, limb workflows, measurements, and `STL`/`OBJ` export.

## Setup

```bash
npm install
npm run dev
```

Open the Vite URL in a browser, then use **Open STL / OBJ** to load a local mesh.

## Windows Desktop App

The Electron shell uses the same Vite/Three.js renderer as the browser app.

```bash
npm run desktop:dev
```

Build an unpacked Windows application or distributable installer and zip with:

```bash
npm run desktop:pack
npm run desktop:dist
```

Generated desktop artifacts are written to `release/`. Keep that directory out of Git and publish installers as GitHub Release assets. See [docs/electron-desktop.md](docs/electron-desktop.md) for details.

## Access From Other Devices

### Same Wi-Fi / local network

The dev server is already configured to listen on your network.

1. Start the app:

```bash
npm run dev
```

2. Find your computer's local IP address:

```powershell
ipconfig
```

3. On another device on the same network, open:

```text
http://YOUR-PC-IP:5173
```

Example: `http://192.168.1.42:5173`

### Free internet deployment with GitHub Pages

This project is now set up for free static hosting on GitHub Pages.

1. Create a new GitHub repository.
2. Put this project in that repository.
3. Push it to the `main` branch.
4. In GitHub, open `Settings -> Pages` and set `Source` to `GitHub Actions`.
5. Push again if needed.

The included workflow at [.github/workflows/deploy-pages.yml](.github/workflows/deploy-pages.yml) will build the app and publish `dist/` automatically.

Your public app URL will look like:

```text
https://YOUR-USERNAME.github.io/YOUR-REPO/
```

Because this app is client-side only, that hosted site stays free and does not need a backend.

## Controls

- Left drag: smooth when **Smooth Brush** mode is enabled
- Right drag: rotate
- Middle drag: pan
- Mouse wheel: zoom
- Switch to **Select** mode for face selection tools
- `Sphere`: paint-select faces on the surface under the cursor
- `Box`: drag a screen-space rectangle to select visible faces
- `Snip / Lasso`: drag a freeform screen-space selection like Blender lasso select
- `Shift`: add to the current selection
- `Ctrl`: subtract from the current selection
- `Delete`: delete the selected faces
- `Fill Hole`: inspect open loops in blue, hover a clean boundary in purple, then left click to patch it
- `Smooth`: local Taubin-style smoothing over the affected region
- `Remesh`: remesh the whole surface or a selected region at a chosen target edge size
- `Thicken`: add shell thickness to the current mesh
- `Positive Limb`: guide a remesh-and-extrude workflow from a selected open boundary
- `Export`: save the edited mesh as `STL` or `OBJ`, with selectable coordinate units
- `Undo / Redo`: restores recent strokes from a short ring buffer
- `Reset View`: frames the loaded mesh again

## Sculpting Data Flow

1. The selected file is parsed with the Three.js STL or OBJ loader and normalized into one indexed `BufferGeometry`.
2. Duplicate vertices are welded, the mesh is centered, and cached adjacency is built for `vertex -> faces`, `vertex -> neighbors`, and triangle-to-triangle traversal.
3. The editable mesh keeps `position` and `normal` typed arrays as the authoritative data used by both the sculpt engine and the Three.js geometry attributes.
4. `three-mesh-bvh` builds one BVH after load. Brush picking uses `firstHitOnly` raycasts, and edits call `boundsTree.refit()` instead of rebuilding the tree every mouse move.
5. Each brush stamp flood-fills locally from the hit triangle, edits only the touched region, recomputes face normals only for dirty faces, then recomputes vertex normals only for dirty vertices before marking sparse GPU update ranges.
6. Selection mode tracks selected triangle ids in a mask, renders them with a deep-purple overlay mesh, and rebuilds editable topology when operations such as delete or remesh are committed.

## Known Limitations

- The editor operates on one merged mesh at a time; scene hierarchies and animation are not preserved.
- Material handling is designed around the supported `OBJ` workflow rather than general-purpose scene formats.
- Very thin shells or self-intersecting meshes can still allow some opposite-surface influence in edge cases.
- Box and snip selection operate on visible triangle centroids, so selection is practical and Blender-like but not yet as exhaustive as a full GPU picking pass.
- Performance is tuned for responsive local edits, but dense remesh and topology operations remain CPU-intensive and overall responsiveness depends on browser, CPU, and GPU limits.

## Hole Fill Notes

The curvature-aware hole fill lives in [docs/hole-fill.md](docs/hole-fill.md). It is designed for small to medium smooth boundary loops, uses a local quadratic surface fit plus constrained fairing, and fails safely on sharp or ambiguous holes.
