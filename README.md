# Fast V1 Mesh Sculptor

Browser-based mesh sculpting focused on responsive local editing. The app loads local `STL`, `OBJ`, and `PLY` meshes, then supports three brushes: `Bump / Inflate`, `Smooth`, and `Flatten`.

## Setup

```bash
npm install
npm run dev
```

Open the Vite URL in a browser, then use **Open STL / OBJ / PLY** to load a local mesh.

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

- Left drag: sculpt when **Sculpt Mode** is enabled
- `Alt` + left drag: orbit camera without leaving sculpt mode
- Right drag: pan
- Mouse wheel: zoom
- Switch to **Select** mode for face selection tools
- `Sphere`: paint-select faces on the surface under the cursor
- `Box`: drag a screen-space rectangle to select visible faces
- `Snip / Lasso`: drag a freeform screen-space selection like Blender lasso select
- `Shift`: add to the current selection
- `Ctrl`: subtract from the current selection
- `Delete`: delete the selected faces
- `Fill Hole`: inspect open loops in blue, hover a clean boundary in purple, then left click to patch it
- `Bump / Inflate`: pushes vertices along their local normals
- `Smooth`: local Taubin-style smoothing over the affected region
- `Flatten`: pushes vertices toward a locally estimated plane
- `Undo / Redo`: restores recent strokes from a short ring buffer
- `Reset View`: frames the loaded mesh again

## Sculpting Data Flow

1. The selected file is parsed with the Three.js STL, OBJ, or PLY loader and normalized into one indexed `BufferGeometry`.
2. Duplicate vertices are welded, the mesh is centered, and cached adjacency is built for `vertex -> faces`, `vertex -> neighbors`, and triangle-to-triangle traversal.
3. The editable mesh keeps `position` and `normal` typed arrays as the authoritative data used by both the sculpt engine and the Three.js geometry attributes.
4. `three-mesh-bvh` builds one BVH after load. Brush picking uses `firstHitOnly` raycasts, and edits call `boundsTree.refit()` instead of rebuilding the tree every mouse move.
5. Each brush stamp flood-fills locally from the hit triangle, edits only the touched region, recomputes face normals only for dirty faces, then recomputes vertex normals only for dirty vertices before marking sparse GPU update ranges.
6. Selection mode tracks selected triangle ids in a mask, renders them with a deep-purple overlay mesh, and rebuilds the editable mesh only when faces are explicitly deleted.

## Known Limitations

- v1 edits one merged mesh at a time and ignores original materials.
- There is no remeshing, topology change, painting, animation, or mesh export pipeline.
- Very thin shells or self-intersecting meshes can still allow some opposite-surface influence in edge cases.
- Box and snip selection operate on visible triangle centroids, so selection is practical and Blender-like but not yet as exhaustive as a full GPU picking pass.
- Performance is tuned for responsive local edits, but very dense meshes will still depend on browser and GPU limits. In practice, meshes in the low hundreds of thousands of triangles should feel workable on a modern desktop; beyond that, stroke density and browser overhead become the main constraint.

## Hole Fill Notes

The curvature-aware hole fill lives in [docs/hole-fill.md](docs/hole-fill.md). It is designed for small to medium smooth boundary loops, uses a local quadratic surface fit plus constrained fairing, and fails safely on sharp or ambiguous holes.
