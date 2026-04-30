# Hole Fill Module

`src/sculpt/hole-fill.ts` implements a local tangent-style hole fill for indexed triangle meshes without depending on Three.js scene objects as the source of truth.

## Core Flow

1. Validate the ordered boundary loop and classify the hole.
2. Gather a small support band around the seam.
3. Fit a best-fit plane and a local quadratic support surface.
4. Generate a dense interior sample set in the projected 2D hole using a target edge length.
5. Build a constrained Delaunay patch from the boundary edges plus sampled interior points.
6. Lift interior patch vertices onto the fitted surface.
7. Fair only the patch interior while keeping the boundary fixed.
8. Reject unstable or low-quality patches.
9. Insert the patch, then recompute only local normals and dirty-region ids.

## Main API

```ts
import { createMesh, detectBoundaryLoops, fillHole } from './src/sculpt/hole-fill';

const mesh = createMesh(positionArray, indexArray);
const boundaryLoops = detectBoundaryLoops(mesh);
const loop = boundaryLoops[0];

const result = fillHole(mesh, loop, {
  supportRingDepth: 2,
  fairingIterations: 20,
  projectionBlend: 0.2,
});

if (result.success) {
  console.log(result.patch?.newFaceIds, result.dirtyRegion?.updatedVertexIds);
} else {
  console.warn(result.reason, result.message);
}
```

## Important Exports

- `createMesh(...)`
- `detectBoundaryLoops(mesh)`
- `validateBoundaryLoop(mesh, loop, options?)`
- `computeHoleStatistics(mesh, loop, options?)`
- `collectSupportBand(mesh, loop, ringDepth)`
- `computeBestFitPlane(points, preferredNormal?)`
- `computeBoundaryFrames(mesh, loop, centroid, averageNormal)`
- `triangulateHole2D(loop2d)`
- `fitQuadraticSurface(points, plane, targetEdgeLength)`
- `refinePatch(patch, targetEdgeLength, options?)`
- `fairPatch(patch, plane, surfaceFit, boundaryFrames, targetEdgeLength, averageNormal, options?)`
- `recomputeLocalNormals(mesh, dirtyFaceIds, dirtyVertexIds)`
- `collectDirtyRegion(mesh, boundaryLoop, newFaceIds, newVertexIds)`
- `fillHole(mesh, boundaryLoop, options?)`

## Limitations

- Tuned for small to medium smooth holes.
- Rejects sharp-feature, branching, or strongly non-manifold loops.
- Uses dense interior sampling plus constrained Delaunay in 2D, then local fairing in 3D.
- Uses local quadratic fitting and iterative fairing instead of a global optimization solve.
- Integration in the current app rebuilds the editable render session after a successful fill, even though the fill module itself returns local dirty-region data.

## Synthetic Cases

The module also exports simple synthetic generators for quick checks:

- `createPlanarHoleCase()`
- `createSphereHoleCase()`
- `createCylinderHoleCase()`
- `createUndulatingHoleCase()`
- `createInvalidNonManifoldCase()`
