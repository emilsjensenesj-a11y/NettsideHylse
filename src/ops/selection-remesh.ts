import '../three-bvh';

import { BufferAttribute, BufferGeometry } from 'three';

import { surfaceRemeshMesh } from './surface-remesh';
import { createEditableMeshData } from '../sculpt/editable-mesh';

export interface SelectionRemeshResult {
  positions: Float32Array;
  indices: Uint32Array;
  referencePositions: Float32Array;
  selectedTriangleMask: Uint8Array;
  triangleSourceIds: Int32Array;
  effectiveEdgeSize: number;
  iterations: number;
  clamped: boolean;
}

interface ExtractedPatch {
  positions: Float32Array;
  indices: Uint32Array;
  referencePositions: Float32Array;
  localToGlobal: Uint32Array;
}

interface BoundaryCandidate {
  globalVertex: number;
  x: number;
  y: number;
  z: number;
  used: boolean;
}

export function remeshSelectedTriangles(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  referencePositions: ArrayLike<number>,
  selectedTriangleMask: Uint8Array,
  targetEdgeSize: number,
): SelectionRemeshResult | null {
  if (selectedTriangleMask.length === 0) {
    return null;
  }

  let selectedCount = 0;
  for (let triangle = 0; triangle < selectedTriangleMask.length; triangle += 1) {
    selectedCount += selectedTriangleMask[triangle] !== 0 ? 1 : 0;
  }
  if (selectedCount === 0) {
    return null;
  }

  const patch = extractSelectedPatch(positions, indices, referencePositions, selectedTriangleMask);
  if (!patch) {
    return null;
  }

  const patchGeometry = new BufferGeometry();
  patchGeometry.setAttribute('position', new BufferAttribute(patch.positions.slice(), 3));
  patchGeometry.setIndex(new BufferAttribute(patch.indices.slice(), 1));
  patchGeometry.computeBoundingBox();
  patchGeometry.computeBoundingSphere();
  (
    patchGeometry as BufferGeometry & {
      computeBoundsTree?: (options?: { maxLeafSize?: number; setBoundingBox?: boolean; indirect?: boolean }) => void;
    }
  ).computeBoundsTree?.({
    maxLeafSize: 20,
    setBoundingBox: false,
    indirect: true,
  });

  const editablePatch = createEditableMeshData(patchGeometry, {
    referencePositions: patch.referencePositions,
  });
  const remeshedPatch = surfaceRemeshMesh(editablePatch, targetEdgeSize, {
    boundaryMode: 'fixed',
  });

  const remeshedPositions = getPositionArray(remeshedPatch.geometry);
  const remeshedIndices = getIndexArray(remeshedPatch.geometry);
  const originalBoundaryCandidates = buildBoundaryCandidates(
    patch.positions,
    patch.indices,
    patch.localToGlobal,
    remeshedPatch.effectiveEdgeSize,
  );
  const remeshedBoundaryVertices = collectBoundaryVertexIds(remeshedIndices);
  const remeshBoundaryToGlobal = mapBoundaryVerticesToOriginal(
    remeshedPositions,
    remeshedBoundaryVertices,
    originalBoundaryCandidates,
    remeshedPatch.effectiveEdgeSize,
  );
  if (!remeshBoundaryToGlobal) {
    return null;
  }

  const tempPositions = Array.from(positions);
  const tempReferencePositions = Array.from(referencePositions);
  const remeshLocalToTempGlobal = new Int32Array(remeshedPositions.length / 3);
  remeshLocalToTempGlobal.fill(-1);

  for (let vertex = 0; vertex < remeshLocalToTempGlobal.length; vertex += 1) {
    const mappedBoundaryVertex = remeshBoundaryToGlobal.get(vertex);
    if (mappedBoundaryVertex != null) {
      remeshLocalToTempGlobal[vertex] = mappedBoundaryVertex;
      continue;
    }

    const nextVertex = tempPositions.length / 3;
    const offset = vertex * 3;
    tempPositions.push(
      remeshedPositions[offset],
      remeshedPositions[offset + 1],
      remeshedPositions[offset + 2],
    );
    tempReferencePositions.push(
      remeshedPositions[offset],
      remeshedPositions[offset + 1],
      remeshedPositions[offset + 2],
    );
    remeshLocalToTempGlobal[vertex] = nextVertex;
  }

  const finalIndices: number[] = [];
  const finalSelectionMask: number[] = [];

  for (let triangle = 0; triangle < selectedTriangleMask.length; triangle += 1) {
    if (selectedTriangleMask[triangle] !== 0) {
      continue;
    }

    const triOffset = triangle * 3;
    finalIndices.push(indices[triOffset], indices[triOffset + 1], indices[triOffset + 2]);
    finalSelectionMask.push(0);
  }

  for (let triangle = 0; triangle < remeshedIndices.length; triangle += 3) {
    finalIndices.push(
      remeshLocalToTempGlobal[remeshedIndices[triangle]],
      remeshLocalToTempGlobal[remeshedIndices[triangle + 1]],
      remeshLocalToTempGlobal[remeshedIndices[triangle + 2]],
    );
    finalSelectionMask.push(1);
  }

  const compacted = compactMesh(tempPositions, tempReferencePositions, finalIndices);
  return {
    positions: compacted.positions,
    indices: compacted.indices,
    referencePositions: compacted.referencePositions,
    selectedTriangleMask: Uint8Array.from(finalSelectionMask),
    triangleSourceIds: createIdentitySources(finalSelectionMask.length),
    effectiveEdgeSize: remeshedPatch.effectiveEdgeSize,
    iterations: remeshedPatch.iterations,
    clamped: remeshedPatch.clamped,
  };
}

function extractSelectedPatch(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  referencePositions: ArrayLike<number>,
  selectedTriangleMask: Uint8Array,
): ExtractedPatch | null {
  const vertexCount = Math.ceil(positions.length / 3);
  const globalToLocal = new Int32Array(vertexCount);
  globalToLocal.fill(-1);
  const localToGlobal: number[] = [];
  const patchIndices: number[] = [];

  for (let triangle = 0; triangle < selectedTriangleMask.length; triangle += 1) {
    if (selectedTriangleMask[triangle] === 0) {
      continue;
    }

    const triOffset = triangle * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const globalVertex = indices[triOffset + corner];
      let localVertex = globalToLocal[globalVertex];
      if (localVertex === -1) {
        localVertex = localToGlobal.length;
        globalToLocal[globalVertex] = localVertex;
        localToGlobal.push(globalVertex);
      }

      patchIndices.push(localVertex);
    }
  }

  if (patchIndices.length === 0) {
    return null;
  }

  const patchPositions = new Float32Array(localToGlobal.length * 3);
  const patchReferencePositions = new Float32Array(localToGlobal.length * 3);
  for (let localVertex = 0; localVertex < localToGlobal.length; localVertex += 1) {
    const globalVertex = localToGlobal[localVertex];
    const globalOffset = globalVertex * 3;
    const localOffset = localVertex * 3;
    patchPositions[localOffset] = positions[globalOffset];
    patchPositions[localOffset + 1] = positions[globalOffset + 1];
    patchPositions[localOffset + 2] = positions[globalOffset + 2];
    patchReferencePositions[localOffset] = referencePositions[globalOffset];
    patchReferencePositions[localOffset + 1] = referencePositions[globalOffset + 1];
    patchReferencePositions[localOffset + 2] = referencePositions[globalOffset + 2];
  }

  return {
    positions: patchPositions,
    indices: Uint32Array.from(patchIndices),
    referencePositions: patchReferencePositions,
    localToGlobal: Uint32Array.from(localToGlobal),
  };
}

function buildBoundaryCandidates(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  localToGlobal: Uint32Array,
  edgeSize: number,
): Map<string, BoundaryCandidate[]> {
  const boundaryVertices = collectBoundaryVertexIds(indices);
  const tolerance = Math.max(edgeSize * 0.05, 1e-5);
  const candidates = new Map<string, BoundaryCandidate[]>();

  for (let i = 0; i < boundaryVertices.length; i += 1) {
    const localVertex = boundaryVertices[i];
    const offset = localVertex * 3;
    const candidate: BoundaryCandidate = {
      globalVertex: localToGlobal[localVertex],
      x: positions[offset],
      y: positions[offset + 1],
      z: positions[offset + 2],
      used: false,
    };
    const key = makePositionKey(candidate.x, candidate.y, candidate.z, tolerance);
    const bucket = candidates.get(key);
    if (bucket) {
      bucket.push(candidate);
    } else {
      candidates.set(key, [candidate]);
    }
  }

  return candidates;
}

function mapBoundaryVerticesToOriginal(
  positions: ArrayLike<number>,
  boundaryVertices: Uint32Array,
  candidatesByKey: Map<string, BoundaryCandidate[]>,
  edgeSize: number,
): Map<number, number> | null {
  if (boundaryVertices.length === 0) {
    return new Map();
  }

  const tolerance = Math.max(edgeSize * 0.05, 1e-5);
  const searchRadiusSq = Math.max(edgeSize * 0.2, 2e-4) ** 2;
  const buckets = Array.from(candidatesByKey.values());
  const result = new Map<number, number>();

  for (let i = 0; i < boundaryVertices.length; i += 1) {
    const vertex = boundaryVertices[i];
    const offset = vertex * 3;
    const x = positions[offset];
    const y = positions[offset + 1];
    const z = positions[offset + 2];
    const key = makePositionKey(x, y, z, tolerance);
    const bucket = candidatesByKey.get(key);
    const exact = bucket?.find((candidate) => !candidate.used);
    if (exact) {
      exact.used = true;
      result.set(vertex, exact.globalVertex);
      continue;
    }

    let bestCandidate: BoundaryCandidate | null = null;
    let bestDistanceSq = Infinity;
    for (let bucketIndex = 0; bucketIndex < buckets.length; bucketIndex += 1) {
      const entries = buckets[bucketIndex];
      for (let candidateIndex = 0; candidateIndex < entries.length; candidateIndex += 1) {
        const candidate = entries[candidateIndex];
        if (candidate.used) {
          continue;
        }

        const dx = x - candidate.x;
        const dy = y - candidate.y;
        const dz = z - candidate.z;
        const distanceSq = dx * dx + dy * dy + dz * dz;
        if (distanceSq < bestDistanceSq) {
          bestDistanceSq = distanceSq;
          bestCandidate = candidate;
        }
      }
    }

    if (!bestCandidate || bestDistanceSq > searchRadiusSq) {
      return null;
    }

    bestCandidate.used = true;
    result.set(vertex, bestCandidate.globalVertex);
  }

  return result;
}

function compactMesh(
  positions: number[],
  referencePositions: number[],
  indices: number[],
): { positions: Float32Array; referencePositions: Float32Array; indices: Uint32Array } {
  const vertexCount = positions.length / 3;
  const remap = new Int32Array(vertexCount);
  remap.fill(-1);
  const compactedPositions: number[] = [];
  const compactedReferencePositions: number[] = [];
  const compactedIndices = new Uint32Array(indices.length);
  let nextVertex = 0;

  for (let i = 0; i < indices.length; i += 1) {
    const sourceVertex = indices[i];
    let targetVertex = remap[sourceVertex];
    if (targetVertex === -1) {
      targetVertex = nextVertex;
      remap[sourceVertex] = targetVertex;
      const offset = sourceVertex * 3;
      compactedPositions.push(positions[offset], positions[offset + 1], positions[offset + 2]);
      compactedReferencePositions.push(
        referencePositions[offset],
        referencePositions[offset + 1],
        referencePositions[offset + 2],
      );
      nextVertex += 1;
    }

    compactedIndices[i] = targetVertex;
  }

  return {
    positions: Float32Array.from(compactedPositions),
    referencePositions: Float32Array.from(compactedReferencePositions),
    indices: compactedIndices,
  };
}

function collectBoundaryVertexIds(indices: ArrayLike<number>): Uint32Array {
  const edgeCounts = new Map<string, { a: number; b: number; count: number }>();
  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    trackEdge(edgeCounts, indices[triangle], indices[triangle + 1]);
    trackEdge(edgeCounts, indices[triangle + 1], indices[triangle + 2]);
    trackEdge(edgeCounts, indices[triangle + 2], indices[triangle]);
  }

  const boundaryVertices = new Set<number>();
  for (const edge of edgeCounts.values()) {
    if (edge.count !== 1) {
      continue;
    }

    boundaryVertices.add(edge.a);
    boundaryVertices.add(edge.b);
  }

  return Uint32Array.from(boundaryVertices);
}

function trackEdge(
  edgeCounts: Map<string, { a: number; b: number; count: number }>,
  a: number,
  b: number,
): void {
  const key = makeEdgeKey(a, b);
  const existing = edgeCounts.get(key);
  if (existing) {
    existing.count += 1;
    return;
  }

  edgeCounts.set(key, { a, b, count: 1 });
}

function getPositionArray(geometry: BufferGeometry): Float32Array {
  const attribute = geometry.getAttribute('position');
  if (!(attribute instanceof BufferAttribute) || !(attribute.array instanceof Float32Array)) {
    throw new Error('Remeshed selection is missing a Float32 position attribute.');
  }

  return attribute.array;
}

function getIndexArray(geometry: BufferGeometry): Uint32Array {
  const index = geometry.getIndex();
  if (!(index instanceof BufferAttribute)) {
    const position = geometry.getAttribute('position');
    if (!(position instanceof BufferAttribute)) {
      throw new Error('Remeshed selection is missing both an index buffer and a valid position attribute.');
    }

    const sequential = new Uint32Array(position.count);
    for (let i = 0; i < sequential.length; i += 1) {
      sequential[i] = i;
    }

    return sequential;
  }

  if (index.array instanceof Uint32Array) {
    return index.array;
  }

  const normalized = new Uint32Array(index.array.length);
  normalized.set(index.array as ArrayLike<number>);
  return normalized;
}

function makeEdgeKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function makePositionKey(x: number, y: number, z: number, tolerance: number): string {
  const invTolerance = 1 / tolerance;
  return `${Math.round(x * invTolerance)}:${Math.round(y * invTolerance)}:${Math.round(z * invTolerance)}`;
}

function createIdentitySources(count: number): Int32Array {
  const sources = new Int32Array(count);
  for (let i = 0; i < count; i += 1) {
    sources[i] = i;
  }

  return sources;
}
