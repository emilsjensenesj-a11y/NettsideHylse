import { BufferAttribute, BufferGeometry, Triangle, Vector3 } from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { HitPointInfo, MeshBVH } from 'three-mesh-bvh';

import type { EditableMeshData } from '../sculpt/editable-mesh';

export interface SurfaceRemeshResult {
  geometry: BufferGeometry;
  effectiveEdgeSize: number;
  iterations: number;
  clamped: boolean;
}

export type RemeshBoundaryMode = 'fixed' | 'refined' | 'free';

export interface SurfaceRemeshOptions {
  boundaryMode?: RemeshBoundaryMode;
  iterations?: number;
}

interface MutableMesh {
  positions: number[];
  indices: number[];
}

interface EdgeRecord {
  key: string;
  a: number;
  b: number;
  faces: number[];
  opposites: number[];
  isBoundary: boolean;
  length: number;
}

interface RemeshTopology {
  edges: EdgeRecord[];
  edgeLookup: Set<string>;
  vertexFaces: number[][];
  vertexNeighbors: number[][];
  boundaryNeighbors: number[][];
  boundaryVertices: boolean[];
  faceNormals: Float32Array;
}

interface BoundaryData {
  segments: Float32Array;
}

const MIN_EDGE_SIZE_MM = 0.05;
const DEFAULT_ITERATIONS = 4;
const MAX_ITERATIONS = 8;
const SPLIT_THRESHOLD_SCALE = 4 / 3;
const COLLAPSE_THRESHOLD_SCALE = 4 / 5;
const INTERIOR_SMOOTH_BLEND = 0.24;
const FREE_BOUNDARY_SMOOTH_BLEND = 0.16;
const FREE_BOUNDARY_SMOOTH_BLEND_SOFT = 0.28;
const MIN_NORMAL_DOT = 0.15;
const MIN_TRIANGLE_AREA = 1e-10;

const queryPoint = new Vector3();
const hitPoint = new Vector3();
const faceNormalVector = new Vector3();
const candidateNormal = new Vector3();
const triangle = new Triangle();

export function surfaceRemeshMesh(
  editableMesh: EditableMeshData,
  targetEdgeSize: number,
  options: SurfaceRemeshOptions = {},
): SurfaceRemeshResult {
  const sourceGeometry = editableMesh.geometry as BufferGeometry & {
    boundsTree?: MeshBVH;
    computeBoundsTree?: (options?: { maxLeafSize?: number; setBoundingBox?: boolean }) => unknown;
  };
  if (!sourceGeometry.boundsTree && sourceGeometry.computeBoundsTree) {
    sourceGeometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });
  }

  const boundsTree = sourceGeometry.boundsTree;
  if (!boundsTree) {
    throw new Error('The active mesh is missing its BVH, so remesh could not start.');
  }

  const minimumEdgeSize = MIN_EDGE_SIZE_MM;
  const effectiveEdgeSize = Math.max(targetEdgeSize, minimumEdgeSize);
  const clamped = effectiveEdgeSize !== targetEdgeSize;
  const iterations = clamp(Math.round(options.iterations ?? DEFAULT_ITERATIONS), 1, MAX_ITERATIONS);
  const boundaryMode = options.boundaryMode ?? 'refined';
  const sourceBoundary = buildBoundaryData(editableMesh.positions, editableMesh.indices);

  const mesh: MutableMesh = {
    positions: Array.from(editableMesh.positions),
    indices: Array.from(editableMesh.indices),
  };

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let topology = buildTopology(mesh);
    splitLongEdges(mesh, topology, editableMesh, boundsTree, sourceBoundary, effectiveEdgeSize, boundaryMode);

    topology = buildTopology(mesh);
    collapseShortEdges(mesh, topology, editableMesh, boundsTree, sourceBoundary, effectiveEdgeSize, boundaryMode);

    topology = buildTopology(mesh);
    flipEdges(mesh, topology, boundaryMode);

    topology = buildTopology(mesh);
    relaxAndProject(mesh, topology, editableMesh, boundsTree, effectiveEdgeSize, boundaryMode);
  }

  let geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(mesh.positions), 3));
  geometry.setIndex(new BufferAttribute(new Uint32Array(mesh.indices), 1));
  geometry = mergeVertices(geometry, Math.max(effectiveEdgeSize * 0.01, 1e-5));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return {
    geometry,
    effectiveEdgeSize,
    iterations,
    clamped,
  };
}

function splitLongEdges(
  mesh: MutableMesh,
  topology: RemeshTopology,
  editableMesh: EditableMeshData,
  boundsTree: MeshBVH,
  boundaryData: BoundaryData,
  targetEdgeSize: number,
  boundaryMode: RemeshBoundaryMode,
): boolean {
  const splitThreshold = targetEdgeSize * SPLIT_THRESHOLD_SCALE;
  const splitVertices = new Map<string, number>();

  for (let edgeIndex = 0; edgeIndex < topology.edges.length; edgeIndex += 1) {
    const edge = topology.edges[edgeIndex];
    if (edge.length <= splitThreshold) {
      continue;
    }

    if (edge.isBoundary && boundaryMode === 'fixed') {
      continue;
    }

    const midpoint = midpointOfEdge(mesh.positions, edge.a, edge.b);
    const projected = edge.isBoundary && boundaryMode === 'refined'
      ? projectPointToBoundary(midpoint, boundaryData)
      : projectPointToSurface(midpoint, boundsTree, editableMesh);
    const nextVertex = mesh.positions.length / 3;
    mesh.positions.push(projected[0], projected[1], projected[2]);
    splitVertices.set(edge.key, nextVertex);
  }

  if (splitVertices.size === 0) {
    return false;
  }

  const nextIndices: number[] = [];
  for (let face = 0; face < mesh.indices.length; face += 3) {
    const a = mesh.indices[face];
    const b = mesh.indices[face + 1];
    const c = mesh.indices[face + 2];
    const mab = splitVertices.get(makeEdgeKey(a, b));
    const mbc = splitVertices.get(makeEdgeKey(b, c));
    const mca = splitVertices.get(makeEdgeKey(c, a));
    const mask = (mab != null ? 1 : 0) | (mbc != null ? 2 : 0) | (mca != null ? 4 : 0);

    switch (mask) {
      case 0:
        pushTriangle(nextIndices, a, b, c);
        break;
      case 1:
        pushTriangle(nextIndices, a, mab as number, c);
        pushTriangle(nextIndices, mab as number, b, c);
        break;
      case 2:
        pushTriangle(nextIndices, a, b, mbc as number);
        pushTriangle(nextIndices, a, mbc as number, c);
        break;
      case 4:
        pushTriangle(nextIndices, a, b, mca as number);
        pushTriangle(nextIndices, mca as number, b, c);
        break;
      case 3:
        pushTriangle(nextIndices, a, mab as number, c);
        pushTriangle(nextIndices, mab as number, mbc as number, c);
        pushTriangle(nextIndices, mab as number, b, mbc as number);
        break;
      case 6:
        pushTriangle(nextIndices, a, b, mca as number);
        pushTriangle(nextIndices, b, mbc as number, mca as number);
        pushTriangle(nextIndices, mbc as number, c, mca as number);
        break;
      case 5:
        pushTriangle(nextIndices, a, mab as number, mca as number);
        pushTriangle(nextIndices, mab as number, b, c);
        pushTriangle(nextIndices, mab as number, c, mca as number);
        break;
      case 7:
        pushTriangle(nextIndices, a, mab as number, mca as number);
        pushTriangle(nextIndices, mab as number, b, mbc as number);
        pushTriangle(nextIndices, mca as number, mbc as number, c);
        pushTriangle(nextIndices, mab as number, mbc as number, mca as number);
        break;
      default:
        break;
    }
  }

  mesh.indices = nextIndices;
  return true;
}

function collapseShortEdges(
  mesh: MutableMesh,
  topology: RemeshTopology,
  editableMesh: EditableMeshData,
  boundsTree: MeshBVH,
  boundaryData: BoundaryData,
  targetEdgeSize: number,
  boundaryMode: RemeshBoundaryMode,
): boolean {
  const collapseThreshold = targetEdgeSize * COLLAPSE_THRESHOLD_SCALE;
  const sortedEdges = topology.edges
    .filter((edge) => edge.length < collapseThreshold)
    .sort((left, right) => left.length - right.length);

  if (sortedEdges.length === 0) {
    return false;
  }

  const vertexCount = mesh.positions.length / 3;
  const remap = new Int32Array(vertexCount);
  const locked = new Uint8Array(vertexCount);
  const nextPositions = mesh.positions.slice();
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    remap[vertex] = vertex;
  }

  let changed = false;
  for (let edgeIndex = 0; edgeIndex < sortedEdges.length; edgeIndex += 1) {
    const edge = sortedEdges[edgeIndex];
    let a = resolveVertex(remap, edge.a);
    let b = resolveVertex(remap, edge.b);
    if (a === b || locked[a] || locked[b]) {
      continue;
    }

    const aBoundary = topology.boundaryVertices[a];
    const bBoundary = topology.boundaryVertices[b];
    if (boundaryMode !== 'free' && (edge.isBoundary || aBoundary || bBoundary)) {
      continue;
    }

    let keep = a;
    let drop = b;
    if (bBoundary && !aBoundary) {
      keep = b;
      drop = a;
      a = keep;
      b = drop;
    }

    const midpoint = midpointOfEdge(nextPositions, keep, drop);
    const candidate = edge.isBoundary
      ? projectPointToBoundary(midpoint, boundaryData)
      : projectPointToSurface(midpoint, boundsTree, editableMesh);

    if (!isCollapseSafe(mesh, topology, keep, drop, candidate)) {
      continue;
    }

    setVertex(nextPositions, keep, candidate);
    remap[drop] = keep;
    locked[keep] = 1;
    locked[drop] = 1;
    changed = true;
  }

  if (!changed) {
    return false;
  }

  const collapsedIndices: number[] = [];
  for (let face = 0; face < mesh.indices.length; face += 3) {
    const a = resolveVertex(remap, mesh.indices[face]);
    const b = resolveVertex(remap, mesh.indices[face + 1]);
    const c = resolveVertex(remap, mesh.indices[face + 2]);
    pushTriangle(collapsedIndices, a, b, c);
  }

  const compacted = compactMesh(nextPositions, collapsedIndices);
  mesh.positions = compacted.positions;
  mesh.indices = compacted.indices;
  return true;
}

function flipEdges(
  mesh: MutableMesh,
  topology: RemeshTopology,
  boundaryMode: RemeshBoundaryMode,
): boolean {
  const nextIndices = mesh.indices.slice();
  const lockedFaces = new Uint8Array(mesh.indices.length / 3);
  let changed = false;

  for (let edgeIndex = 0; edgeIndex < topology.edges.length; edgeIndex += 1) {
    const edge = topology.edges[edgeIndex];
    if (edge.isBoundary || edge.faces.length !== 2) {
      continue;
    }

    const c = edge.opposites[0];
    const d = edge.opposites[1];
    if (c === d || topology.edgeLookup.has(makeEdgeKey(c, d))) {
      continue;
    }

    if (boundaryMode !== 'free') {
      if (
        topology.boundaryVertices[edge.a] ||
        topology.boundaryVertices[edge.b] ||
        topology.boundaryVertices[c] ||
        topology.boundaryVertices[d]
      ) {
        continue;
      }
    }

    const face0 = edge.faces[0];
    const face1 = edge.faces[1];
    if (lockedFaces[face0] || lockedFaces[face1]) {
      continue;
    }

    const currentError =
      valenceError(topology, edge.a) +
      valenceError(topology, edge.b) +
      valenceError(topology, c) +
      valenceError(topology, d);
    const flippedError =
      idealValenceError(topology, edge.a, -1) +
      idealValenceError(topology, edge.b, -1) +
      idealValenceError(topology, c, 1) +
      idealValenceError(topology, d, 1);
    if (flippedError >= currentError - 0.1) {
      continue;
    }

    const reference = averageFaceNormal(topology.faceNormals, face0, face1);
    const tri0 = orientTriangle(mesh.positions, c, d, edge.b, reference);
    const tri1 = orientTriangle(mesh.positions, d, c, edge.a, reference);
    if (!triangleIsUsable(mesh.positions, tri0[0], tri0[1], tri0[2], reference)) {
      continue;
    }
    if (!triangleIsUsable(mesh.positions, tri1[0], tri1[1], tri1[2], reference)) {
      continue;
    }

    writeTriangle(nextIndices, face0, tri0[0], tri0[1], tri0[2]);
    writeTriangle(nextIndices, face1, tri1[0], tri1[1], tri1[2]);
    lockedFaces[face0] = 1;
    lockedFaces[face1] = 1;
    changed = true;
  }

  if (changed) {
    mesh.indices = nextIndices;
  }

  return changed;
}

function relaxAndProject(
  mesh: MutableMesh,
  topology: RemeshTopology,
  editableMesh: EditableMeshData,
  boundsTree: MeshBVH,
  targetEdgeSize: number,
  boundaryMode: RemeshBoundaryMode,
): void {
  const nextPositions = mesh.positions.slice();
  const freeBoundaryBlend =
    boundaryMode === 'free' ? FREE_BOUNDARY_SMOOTH_BLEND_SOFT : FREE_BOUNDARY_SMOOTH_BLEND;

  for (let vertex = 0; vertex < topology.vertexNeighbors.length; vertex += 1) {
    const neighbors =
      topology.boundaryVertices[vertex] && boundaryMode === 'free' && topology.boundaryNeighbors[vertex].length >= 2
        ? topology.boundaryNeighbors[vertex]
        : topology.vertexNeighbors[vertex];

    if (neighbors.length === 0) {
      continue;
    }

    if (topology.boundaryVertices[vertex] && boundaryMode !== 'free') {
      continue;
    }

    const current = getVertex(mesh.positions, vertex);
    const average = averageNeighborPosition(mesh.positions, neighbors);
    const projectedCurrent = projectWithNormal(current, boundsTree, editableMesh);
    const normal = projectedCurrent.normal;
    const deltaX = average[0] - current[0];
    const deltaY = average[1] - current[1];
    const deltaZ = average[2] - current[2];
    const normalDot = deltaX * normal[0] + deltaY * normal[1] + deltaZ * normal[2];
    const tangentX = deltaX - normal[0] * normalDot;
    const tangentY = deltaY - normal[1] * normalDot;
    const tangentZ = deltaZ - normal[2] * normalDot;
    const blend = topology.boundaryVertices[vertex] ? freeBoundaryBlend : INTERIOR_SMOOTH_BLEND;
    const candidate: [number, number, number] = [
      current[0] + tangentX * blend,
      current[1] + tangentY * blend,
      current[2] + tangentZ * blend,
    ];

    const projected = projectPointToSurface(candidate, boundsTree, editableMesh);
    setVertex(nextPositions, vertex, projected);
  }

  mesh.positions = nextPositions;
}

function buildTopology(mesh: MutableMesh): RemeshTopology {
  const vertexCount = mesh.positions.length / 3;
  const faceCount = mesh.indices.length / 3;
  const neighborSets = Array.from({ length: vertexCount }, () => new Set<number>());
  const boundaryNeighborSets = Array.from({ length: vertexCount }, () => new Set<number>());
  const vertexFaces = Array.from({ length: vertexCount }, () => [] as number[]);
  const edgeMap = new Map<string, EdgeRecord>();
  const faceNormals = new Float32Array(faceCount * 3);

  for (let face = 0; face < mesh.indices.length; face += 3) {
    const faceIndex = face / 3;
    const a = mesh.indices[face];
    const b = mesh.indices[face + 1];
    const c = mesh.indices[face + 2];
    vertexFaces[a].push(faceIndex);
    vertexFaces[b].push(faceIndex);
    vertexFaces[c].push(faceIndex);
    neighborSets[a].add(b);
    neighborSets[a].add(c);
    neighborSets[b].add(a);
    neighborSets[b].add(c);
    neighborSets[c].add(a);
    neighborSets[c].add(b);

    const normal = computeFaceNormal(mesh.positions, a, b, c);
    const normalOffset = faceIndex * 3;
    faceNormals[normalOffset] = normal[0];
    faceNormals[normalOffset + 1] = normal[1];
    faceNormals[normalOffset + 2] = normal[2];

    trackEdge(edgeMap, mesh.positions, a, b, c, faceIndex);
    trackEdge(edgeMap, mesh.positions, b, c, a, faceIndex);
    trackEdge(edgeMap, mesh.positions, c, a, b, faceIndex);
  }

  const edges = Array.from(edgeMap.values());
  const edgeLookup = new Set<string>();
  const boundaryVertices = new Array<boolean>(vertexCount).fill(false);
  for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex += 1) {
    const edge = edges[edgeIndex];
    edgeLookup.add(edge.key);
    if (!edge.isBoundary) {
      continue;
    }

    boundaryVertices[edge.a] = true;
    boundaryVertices[edge.b] = true;
    boundaryNeighborSets[edge.a].add(edge.b);
    boundaryNeighborSets[edge.b].add(edge.a);
  }

  return {
    edges,
    edgeLookup,
    vertexFaces,
    vertexNeighbors: neighborSets.map((entry) => Array.from(entry)),
    boundaryNeighbors: boundaryNeighborSets.map((entry) => Array.from(entry)),
    boundaryVertices,
    faceNormals,
  };
}

function buildBoundaryData(positions: ArrayLike<number>, indices: ArrayLike<number>): BoundaryData {
  const edgeCounts = new Map<string, { a: number; b: number; count: number }>();
  for (let face = 0; face < indices.length; face += 3) {
    countBoundaryEdge(edgeCounts, indices[face], indices[face + 1]);
    countBoundaryEdge(edgeCounts, indices[face + 1], indices[face + 2]);
    countBoundaryEdge(edgeCounts, indices[face + 2], indices[face]);
  }

  const segments: number[] = [];
  for (const edge of edgeCounts.values()) {
    if (edge.count !== 1) {
      continue;
    }

    const aOffset = edge.a * 3;
    const bOffset = edge.b * 3;
    segments.push(
      positions[aOffset],
      positions[aOffset + 1],
      positions[aOffset + 2],
      positions[bOffset],
      positions[bOffset + 1],
      positions[bOffset + 2],
    );
  }

  return {
    segments: Float32Array.from(segments),
  };
}

function trackEdge(
  edgeMap: Map<string, EdgeRecord>,
  positions: number[],
  a: number,
  b: number,
  opposite: number,
  faceIndex: number,
): void {
  const key = makeEdgeKey(a, b);
  const existing = edgeMap.get(key);
  if (existing) {
    existing.faces.push(faceIndex);
    existing.opposites.push(opposite);
    existing.isBoundary = false;
    return;
  }

  edgeMap.set(key, {
    key,
    a: Math.min(a, b),
    b: Math.max(a, b),
    faces: [faceIndex],
    opposites: [opposite],
    isBoundary: true,
    length: distanceBetweenVertices(positions, a, b),
  });
}

function countBoundaryEdge(
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

  edgeCounts.set(key, {
    a,
    b,
    count: 1,
  });
}

function isCollapseSafe(
  mesh: MutableMesh,
  topology: RemeshTopology,
  keep: number,
  drop: number,
  candidate: [number, number, number],
): boolean {
  const incidentFaces = new Set<number>([
    ...topology.vertexFaces[keep],
    ...topology.vertexFaces[drop],
  ]);

  for (const faceIndex of incidentFaces) {
    const offset = faceIndex * 3;
    let a = mesh.indices[offset];
    let b = mesh.indices[offset + 1];
    let c = mesh.indices[offset + 2];
    if (a === drop) a = keep;
    if (b === drop) b = keep;
    if (c === drop) c = keep;
    if (a === b || b === c || c === a) {
      continue;
    }

    const referenceOffset = faceIndex * 3;
    const reference: [number, number, number] = [
      topology.faceNormals[referenceOffset],
      topology.faceNormals[referenceOffset + 1],
      topology.faceNormals[referenceOffset + 2],
    ];
    const normal = computeFaceNormal(mesh.positions, a, b, c, keep, candidate);
    const area = Math.hypot(normal[0], normal[1], normal[2]);
    if (area <= MIN_TRIANGLE_AREA) {
      return false;
    }

    const referenceLength = Math.hypot(reference[0], reference[1], reference[2]);
    if (referenceLength > 1e-10) {
      const dotValue =
        (normal[0] * reference[0] + normal[1] * reference[1] + normal[2] * reference[2]) /
        (area * referenceLength);
      if (dotValue < MIN_NORMAL_DOT) {
        return false;
      }
    }
  }

  return true;
}

function compactMesh(
  positions: number[],
  indices: number[],
): MutableMesh {
  const vertexMap = new Int32Array(Math.ceil(positions.length / 3));
  vertexMap.fill(-1);
  const nextPositions: number[] = [];
  const nextIndices: number[] = [];

  for (let index = 0; index < indices.length; index += 1) {
    const sourceVertex = indices[index];
    let targetVertex = vertexMap[sourceVertex];
    if (targetVertex === -1) {
      targetVertex = nextPositions.length / 3;
      vertexMap[sourceVertex] = targetVertex;
      const offset = sourceVertex * 3;
      nextPositions.push(positions[offset], positions[offset + 1], positions[offset + 2]);
    }

    nextIndices.push(targetVertex);
  }

  return {
    positions: nextPositions,
    indices: nextIndices,
  };
}

function midpointOfEdge(positions: number[], a: number, b: number): [number, number, number] {
  const aOffset = a * 3;
  const bOffset = b * 3;
  return [
    (positions[aOffset] + positions[bOffset]) * 0.5,
    (positions[aOffset + 1] + positions[bOffset + 1]) * 0.5,
    (positions[aOffset + 2] + positions[bOffset + 2]) * 0.5,
  ];
}

function averageNeighborPosition(positions: number[], neighbors: number[]): [number, number, number] {
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  for (let neighborIndex = 0; neighborIndex < neighbors.length; neighborIndex += 1) {
    const offset = neighbors[neighborIndex] * 3;
    sumX += positions[offset];
    sumY += positions[offset + 1];
    sumZ += positions[offset + 2];
  }

  const invCount = 1 / neighbors.length;
  return [sumX * invCount, sumY * invCount, sumZ * invCount];
}

function projectPointToSurface(
  point: [number, number, number],
  boundsTree: MeshBVH,
  editableMesh: EditableMeshData,
): [number, number, number] {
  const projection = projectWithNormal(point, boundsTree, editableMesh);
  return projection.point;
}

function projectWithNormal(
  point: [number, number, number],
  boundsTree: MeshBVH,
  editableMesh: EditableMeshData,
): { point: [number, number, number]; normal: [number, number, number] } {
  queryPoint.set(point[0], point[1], point[2]);
  const target: HitPointInfo = {
    point: hitPoint,
    distance: 0,
    faceIndex: -1,
  };
  const hit = boundsTree.closestPointToPoint(queryPoint, target);
  if (!hit || hit.faceIndex < 0) {
    return {
      point,
      normal: [0, 0, 1],
    };
  }

  const faceOffset = hit.faceIndex * 3;
  faceNormalVector.set(
    editableMesh.faceNormals[faceOffset],
    editableMesh.faceNormals[faceOffset + 1],
    editableMesh.faceNormals[faceOffset + 2],
  );
  if (faceNormalVector.lengthSq() <= 1e-12) {
    const ia = editableMesh.indices[faceOffset] * 3;
    const ib = editableMesh.indices[faceOffset + 1] * 3;
    const ic = editableMesh.indices[faceOffset + 2] * 3;
    triangle.a.set(
      editableMesh.positions[ia],
      editableMesh.positions[ia + 1],
      editableMesh.positions[ia + 2],
    );
    triangle.b.set(
      editableMesh.positions[ib],
      editableMesh.positions[ib + 1],
      editableMesh.positions[ib + 2],
    );
    triangle.c.set(
      editableMesh.positions[ic],
      editableMesh.positions[ic + 1],
      editableMesh.positions[ic + 2],
    );
    triangle.getNormal(faceNormalVector);
  } else {
    faceNormalVector.normalize();
  }

  return {
    point: [hit.point.x, hit.point.y, hit.point.z],
    normal: [faceNormalVector.x, faceNormalVector.y, faceNormalVector.z],
  };
}

function projectPointToBoundary(
  point: [number, number, number],
  boundaryData: BoundaryData,
): [number, number, number] {
  if (boundaryData.segments.length === 0) {
    return point;
  }

  let bestDistanceSq = Number.POSITIVE_INFINITY;
  let bestX = point[0];
  let bestY = point[1];
  let bestZ = point[2];

  for (let segment = 0; segment < boundaryData.segments.length; segment += 6) {
    const ax = boundaryData.segments[segment];
    const ay = boundaryData.segments[segment + 1];
    const az = boundaryData.segments[segment + 2];
    const bx = boundaryData.segments[segment + 3];
    const by = boundaryData.segments[segment + 4];
    const bz = boundaryData.segments[segment + 5];
    const abx = bx - ax;
    const aby = by - ay;
    const abz = bz - az;
    const denom = abx * abx + aby * aby + abz * abz;
    let t = 0;
    if (denom > 1e-12) {
      t = ((point[0] - ax) * abx + (point[1] - ay) * aby + (point[2] - az) * abz) / denom;
      t = clamp(t, 0, 1);
    }

    const px = ax + abx * t;
    const py = ay + aby * t;
    const pz = az + abz * t;
    const dx = point[0] - px;
    const dy = point[1] - py;
    const dz = point[2] - pz;
    const distanceSq = dx * dx + dy * dy + dz * dz;
    if (distanceSq >= bestDistanceSq) {
      continue;
    }

    bestDistanceSq = distanceSq;
    bestX = px;
    bestY = py;
    bestZ = pz;
  }

  return [bestX, bestY, bestZ];
}

function orientTriangle(
  positions: number[],
  a: number,
  b: number,
  c: number,
  reference: [number, number, number],
): [number, number, number] {
  const normal = computeFaceNormal(positions, a, b, c);
  const dotValue = normal[0] * reference[0] + normal[1] * reference[1] + normal[2] * reference[2];
  return dotValue < 0 ? [a, c, b] : [a, b, c];
}

function triangleIsUsable(
  positions: number[],
  a: number,
  b: number,
  c: number,
  reference: [number, number, number],
): boolean {
  if (a === b || b === c || c === a) {
    return false;
  }

  const normal = computeFaceNormal(positions, a, b, c);
  const area = Math.hypot(normal[0], normal[1], normal[2]);
  if (area <= MIN_TRIANGLE_AREA) {
    return false;
  }

  const referenceLength = Math.hypot(reference[0], reference[1], reference[2]);
  if (referenceLength <= 1e-10) {
    return true;
  }

  const dotValue =
    (normal[0] * reference[0] + normal[1] * reference[1] + normal[2] * reference[2]) /
    (area * referenceLength);
  return dotValue > MIN_NORMAL_DOT;
}

function averageFaceNormal(
  faceNormals: Float32Array,
  face0: number,
  face1: number,
): [number, number, number] {
  const offset0 = face0 * 3;
  const offset1 = face1 * 3;
  const x = faceNormals[offset0] + faceNormals[offset1];
  const y = faceNormals[offset0 + 1] + faceNormals[offset1 + 1];
  const z = faceNormals[offset0 + 2] + faceNormals[offset1 + 2];
  const length = Math.hypot(x, y, z);
  if (length <= 1e-12) {
    return [0, 0, 1];
  }

  return [x / length, y / length, z / length];
}

function computeFaceNormal(
  positions: number[],
  a: number,
  b: number,
  c: number,
  overrideVertex = -1,
  overridePosition?: [number, number, number],
): [number, number, number] {
  const aPos = readVertex(positions, a, overrideVertex, overridePosition);
  const bPos = readVertex(positions, b, overrideVertex, overridePosition);
  const cPos = readVertex(positions, c, overrideVertex, overridePosition);
  const abx = bPos[0] - aPos[0];
  const aby = bPos[1] - aPos[1];
  const abz = bPos[2] - aPos[2];
  const acx = cPos[0] - aPos[0];
  const acy = cPos[1] - aPos[1];
  const acz = cPos[2] - aPos[2];
  return [
    aby * acz - abz * acy,
    abz * acx - abx * acz,
    abx * acy - aby * acx,
  ];
}

function readVertex(
  positions: number[],
  vertex: number,
  overrideVertex = -1,
  overridePosition?: [number, number, number],
): [number, number, number] {
  if (vertex === overrideVertex && overridePosition) {
    return overridePosition;
  }

  return getVertex(positions, vertex);
}

function valenceError(topology: RemeshTopology, vertex: number): number {
  return idealValenceError(topology, vertex, 0);
}

function idealValenceError(topology: RemeshTopology, vertex: number, delta: number): number {
  const currentValence = topology.vertexNeighbors[vertex].length + delta;
  const ideal = topology.boundaryVertices[vertex] ? 4 : 6;
  const error = currentValence - ideal;
  return error * error;
}

function resolveVertex(remap: Int32Array, vertex: number): number {
  let current = vertex;
  while (remap[current] !== current) {
    current = remap[current];
  }

  let path = vertex;
  while (remap[path] !== path) {
    const next = remap[path];
    remap[path] = current;
    path = next;
  }

  return current;
}

function distanceBetweenVertices(positions: number[], a: number, b: number): number {
  const aOffset = a * 3;
  const bOffset = b * 3;
  const dx = positions[bOffset] - positions[aOffset];
  const dy = positions[bOffset + 1] - positions[aOffset + 1];
  const dz = positions[bOffset + 2] - positions[aOffset + 2];
  return Math.hypot(dx, dy, dz);
}

function getVertex(positions: number[], vertex: number): [number, number, number] {
  const offset = vertex * 3;
  return [positions[offset], positions[offset + 1], positions[offset + 2]];
}

function setVertex(positions: number[], vertex: number, value: [number, number, number]): void {
  const offset = vertex * 3;
  positions[offset] = value[0];
  positions[offset + 1] = value[1];
  positions[offset + 2] = value[2];
}

function pushTriangle(target: number[], a: number, b: number, c: number): void {
  if (a === b || b === c || c === a) {
    return;
  }

  target.push(a, b, c);
}

function writeTriangle(target: number[], faceIndex: number, a: number, b: number, c: number): void {
  const offset = faceIndex * 3;
  target[offset] = a;
  target[offset + 1] = b;
  target[offset + 2] = c;
}

function makeEdgeKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
