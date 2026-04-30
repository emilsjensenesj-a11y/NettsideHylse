import cdt2d from 'cdt2d';
import earcut from 'earcut';

export interface Mesh {
  positions: number[];
  indices: number[];
  vertexFaces: number[][];
  vertexNeighbors: number[][];
  edgeToFaces: Map<string, number[]>;
  faceNormals: number[];
  vertexNormals: number[];
  dirtyFaces: Set<number>;
  dirtyVertices: Set<number>;
}

export interface FillOptions {
  supportRingDepth: number;
  maxHoleDiameter: number;
  planarityThreshold: number;
  sharpFeatureThreshold: number;
  targetEdgeLengthScale: number;
  refineMaxPasses: number;
  refineEdgeLengthMultiplier: number;
  lightRefineMaxPasses: number;
  lightRefineEdgeLengthMultiplier: number;
  lightFillMaxBoundaryVertices: number;
  lightFillMaxDiameterToEdgeRatio: number;
  fairingIterations: number;
  fairingStep: number;
  projectionBlend: number;
  maxAllowedNormalDeviation: number;
  maxTriangleAspectRatio: number;
  allowPlanarFallback: boolean;
  allowSoftSeamRelaxation: boolean;
  useCotangentFairing: boolean;
  useBiLaplacianFairing: boolean;
  useLocalSeamRelaxation: boolean;
  ignoreSharpFeatureValidation?: boolean;
  debugLog?: ((stage: string, details?: Record<string, unknown>) => void) | null;
}

export type FillFailureReason =
  | 'invalid_loop'
  | 'loop_too_short'
  | 'duplicate_vertices'
  | 'non_boundary_edge'
  | 'non_manifold_boundary'
  | 'non_simple_projection'
  | 'insufficient_support'
  | 'hole_too_large'
  | 'sharp_feature'
  | 'triangulation_failed'
  | 'surface_fit_failed'
  | 'triangle_quality'
  | 'fairing_unstable';

export interface PlaneFit {
  origin: Vec3;
  normal: Vec3;
  uAxis: Vec3;
  vAxis: Vec3;
  eigenValues: Vec3;
  planarity: number;
}

export interface BoundaryFrame {
  vertexId: number;
  position: Vec3;
  normal: Vec3;
  tangent: Vec3;
  inward: Vec3;
}

export interface SupportBand {
  boundaryVertices: number[];
  seamAdjacentVertices: number[];
  supportVertices: number[];
  supportFaces: number[];
}

interface SeamSupportVertex {
  position: Vec3;
  normal: Vec3;
  uv: Vec2;
}

export interface HoleStats {
  perimeter: number;
  estimatedDiameter: number;
  averageBoundaryEdgeLength: number;
  medianBoundaryEdgeLength: number;
  centroid: Vec3;
  averageNormal: Vec3;
  plane: PlaneFit;
  planarityScore: number;
  curvatureVariation: number;
  seamSharpness: number;
  targetEdgeLength: number;
  classification: 'planar' | 'smooth' | 'sharp' | 'too_large' | 'risky';
}

export interface ValidationResult {
  valid: boolean;
  reason: FillFailureReason | null;
  stats: HoleStats | null;
  message: string;
}

export interface DirtyRegion {
  newVertexIds: number[];
  newFaceIds: number[];
  seamAdjacentVertexIds: number[];
  seamAdjacentFaceIds: number[];
  updatedVertexIds: number[];
  updatedFaceIds: number[];
}

export interface FillPatch {
  newVertexIds: number[];
  newFaceIds: number[];
  boundaryVertexIds: number[];
  interiorVertexIds: number[];
  seamAdjacentVertexIds: number[];
  dirtyRegion: DirtyRegion;
}

export interface FillTimings {
  validationMs: number;
  supportMs: number;
  surfaceFitMs: number;
  triangulationMs: number;
  refinementMs: number;
  fairingMs: number;
  insertionMs: number;
  totalMs: number;
}

export interface FillResult {
  success: boolean;
  reason: FillFailureReason | null;
  message: string;
  patch: FillPatch | null;
  dirtyRegion: DirtyRegion | null;
  stats: HoleStats | null;
  timings: FillTimings;
}

interface PatchQualityReport {
  valid: boolean;
  reason: 'zero_area' | 'flipped' | 'aspect_ratio' | null;
  triangleIndex: number;
  areaTwice: number;
  normalDot: number;
  aspectRatio: number;
}

export interface QuadraticSurfaceFit {
  coefficients: [number, number, number, number, number, number];
  rmsError: number;
  sampleCount: number;
}

export interface TriangulationResult {
  success: boolean;
  triangles: number[];
  reason: FillFailureReason | null;
  reversedWinding?: boolean;
  method?: 'earcut' | 'earclip';
}

export interface SyntheticHoleCase {
  name: string;
  mesh: Mesh;
  boundaryLoops: number[][];
}

type Vec2 = [number, number];
type Vec3 = [number, number, number];

export interface PatchVertex {
  uv: Vec2;
  position: Vec3;
  isBoundary: boolean;
  meshVertexId: number | null;
}

export interface PatchTriangle {
  a: number;
  b: number;
  c: number;
}

export interface PatchWork {
  vertices: PatchVertex[];
  triangles: PatchTriangle[];
  boundaryVertexCount: number;
  adjacency: number[][];
}

const DEFAULT_OPTIONS: FillOptions = {
  supportRingDepth: 2,
  maxHoleDiameter: Number.POSITIVE_INFINITY,
  planarityThreshold: 0.03,
  sharpFeatureThreshold: Math.PI * 0.28,
  targetEdgeLengthScale: 1,
  refineMaxPasses: 1,
  refineEdgeLengthMultiplier: 3.4,
  lightRefineMaxPasses: 0,
  lightRefineEdgeLengthMultiplier: 6,
  lightFillMaxBoundaryVertices: 64,
  lightFillMaxDiameterToEdgeRatio: 16,
  fairingIterations: 18,
  fairingStep: 0.34,
  projectionBlend: 0.2,
  maxAllowedNormalDeviation: Math.PI * 0.52,
  maxTriangleAspectRatio: 9,
  allowPlanarFallback: true,
  allowSoftSeamRelaxation: true,
  useCotangentFairing: true,
  useBiLaplacianFairing: true,
  useLocalSeamRelaxation: true,
  ignoreSharpFeatureValidation: false,
  debugLog: null,
};

export function createMesh(
  positionsInput: ArrayLike<number>,
  indicesInput: ArrayLike<number>,
): Mesh {
  const positions = Array.from(positionsInput);
  const indices = Array.from(indicesInput);
  const vertexCount = positions.length / 3;
  const mesh: Mesh = {
    positions,
    indices,
    vertexFaces: Array.from({ length: vertexCount }, () => []),
    vertexNeighbors: Array.from({ length: vertexCount }, () => []),
    edgeToFaces: new Map<string, number[]>(),
    faceNormals: new Array(indices.length).fill(0),
    vertexNormals: new Array(positions.length).fill(0),
    dirtyFaces: new Set<number>(),
    dirtyVertices: new Set<number>(),
  };

  for (let face = 0; face < indices.length / 3; face += 1) {
    const offset = face * 3;
    addFaceConnectivity(mesh, face, indices[offset], indices[offset + 1], indices[offset + 2]);
  }

  recomputeLocalNormals(
    mesh,
    createSequentialIds(indices.length / 3),
    createSequentialIds(vertexCount),
  );
  mesh.dirtyFaces.clear();
  mesh.dirtyVertices.clear();
  return mesh;
}

export function detectBoundaryLoops(mesh: Mesh): number[][] {
  const boundaryNeighbors = new Map<number, number[]>();

  for (const [key, faces] of mesh.edgeToFaces) {
    if (faces.length !== 1) {
      continue;
    }

    const [a, b] = parseEdgeKey(key);
    pushBoundaryNeighbor(boundaryNeighbors, a, b);
    pushBoundaryNeighbor(boundaryNeighbors, b, a);
  }

  const visitedEdges = new Set<string>();
  const loops: number[][] = [];

  for (const [vertex, neighbors] of boundaryNeighbors) {
    if (neighbors.length !== 2) {
      continue;
    }

    for (let neighborIndex = 0; neighborIndex < neighbors.length; neighborIndex += 1) {
      const neighbor = neighbors[neighborIndex];
      const startEdge = makeEdgeKey(vertex, neighbor);
      if (visitedEdges.has(startEdge)) {
        continue;
      }

      const loop = orderBoundaryLoop(boundaryNeighbors, vertex, neighbor, visitedEdges);
      if (loop.length >= 3) {
        loops.push(loop);
      }
    }
  }

  return dedupeLoops(loops);
}

export function validateBoundaryLoop(
  mesh: Mesh,
  boundaryLoop: number[],
  options: Partial<FillOptions> = {},
): ValidationResult {
  const resolvedOptions = { ...DEFAULT_OPTIONS, ...options };
  if (boundaryLoop.length < 3) {
    return {
      valid: false,
      reason: 'loop_too_short',
      stats: null,
      message: 'The boundary loop must contain at least three distinct vertices.',
    };
  }

  for (let i = 0; i < boundaryLoop.length; i += 1) {
    const current = boundaryLoop[i];
    const next = boundaryLoop[(i + 1) % boundaryLoop.length];
    if (current === next) {
      return {
        valid: false,
        reason: 'duplicate_vertices',
        stats: null,
        message: 'The boundary loop contains duplicated consecutive vertices.',
      };
    }

    const faces = mesh.edgeToFaces.get(makeEdgeKey(current, next));
    if (!faces || faces.length !== 1) {
      return {
        valid: false,
        reason: faces && faces.length > 2 ? 'non_manifold_boundary' : 'non_boundary_edge',
        stats: null,
        message: 'The supplied loop does not follow clean boundary edges.',
      };
    }
  }

  const boundaryDegree = new Map<number, number>();
  for (let i = 0; i < boundaryLoop.length; i += 1) {
    const current = boundaryLoop[i];
    const next = boundaryLoop[(i + 1) % boundaryLoop.length];
    boundaryDegree.set(current, (boundaryDegree.get(current) ?? 0) + 1);
    boundaryDegree.set(next, (boundaryDegree.get(next) ?? 0) + 1);
  }

  for (const degree of boundaryDegree.values()) {
    if (degree !== 2) {
      return {
        valid: false,
        reason: 'non_manifold_boundary',
        stats: null,
        message: 'The loop has branching or ambiguous non-manifold boundary connectivity.',
      };
    }
  }

  const stats = computeHoleStatistics(mesh, boundaryLoop, resolvedOptions);
  const projectedLoop = boundaryLoop.map((vertexId) =>
    projectPointToPlane(getVertex(mesh, vertexId), stats.plane),
  );

  if (hasPolygonSelfIntersection(projectedLoop.map((point) => [point[0], point[1]]))) {
    return {
      valid: false,
      reason: 'non_simple_projection',
      stats,
      message: 'The projected hole boundary is self-intersecting, so tangent fill was rejected.',
    };
  }

  if (stats.estimatedDiameter > resolvedOptions.maxHoleDiameter) {
    return {
      valid: false,
      reason: 'hole_too_large',
      stats,
      message: 'The hole is larger than the allowed tangent-fill diameter.',
    };
  }

  if (stats.classification === 'sharp' || stats.classification === 'risky') {
    if (resolvedOptions.ignoreSharpFeatureValidation) {
      return {
        valid: true,
        reason: null,
        stats,
        message: 'Sharp-feature validation was bypassed for this fill attempt.',
      };
    }

    return {
      valid: false,
      reason: 'sharp_feature',
      stats,
      message: 'The loop sits on a sharp or irregular seam and was rejected for safe tangent fill.',
    };
  }

  return {
    valid: true,
    reason: null,
    stats,
    message: 'The boundary loop is a valid tangent-fill candidate.',
  };
}

export function computeHoleStatistics(
  mesh: Mesh,
  boundaryLoop: number[],
  options: Partial<FillOptions> = {},
): HoleStats {
  const resolvedOptions = { ...DEFAULT_OPTIONS, ...options };
  const points = boundaryLoop.map((vertexId) => getVertex(mesh, vertexId));
  const centroid = averagePoints(points);
  const averageNormal = averageVertexNormals(mesh, boundaryLoop);
  const plane = computeBestFitPlane(points, averageNormal);

  let perimeter = 0;
  const edgeLengths: number[] = [];
  for (let i = 0; i < boundaryLoop.length; i += 1) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    const edgeLength = length(subtract(next, current));
    edgeLengths.push(edgeLength);
    perimeter += edgeLength;
  }

  let estimatedDiameter = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      estimatedDiameter = Math.max(estimatedDiameter, distance(points[i], points[j]));
    }
  }

  const supportBand = collectSupportBand(mesh, boundaryLoop, resolvedOptions.supportRingDepth);
  const targetEdgeLength = estimateTargetEdgeLength(mesh, boundaryLoop, supportBand);
  const boundaryNormals = boundaryLoop.map((vertexId) =>
    orientNormalToReference(getVertexNormal(mesh, vertexId), plane.normal),
  );

  let curvatureAccumulator = 0;
  let seamSharpness = 0;
  for (let i = 0; i < boundaryNormals.length; i += 1) {
    const current = boundaryNormals[i];
    const next = boundaryNormals[(i + 1) % boundaryNormals.length];
    const angle = safeAcos(clamp(dot(current, next), -1, 1));
    curvatureAccumulator += angle;
    seamSharpness = Math.max(seamSharpness, angle);
  }

  const curvatureVariation =
    boundaryNormals.length > 0 ? curvatureAccumulator / boundaryNormals.length : 0;
  const medianBoundaryEdgeLength = median(edgeLengths);
  const averageBoundaryEdgeLength = perimeter / Math.max(edgeLengths.length, 1);
  let classification: HoleStats['classification'] = 'smooth';
  if (plane.planarity < resolvedOptions.planarityThreshold && curvatureVariation < 0.14) {
    classification = 'planar';
  } else if (seamSharpness > resolvedOptions.sharpFeatureThreshold) {
    classification = 'sharp';
  } else if (curvatureVariation > resolvedOptions.sharpFeatureThreshold * 0.7) {
    classification = 'risky';
  }

  return {
    perimeter,
    estimatedDiameter,
    averageBoundaryEdgeLength,
    medianBoundaryEdgeLength,
    centroid,
    averageNormal: orientNormalToReference(averageNormal, plane.normal),
    plane,
    planarityScore: plane.planarity,
    curvatureVariation,
    seamSharpness,
    targetEdgeLength,
    classification,
  };
}

export function collectSupportBand(
  mesh: Mesh,
  boundaryLoop: number[],
  ringDepth: number,
): SupportBand {
  const boundarySet = new Set(boundaryLoop);
  const visited = new Set(boundaryLoop);
  let frontier = boundaryLoop.slice();
  const seamAdjacentVertices: number[] = [];
  const supportVertices: number[] = [];

  for (let ring = 1; ring <= ringDepth; ring += 1) {
    const nextFrontier: number[] = [];
    for (let i = 0; i < frontier.length; i += 1) {
      const vertex = frontier[i];
      const neighbors = mesh.vertexNeighbors[vertex] ?? [];
      for (let neighborIndex = 0; neighborIndex < neighbors.length; neighborIndex += 1) {
        const neighbor = neighbors[neighborIndex];
        if (boundarySet.has(neighbor) || visited.has(neighbor)) {
          continue;
        }

        visited.add(neighbor);
        nextFrontier.push(neighbor);
        supportVertices.push(neighbor);
      }
    }

    if (ring === 1) {
      seamAdjacentVertices.push(...nextFrontier);
    }

    frontier = nextFrontier;
    if (frontier.length === 0) {
      break;
    }
  }

  const faceSet = new Set<number>();
  for (let i = 0; i < boundaryLoop.length; i += 1) {
    const vertexFaces = mesh.vertexFaces[boundaryLoop[i]] ?? [];
    for (let faceIndex = 0; faceIndex < vertexFaces.length; faceIndex += 1) {
      faceSet.add(vertexFaces[faceIndex]);
    }
  }

  for (let i = 0; i < supportVertices.length; i += 1) {
    const vertexFaces = mesh.vertexFaces[supportVertices[i]] ?? [];
    for (let faceIndex = 0; faceIndex < vertexFaces.length; faceIndex += 1) {
      faceSet.add(vertexFaces[faceIndex]);
    }
  }

  return {
    boundaryVertices: boundaryLoop.slice(),
    seamAdjacentVertices,
    supportVertices,
    supportFaces: Array.from(faceSet),
  };
}

export function computeBestFitPlane(points: Vec3[], preferredNormal?: Vec3): PlaneFit {
  const origin = averagePoints(points);
  let cxx = 0;
  let cxy = 0;
  let cxz = 0;
  let cyy = 0;
  let cyz = 0;
  let czz = 0;

  for (let i = 0; i < points.length; i += 1) {
    const x = points[i][0] - origin[0];
    const y = points[i][1] - origin[1];
    const z = points[i][2] - origin[2];
    cxx += x * x;
    cxy += x * y;
    cxz += x * z;
    cyy += y * y;
    cyz += y * z;
    czz += z * z;
  }

  const jacobi = diagonalizeSymmetric3x3([
    [cxx, cxy, cxz],
    [cxy, cyy, cyz],
    [cxz, cyz, czz],
  ]);
  const sorted = sortEigenPairs(jacobi.values, jacobi.vectors);
  let normal = normalize(sorted.vectors[0]);
  let vAxis = normalize(sorted.vectors[1]);
  let uAxis = normalize(cross(vAxis, normal));
  vAxis = normalize(cross(normal, uAxis));

  if (preferredNormal && dot(normal, preferredNormal) < 0) {
    normal = scale(normal, -1);
    uAxis = scale(uAxis, -1);
  }

  const varianceSum = Math.max(sorted.values[0] + sorted.values[1] + sorted.values[2], 1e-9);
  const planarity = Math.sqrt(Math.max(sorted.values[0], 0) / varianceSum);

  return {
    origin,
    normal,
    uAxis,
    vAxis,
    eigenValues: [sorted.values[2], sorted.values[1], sorted.values[0]],
    planarity,
  };
}

export function computeBoundaryFrames(
  mesh: Mesh,
  boundaryLoop: number[],
  centroid: Vec3,
  averageNormal: Vec3,
): BoundaryFrame[] {
  const frames: BoundaryFrame[] = [];

  for (let i = 0; i < boundaryLoop.length; i += 1) {
    const prevId = boundaryLoop[(i + boundaryLoop.length - 1) % boundaryLoop.length];
    const currentId = boundaryLoop[i];
    const nextId = boundaryLoop[(i + 1) % boundaryLoop.length];

    const prev = getVertex(mesh, prevId);
    const current = getVertex(mesh, currentId);
    const next = getVertex(mesh, nextId);
    const tangent = normalize(subtract(next, prev));
    const normal = orientNormalToReference(getVertexNormal(mesh, currentId), averageNormal);

    // The inward direction comes from the boundary tangent frame so seam continuity follows
    // the surrounding surface flow rather than assuming the hole should close on a flat cap.
    let inward = normalize(cross(normal, tangent));
    if (dot(inward, subtract(centroid, current)) < 0) {
      inward = scale(inward, -1);
    }

    frames.push({
      vertexId: currentId,
      position: current,
      normal,
      tangent,
      inward,
    });
  }

  return frames;
}

export function triangulateHole2D(loop2d: Vec2[]): TriangulationResult {
  if (loop2d.length < 3) {
    return {
      success: false,
      triangles: [],
      reason: 'loop_too_short',
      reversedWinding: false,
      method: 'earclip',
    };
  }

  const signedArea = polygonArea(loop2d);
  if (Math.abs(signedArea) <= 1e-10) {
    return {
      success: false,
      triangles: [],
      reason: 'triangulation_failed',
      reversedWinding: false,
      method: 'earclip',
    };
  }

  const winding = signedArea > 0 ? 1 : -1;
  const flattenedLoop = new Array<number>(loop2d.length * 2);
  for (let i = 0; i < loop2d.length; i += 1) {
    flattenedLoop[i * 2] = loop2d[i][0];
    flattenedLoop[i * 2 + 1] = loop2d[i][1];
  }

  const earcutTriangles = earcut(flattenedLoop, undefined, 2);
  if (earcutTriangles.length >= 3) {
    const orientedTriangles = orientTrianglesToWinding(loop2d, earcutTriangles, winding);
    return {
      success: true,
      triangles: orientedTriangles,
      reason: null,
      reversedWinding: false,
      method: 'earcut',
    };
  }

  const primary = earClipTriangulation(loop2d, winding);
  if (primary) {
    return {
      success: true,
      triangles: primary,
      reason: null,
      reversedWinding: false,
      method: 'earclip',
    };
  }

  const reversed = earClipTriangulation(loop2d, -winding);
  if (reversed) {
    return {
      success: true,
      triangles: reversed,
      reason: null,
      reversedWinding: true,
      method: 'earclip',
    };
  }

  return {
    success: false,
    triangles: [],
    reason: 'triangulation_failed',
    reversedWinding: false,
    method: 'earclip',
  };
}

function earClipTriangulation(loop2d: Vec2[], winding: number): number[] | null {
  const remaining = createSequentialIds(loop2d.length);
  if (winding < 0) {
    remaining.reverse();
  }

  const triangles: number[] = [];
  let guard = 0;

  while (remaining.length > 3 && guard < loop2d.length * loop2d.length) {
    let clippedEar = false;

    for (let i = 0; i < remaining.length; i += 1) {
      const prev = remaining[(i + remaining.length - 1) % remaining.length];
      const current = remaining[i];
      const next = remaining[(i + 1) % remaining.length];

      if (!isConvexCorner(loop2d[prev], loop2d[current], loop2d[next], winding)) {
        continue;
      }

      if (containsOtherPolygonPoint(loop2d, remaining, prev, current, next)) {
        continue;
      }

      triangles.push(prev, current, next);
      remaining.splice(i, 1);
      clippedEar = true;
      break;
    }

    if (!clippedEar) {
      return null;
    }

    guard += 1;
  }

  if (remaining.length === 3) {
    triangles.push(remaining[0], remaining[1], remaining[2]);
  }

  return triangles;
}

function orientTrianglesToWinding(loop2d: Vec2[], triangles: number[], winding: number): number[] {
  const oriented = triangles.slice();
  for (let i = 0; i < oriented.length; i += 3) {
    const a = oriented[i];
    const b = oriented[i + 1];
    const c = oriented[i + 2];
    const area = orient2D(loop2d[a], loop2d[b], loop2d[c]);
    if (winding > 0 ? area < 0 : area > 0) {
      oriented[i + 1] = c;
      oriented[i + 2] = b;
    }
  }

  return oriented;
}

function computeFanFillCenterUv(loop2d: Vec2[]): Vec2 | null {
  const kernelCenter = computePolygonKernelCenter2D(loop2d);
  if (kernelCenter && isValidFanCenter(loop2d, kernelCenter)) {
    return kernelCenter;
  }

  const centroid = computePolygonCentroid2D(loop2d);
  if (isValidFanCenter(loop2d, centroid)) {
    return centroid;
  }

  const average = averagePoints2D(loop2d);
  if (isValidFanCenter(loop2d, average)) {
    return average;
  }

  return null;
}

function computePolygonKernelCenter2D(points: Vec2[]): Vec2 | null {
  if (points.length < 3) {
    return null;
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < points.length; i += 1) {
    minX = Math.min(minX, points[i][0]);
    minY = Math.min(minY, points[i][1]);
    maxX = Math.max(maxX, points[i][0]);
    maxY = Math.max(maxY, points[i][1]);
  }

  const extent = Math.max(maxX - minX, maxY - minY, 1);
  const margin = extent * 2;
  const winding = polygonArea(points) >= 0 ? 1 : -1;
  let kernel: Vec2[] = [
    [minX - margin, minY - margin],
    [maxX + margin, minY - margin],
    [maxX + margin, maxY + margin],
    [minX - margin, maxY + margin],
  ];

  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    kernel = clipPolygonToHalfPlane(kernel, a, b, winding);
    if (kernel.length < 3) {
      return null;
    }
  }

  return computePolygonCentroid2D(kernel);
}

export function refinePatch(
  patch: PatchWork,
  targetEdgeLength: number,
  options: Partial<FillOptions> = {},
): PatchWork {
  const resolvedOptions = { ...DEFAULT_OPTIONS, ...options };
  const maxEdgeLength = Math.max(
    targetEdgeLength * resolvedOptions.refineEdgeLengthMultiplier,
    targetEdgeLength + 1e-6,
  );

  for (let pass = 0; pass < resolvedOptions.refineMaxPasses; pass += 1) {
    let changed = false;
    const refinedTriangles: PatchTriangle[] = [];

    for (let triangleIndex = 0; triangleIndex < patch.triangles.length; triangleIndex += 1) {
      const triangle = patch.triangles[triangleIndex];
      const a = patch.vertices[triangle.a].uv;
      const b = patch.vertices[triangle.b].uv;
      const c = patch.vertices[triangle.c].uv;
      const edgeAB = distance2D(a, b);
      const edgeBC = distance2D(b, c);
      const edgeCA = distance2D(c, a);
      const longestEdge = Math.max(edgeAB, edgeBC, edgeCA);

      if (longestEdge <= maxEdgeLength) {
        refinedTriangles.push(triangle);
        continue;
      }

      changed = true;
      const centroid: Vec2 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3];
      const centroidIndex = patch.vertices.length;
      patch.vertices.push({
        uv: centroid,
        position: [0, 0, 0],
        isBoundary: false,
        meshVertexId: null,
      });

      refinedTriangles.push(
        { a: triangle.a, b: triangle.b, c: centroidIndex },
        { a: triangle.b, b: triangle.c, c: centroidIndex },
        { a: triangle.c, b: triangle.a, c: centroidIndex },
      );
    }

    patch.triangles = refinedTriangles;
    if (!changed) {
      break;
    }
  }

  patch.adjacency = buildPatchAdjacency(patch.vertices.length, patch.triangles);
  return patch;
}

export function fitQuadraticSurface(
  points: Vec3[],
  plane: PlaneFit,
  targetEdgeLength: number,
): QuadraticSurfaceFit {
  if (points.length < 6) {
    return {
      coefficients: [0, 0, 0, 0, 0, 0],
      rmsError: 0,
      sampleCount: points.length,
    };
  }

  const normalMatrix = createMatrix(6, 6);
  const rhs = new Array(6).fill(0);
  let weightSum = 0;
  let errorSum = 0;

  for (let i = 0; i < points.length; i += 1) {
    const local = projectPointToPlane(points[i], plane);
    const u = local[0];
    const v = local[1];
    const w = local[2];
    const basis = [u * u, u * v, v * v, u, v, 1];
    const distanceWeight =
      1 / Math.max(Math.hypot(u, v), Math.max(targetEdgeLength * 0.6, 1e-5));
    const weight = Math.max(distanceWeight, 0.1);

    for (let row = 0; row < 6; row += 1) {
      rhs[row] += basis[row] * w * weight;
      for (let column = 0; column < 6; column += 1) {
        normalMatrix[row][column] += basis[row] * basis[column] * weight;
      }
    }

    weightSum += weight;
  }

  for (let diagonal = 0; diagonal < 6; diagonal += 1) {
    normalMatrix[diagonal][diagonal] += 1e-7;
  }

  const solved = solveLinearSystem(normalMatrix, rhs);
  if (!solved) {
    return {
      coefficients: [0, 0, 0, 0, 0, 0],
      rmsError: Number.POSITIVE_INFINITY,
      sampleCount: points.length,
    };
  }

  for (let i = 0; i < points.length; i += 1) {
    const local = projectPointToPlane(points[i], plane);
    const w = evaluateQuadraticSurface(solved as QuadraticSurfaceFit['coefficients'], local[0], local[1]);
    const delta = w - local[2];
    errorSum += delta * delta;
  }

  return {
    coefficients: solved as QuadraticSurfaceFit['coefficients'],
    rmsError: Math.sqrt(errorSum / Math.max(weightSum, 1)),
    sampleCount: points.length,
  };
}

export function fairPatch(
  patch: PatchWork,
  plane: PlaneFit,
  surfaceFit: QuadraticSurfaceFit,
  boundaryFrames: BoundaryFrame[],
  seamSupport: SeamSupportVertex[],
  targetEdgeLength: number,
  averageNormal: Vec3,
  options: Partial<FillOptions> = {},
): boolean {
  const resolvedOptions = { ...DEFAULT_OPTIONS, ...options };
  const nextPositions: Vec3[] = Array.from({ length: patch.vertices.length }, () => [0, 0, 0]);
  const firstPassTargets: Vec3[] = Array.from({ length: patch.vertices.length }, () => [0, 0, 0]);
  const secondPassTargets: Vec3[] = Array.from({ length: patch.vertices.length }, () => [0, 0, 0]);
  const boundaryFrameByMeshVertex = new Map<number, BoundaryFrame>();
  for (let i = 0; i < boundaryFrames.length; i += 1) {
    boundaryFrameByMeshVertex.set(boundaryFrames[i].vertexId, boundaryFrames[i]);
  }
  const incidentTriangles = buildPatchIncidentTriangles(patch.vertices.length, patch.triangles);
  const boundaryDepth = computePatchBoundaryDepths(patch);
  const currentPositions = Array.from({ length: patch.vertices.length }, (_, vertexIndex) =>
    patch.vertices[vertexIndex].position.slice() as Vec3,
  );

  for (let iteration = 0; iteration < resolvedOptions.fairingIterations; iteration += 1) {
    let maxMoveSq = 0;

    for (let vertexIndex = patch.boundaryVertexCount; vertexIndex < patch.vertices.length; vertexIndex += 1) {
      firstPassTargets[vertexIndex] = computePatchSmoothTarget(
        patch,
        incidentTriangles,
        currentPositions,
        vertexIndex,
        resolvedOptions.useCotangentFairing,
      );
    }

    if (resolvedOptions.useBiLaplacianFairing) {
      const firstPassPositions = createMixedPatchPositionSource(
        patch,
        currentPositions,
        firstPassTargets,
      );
      for (let vertexIndex = patch.boundaryVertexCount; vertexIndex < patch.vertices.length; vertexIndex += 1) {
        secondPassTargets[vertexIndex] = computePatchSmoothTarget(
          patch,
          incidentTriangles,
          firstPassPositions,
          vertexIndex,
          resolvedOptions.useCotangentFairing,
        );
      }
    }

    for (let vertexIndex = patch.boundaryVertexCount; vertexIndex < patch.vertices.length; vertexIndex += 1) {
      const vertex = patch.vertices[vertexIndex];
      const neighbors = patch.adjacency[vertexIndex];
      if (!neighbors || neighbors.length === 0) {
        nextPositions[vertexIndex] = vertex.position.slice() as Vec3;
        continue;
      }

      let boundaryNeighborCount = 0;
      for (let neighborIndex = 0; neighborIndex < neighbors.length; neighborIndex += 1) {
        if (patch.vertices[neighbors[neighborIndex]].isBoundary) {
          boundaryNeighborCount += 1;
        }
      }

      const surfaceTarget = pointOnSurface(plane, surfaceFit, vertex.uv[0], vertex.uv[1]);
      const smoothingTarget = resolvedOptions.useBiLaplacianFairing
        ? secondPassTargets[vertexIndex]
        : firstPassTargets[vertexIndex];

      // The patch remains local and constrained: the boundary stays fixed while the interior
      // repeatedly smooths and then gets nudged back toward the fitted support surface.
      let next = lerpVec3(vertex.position, smoothingTarget, resolvedOptions.fairingStep);

      if (resolvedOptions.useCotangentFairing && boundaryDepth[vertexIndex] <= 2) {
        const seamPlaneTarget = computeSeamPlaneTarget(
          next,
          neighbors,
          patch,
          boundaryFrameByMeshVertex,
        );
        if (seamPlaneTarget) {
          next = lerpVec3(next, seamPlaneTarget, boundaryDepth[vertexIndex] === 1 ? 0.22 : 0.12);
        }
      }

      const projectionBlend =
        resolvedOptions.projectionBlend *
        (resolvedOptions.allowSoftSeamRelaxation && boundaryNeighborCount > 0 ? 1.18 : 1);
      next = lerpVec3(next, surfaceTarget, projectionBlend);

      if (resolvedOptions.useLocalSeamRelaxation && boundaryDepth[vertexIndex] <= 2 && seamSupport.length > 0) {
        const seamRelaxTarget = computeSeamRelaxTarget(
          next,
          vertex.uv,
          seamSupport,
          targetEdgeLength,
        );
        if (seamRelaxTarget) {
          next = lerpVec3(next, seamRelaxTarget, boundaryDepth[vertexIndex] === 1 ? 0.16 : 0.08);
        }
      }

      if (resolvedOptions.allowSoftSeamRelaxation && boundaryNeighborCount > 0) {
        const seamBias = estimateSeamBias(patch, neighbors, boundaryFrameByMeshVertex, targetEdgeLength);
        next = lerpVec3(next, seamBias, 0.1);
      }

      const stabilized = stabilizeVertexMove(
        patch,
        vertexIndex,
        next,
        averageNormal,
        resolvedOptions.maxTriangleAspectRatio,
      );
      nextPositions[vertexIndex] = stabilized;
      maxMoveSq = Math.max(maxMoveSq, distanceSquared(vertex.position, stabilized));
    }

    for (let vertexIndex = patch.boundaryVertexCount; vertexIndex < patch.vertices.length; vertexIndex += 1) {
      patch.vertices[vertexIndex].position = nextPositions[vertexIndex].slice() as Vec3;
      currentPositions[vertexIndex] = nextPositions[vertexIndex].slice() as Vec3;
    }

    if (maxMoveSq < targetEdgeLength * targetEdgeLength * 1e-4) {
      return true;
    }
  }

  return true;
}

export function recomputeLocalNormals(
  mesh: Mesh,
  dirtyFaceIds: number[],
  dirtyVertexIds: number[],
): void {
  for (let i = 0; i < dirtyFaceIds.length; i += 1) {
    const faceId = dirtyFaceIds[i];
    const offset = faceId * 3;
    const a = mesh.indices[offset];
    const b = mesh.indices[offset + 1];
    const c = mesh.indices[offset + 2];
    const normal = computeTriangleNormal(getVertex(mesh, a), getVertex(mesh, b), getVertex(mesh, c));
    mesh.faceNormals[offset] = normal[0];
    mesh.faceNormals[offset + 1] = normal[1];
    mesh.faceNormals[offset + 2] = normal[2];
  }

  for (let i = 0; i < dirtyVertexIds.length; i += 1) {
    const vertexId = dirtyVertexIds[i];
    const faces = mesh.vertexFaces[vertexId] ?? [];
    let nx = 0;
    let ny = 0;
    let nz = 0;
    for (let faceIndex = 0; faceIndex < faces.length; faceIndex += 1) {
      const face = faces[faceIndex] * 3;
      nx += mesh.faceNormals[face];
      ny += mesh.faceNormals[face + 1];
      nz += mesh.faceNormals[face + 2];
    }

    const normal = normalize([nx, ny, nz]);
    const offset = vertexId * 3;
    mesh.vertexNormals[offset] = normal[0];
    mesh.vertexNormals[offset + 1] = normal[1];
    mesh.vertexNormals[offset + 2] = normal[2];
  }
}

export function collectDirtyRegion(
  mesh: Mesh,
  boundaryLoop: number[],
  newFaceIds: number[],
  newVertexIds: number[],
): DirtyRegion {
  const newFaceSet = new Set(newFaceIds);
  const boundarySet = new Set(boundaryLoop);
  const newVertexSet = new Set(newVertexIds);
  const updatedVertexSet = new Set<number>(boundaryLoop);
  const seamAdjacentVertexSet = new Set<number>();
  const seamAdjacentFaceSet = new Set<number>();

  for (let i = 0; i < boundaryLoop.length; i += 1) {
    const vertex = boundaryLoop[i];
    const neighbors = mesh.vertexNeighbors[vertex] ?? [];
    for (let neighborIndex = 0; neighborIndex < neighbors.length; neighborIndex += 1) {
      const neighbor = neighbors[neighborIndex];
      if (newVertexSet.has(neighbor) || boundarySet.has(neighbor)) {
        continue;
      }

      seamAdjacentVertexSet.add(neighbor);
      updatedVertexSet.add(neighbor);
    }

    const boundaryFaces = mesh.vertexFaces[vertex] ?? [];
    for (let faceIndex = 0; faceIndex < boundaryFaces.length; faceIndex += 1) {
      const faceId = boundaryFaces[faceIndex];
      if (!newFaceSet.has(faceId)) {
        seamAdjacentFaceSet.add(faceId);
      }
    }
  }

  for (let i = 0; i < newVertexIds.length; i += 1) {
    updatedVertexSet.add(newVertexIds[i]);
  }

  return {
    newVertexIds,
    newFaceIds,
    seamAdjacentVertexIds: Array.from(seamAdjacentVertexSet),
    seamAdjacentFaceIds: Array.from(seamAdjacentFaceSet),
    updatedVertexIds: Array.from(updatedVertexSet),
    updatedFaceIds: Array.from(new Set([...newFaceIds, ...seamAdjacentFaceSet])),
  };
}

export function fillHole(
  mesh: Mesh,
  boundaryLoop: number[],
  options: Partial<FillOptions> = {},
): FillResult {
  const resolvedOptions = { ...DEFAULT_OPTIONS, ...options };
  const debugLog = resolvedOptions.debugLog;
  const timings: FillTimings = {
    validationMs: 0,
    supportMs: 0,
    surfaceFitMs: 0,
    triangulationMs: 0,
    refinementMs: 0,
    fairingMs: 0,
    insertionMs: 0,
    totalMs: 0,
  };
  const totalStart = now();
  debugLog?.('fill:start', {
    boundaryVertexCount: boundaryLoop.length,
    faceCount: mesh.indices.length / 3,
    vertexCount: mesh.positions.length / 3,
  });

  const validationStart = now();
  const validation = validateBoundaryLoop(mesh, boundaryLoop, resolvedOptions);
  timings.validationMs = now() - validationStart;
  if (!validation.valid || !validation.stats) {
    timings.totalMs = now() - totalStart;
    debugLog?.('fill:validation-failed', {
      reason: validation.reason,
      message: validation.message,
      validationMs: timings.validationMs,
    });
    return {
      success: false,
      reason: validation.reason,
      message: validation.message,
      patch: null,
      dirtyRegion: null,
      stats: validation.stats,
      timings,
    };
  }

  const stats = validation.stats;
  if (
    resolvedOptions.ignoreSharpFeatureValidation &&
    (stats.classification === 'sharp' || stats.classification === 'risky')
  ) {
    debugLog?.('fill:validation-bypassed', {
      classification: stats.classification,
      validationMessage: validation.message,
    });
  }
  debugLog?.('fill:validated', {
    classification: stats.classification,
    perimeter: stats.perimeter,
    diameter: stats.estimatedDiameter,
    planarity: stats.planarityScore,
    curvatureVariation: stats.curvatureVariation,
    seamSharpness: stats.seamSharpness,
    targetEdgeLength: stats.targetEdgeLength,
  });
  const supportStart = now();
  const supportBand = collectSupportBand(mesh, boundaryLoop, resolvedOptions.supportRingDepth);
  timings.supportMs = now() - supportStart;
  if (supportBand.supportVertices.length < 4) {
    timings.totalMs = now() - totalStart;
    debugLog?.('fill:insufficient-support', {
      supportVertexCount: supportBand.supportVertices.length,
      seamAdjacentCount: supportBand.seamAdjacentVertices.length,
      supportMs: timings.supportMs,
    });
    return {
      success: false,
      reason: 'insufficient_support',
      message: 'Not enough local support vertices were found around the hole.',
      patch: null,
      dirtyRegion: null,
      stats,
      timings,
    };
  }
  debugLog?.('fill:support-band', {
    supportVertexCount: supportBand.supportVertices.length,
    seamAdjacentCount: supportBand.seamAdjacentVertices.length,
    supportFaceCount: supportBand.supportFaces.length,
  });

  const boundaryFrames = computeBoundaryFrames(
    mesh,
    boundaryLoop,
    stats.centroid,
    stats.averageNormal,
  );
  const seamSupport = supportBand.seamAdjacentVertices.map((vertexId) => {
    const position = getVertex(mesh, vertexId);
    const projected = projectPointToPlane(position, stats.plane);
    return {
      position,
      normal: orientNormalToReference(getVertexNormal(mesh, vertexId), stats.averageNormal),
      uv: [projected[0], projected[1]] as Vec2,
    };
  });

  const boundaryUv = boundaryLoop
    .map((vertexId) => projectPointToPlane(getVertex(mesh, vertexId), stats.plane))
    .map((projected) => [projected[0], projected[1]] as Vec2);

  const supportPoints = [
    ...boundaryLoop.map((vertexId) => getVertex(mesh, vertexId)),
    ...supportBand.supportVertices.map((vertexId) => getVertex(mesh, vertexId)),
  ];

  const surfaceFitStart = now();
  const surfaceFit = fitQuadraticSurface(
    supportPoints,
    stats.plane,
    stats.targetEdgeLength,
  );
  timings.surfaceFitMs = now() - surfaceFitStart;
  if (!Number.isFinite(surfaceFit.rmsError)) {
    timings.totalMs = now() - totalStart;
    debugLog?.('fill:surface-fit-failed', {
      rmsError: surfaceFit.rmsError,
      sampleCount: surfaceFit.sampleCount,
      surfaceFitMs: timings.surfaceFitMs,
    });
    return {
      success: false,
      reason: 'surface_fit_failed',
      message: 'The local support surface fit was unstable, so the hole was left unchanged.',
      patch: null,
      dirtyRegion: null,
      stats,
      timings,
    };
  }
  debugLog?.('fill:surface-fit', {
    rmsError: surfaceFit.rmsError,
    sampleCount: surfaceFit.sampleCount,
    surfaceFitMs: timings.surfaceFitMs,
  });

  const isSmoothLike = stats.classification === 'planar' || stats.classification === 'smooth';
  const sampledEligible =
    isSmoothLike &&
    boundaryLoop.length <= Math.max(resolvedOptions.lightFillMaxBoundaryVertices, 96) &&
    surfaceFit.rmsError <= Math.max(stats.targetEdgeLength * 0.9, 0.18);
  const useSampledFirst =
    sampledEligible &&
    stats.estimatedDiameter <=
      stats.targetEdgeLength * (resolvedOptions.lightFillMaxDiameterToEdgeRatio * 1.2) &&
    surfaceFit.rmsError <= Math.max(stats.targetEdgeLength * 0.55, 0.1);
  const useConservativeFullFirst =
    !useSampledFirst &&
    (stats.classification === 'sharp' ||
      stats.classification === 'risky' ||
      surfaceFit.rmsError > Math.max(stats.targetEdgeLength * 0.6, 0.12));

  const strategies: Array<{
    name: string;
    builder: 'sampled' | 'triangulated';
    refineOptions: FillOptions;
    fairOptions: FillOptions;
  }> = useSampledFirst
    ? [
        {
          name: 'light-sampled',
          builder: 'sampled',
          refineOptions: {
            ...resolvedOptions,
            refineMaxPasses: resolvedOptions.lightRefineMaxPasses,
            refineEdgeLengthMultiplier: resolvedOptions.lightRefineEdgeLengthMultiplier,
          },
          fairOptions: {
            ...resolvedOptions,
            fairingIterations: Math.min(resolvedOptions.fairingIterations, 12),
            fairingStep: Math.min(resolvedOptions.fairingStep, 0.2),
            projectionBlend: Math.min(Math.max(resolvedOptions.projectionBlend, 0.16), 0.2),
            maxTriangleAspectRatio: Math.max(resolvedOptions.maxTriangleAspectRatio, 16),
          },
        },
        {
          name: 'light',
          builder: 'triangulated',
          refineOptions: {
            ...resolvedOptions,
            refineMaxPasses: resolvedOptions.lightRefineMaxPasses,
            refineEdgeLengthMultiplier: resolvedOptions.lightRefineEdgeLengthMultiplier,
          },
          fairOptions: {
            ...resolvedOptions,
            fairingIterations: Math.min(resolvedOptions.fairingIterations, 10),
            fairingStep: Math.min(resolvedOptions.fairingStep, 0.26),
            projectionBlend: Math.min(resolvedOptions.projectionBlend, 0.14),
            maxTriangleAspectRatio: Math.max(resolvedOptions.maxTriangleAspectRatio, 16),
          },
        },
        {
          name: 'full',
          builder: 'triangulated',
          refineOptions: resolvedOptions,
          fairOptions: resolvedOptions,
        },
      ]
    : isSmoothLike
      ? [
          {
            name: 'sampled-safe',
            builder: 'sampled',
            refineOptions: {
              ...resolvedOptions,
              refineMaxPasses: resolvedOptions.lightRefineMaxPasses,
              refineEdgeLengthMultiplier: resolvedOptions.lightRefineEdgeLengthMultiplier,
            },
          fairOptions: {
            ...resolvedOptions,
            fairingIterations: Math.min(resolvedOptions.fairingIterations, 14),
            fairingStep: Math.min(resolvedOptions.fairingStep, 0.22),
            projectionBlend: Math.min(Math.max(resolvedOptions.projectionBlend, 0.15), 0.2),
            maxTriangleAspectRatio: Math.max(resolvedOptions.maxTriangleAspectRatio, 15),
          },
        },
          {
            name: 'full',
            builder: 'triangulated',
            refineOptions: resolvedOptions,
            fairOptions: resolvedOptions,
          },
        ]
      : [
        ...(useConservativeFullFirst
          ? [
              {
                name: 'conservative',
                builder: 'triangulated' as const,
                refineOptions: {
                  ...resolvedOptions,
                  refineMaxPasses: 0,
                  refineEdgeLengthMultiplier: Math.max(
                    resolvedOptions.refineEdgeLengthMultiplier,
                    4.2,
                  ),
                },
                fairOptions: {
                  ...resolvedOptions,
                  fairingIterations: Math.min(resolvedOptions.fairingIterations, 10),
                  fairingStep: Math.min(resolvedOptions.fairingStep, 0.2),
                  projectionBlend: Math.min(resolvedOptions.projectionBlend, 0.1),
                  allowSoftSeamRelaxation: false,
                },
              },
            ]
          : []),
        {
          name: 'full',
          builder: 'triangulated',
          refineOptions: resolvedOptions,
          fairOptions: resolvedOptions,
        },
      ];

  debugLog?.('fill:strategy', {
    isSmoothLike,
    sampledEligible,
    useSampledFirst,
    hasSampledStrategy: sampledEligible,
    strategyNames: strategies.map((strategy) => strategy.name),
  });

  let patch: PatchWork | null = null;
  let lastQualityReport: PatchQualityReport | null = null;
  let lastFaired = false;
  let lastStrategyName = 'full';
  const triangulationState: { result: TriangulationResult | null } = { result: null };

  const ensureTriangulation = (): TriangulationResult => {
    if (triangulationState.result) {
      return triangulationState.result;
    }

    const triangulationStart = now();
    triangulationState.result = triangulateHole2D(boundaryUv);
    timings.triangulationMs += now() - triangulationStart;
    if (!triangulationState.result.success) {
      debugLog?.('fill:triangulation-failed', {
        reason: triangulationState.result.reason,
        reversedWinding: triangulationState.result.reversedWinding ?? false,
        method: triangulationState.result.method ?? 'earclip',
        triangulationMs: timings.triangulationMs,
      });
      return triangulationState.result;
    }

    debugLog?.('fill:triangulated', {
      patchTriangleCount: triangulationState.result.triangles.length / 3,
      reversedWinding: triangulationState.result.reversedWinding ?? false,
      method: triangulationState.result.method ?? 'earclip',
      triangulationMs: timings.triangulationMs,
    });
    return triangulationState.result;
  };

  for (let strategyIndex = 0; strategyIndex < strategies.length; strategyIndex += 1) {
    const strategy = strategies[strategyIndex];
    lastStrategyName = strategy.name;
    debugLog?.('fill:strategy-start', {
      name: strategy.name,
      builder: strategy.builder,
      refineMaxPasses: strategy.refineOptions.refineMaxPasses,
      refineEdgeLengthMultiplier: strategy.refineOptions.refineEdgeLengthMultiplier,
      fairingIterations: strategy.fairOptions.fairingIterations,
      fairingStep: strategy.fairOptions.fairingStep,
      projectionBlend: strategy.fairOptions.projectionBlend,
      useCotangentFairing: strategy.fairOptions.useCotangentFairing,
      useBiLaplacianFairing: strategy.fairOptions.useBiLaplacianFairing,
      useLocalSeamRelaxation: strategy.fairOptions.useLocalSeamRelaxation,
    });

    let candidatePatch: PatchWork;
    if (strategy.builder === 'sampled') {
      candidatePatch = createSampledPatch(
        mesh,
        boundaryLoop,
        boundaryUv,
        stats.targetEdgeLength,
      );
    } else {
      const triangulationResult = ensureTriangulation();
      if (!triangulationResult.success) {
        continue;
      }

      candidatePatch = createInitialPatch(
        mesh,
        boundaryLoop,
        triangulationResult.triangles,
        boundaryUv,
      );
    }

    const refinementStart = now();
    const refinedPatch = refinePatch(
      candidatePatch,
      stats.targetEdgeLength * strategy.refineOptions.targetEdgeLengthScale,
      strategy.refineOptions,
    );
    timings.refinementMs += now() - refinementStart;
    debugLog?.('fill:refined', {
      strategy: strategy.name,
      patchVertexCount: refinedPatch.vertices.length,
      patchTriangleCount: refinedPatch.triangles.length,
      refinementMs: timings.refinementMs,
    });

    initializeInteriorPatchPositions(refinedPatch, stats.plane, surfaceFit);
    orientPatchTrianglesTowardNormal(refinedPatch, stats.averageNormal);

    const fairingStart = now();
    const faired = fairPatch(
      refinedPatch,
      stats.plane,
      surfaceFit,
      boundaryFrames,
      seamSupport,
      stats.targetEdgeLength,
      stats.averageNormal,
      strategy.fairOptions,
    );
    timings.fairingMs += now() - fairingStart;
    orientPatchTrianglesTowardNormal(refinedPatch, stats.averageNormal);
    const qualityReport = validatePatchQualityDetailed(
      refinedPatch,
      stats.averageNormal,
      strategy.fairOptions.maxTriangleAspectRatio,
    );

    if (faired && qualityReport.valid) {
      patch = refinedPatch;
      lastFaired = true;
      lastQualityReport = qualityReport;
      debugLog?.('fill:faired', {
        strategy: strategy.name,
        fairingMs: timings.fairingMs,
        patchVertexCount: refinedPatch.vertices.length,
        patchTriangleCount: refinedPatch.triangles.length,
      });
      break;
    }

    lastFaired = faired;
    lastQualityReport = qualityReport;
    debugLog?.('fill:strategy-failed', {
      strategy: strategy.name,
      faired,
      qualityReason: qualityReport.reason,
      triangleIndex: qualityReport.triangleIndex,
      areaTwice: qualityReport.areaTwice,
      normalDot: qualityReport.normalDot,
      aspectRatio: qualityReport.aspectRatio,
      patchVertexCount: refinedPatch.vertices.length,
      patchTriangleCount: refinedPatch.triangles.length,
    });
  }

  const triangulationResult = triangulationState.result;
  if (!patch && triangulationResult !== null && !triangulationResult.success) {
    timings.totalMs = now() - totalStart;
    return {
      success: false,
      reason: triangulationResult.reason,
      message: 'The boundary loop could not be triangulated safely.',
      patch: null,
      dirtyRegion: null,
      stats,
      timings,
    };
  }

  if (!patch) {
    timings.totalMs = now() - totalStart;
    debugLog?.('fill:fairing-failed', {
      strategy: lastStrategyName,
      faired: lastFaired,
      fairingMs: timings.fairingMs,
      qualityReason: lastQualityReport?.reason,
      triangleIndex: lastQualityReport?.triangleIndex ?? -1,
      areaTwice: lastQualityReport?.areaTwice ?? 0,
      normalDot: lastQualityReport?.normalDot ?? 0,
      aspectRatio: lastQualityReport?.aspectRatio ?? 0,
    });
    return {
      success: false,
      reason: 'fairing_unstable',
      message: `The ${lastStrategyName} patch failed the ${describePatchQualityReason(lastQualityReport?.reason)} quality check after fairing.`,
      patch: null,
      dirtyRegion: null,
      stats,
      timings,
    };
  }

  const seamDeviation = measureSeamNormalDeviation(mesh, boundaryLoop, patch);
  if (seamDeviation > resolvedOptions.maxAllowedNormalDeviation) {
    timings.totalMs = now() - totalStart;
    debugLog?.('fill:seam-rejected', {
      seamDeviation,
      maxAllowedNormalDeviation: resolvedOptions.maxAllowedNormalDeviation,
    });
    return {
      success: false,
      reason: 'triangle_quality',
      message: 'The proposed fill would create an excessive seam normal break, so it was rejected.',
      patch: null,
      dirtyRegion: null,
      stats,
      timings,
    };
  }

  const insertionStart = now();
  const insertion = insertPatch(mesh, patch, stats.averageNormal);
  const dirtyRegion = collectDirtyRegion(mesh, boundaryLoop, insertion.newFaceIds, insertion.newVertexIds);
  recomputeLocalNormals(mesh, insertion.newFaceIds, dirtyRegion.updatedVertexIds);
  mesh.dirtyFaces = new Set(insertion.newFaceIds);
  mesh.dirtyVertices = new Set(dirtyRegion.updatedVertexIds);
  timings.insertionMs = now() - insertionStart;
  timings.totalMs = now() - totalStart;
  debugLog?.('fill:success', {
    newVertexCount: insertion.newVertexIds.length,
    newFaceCount: insertion.newFaceIds.length,
    updatedVertexCount: dirtyRegion.updatedVertexIds.length,
    updatedFaceCount: dirtyRegion.updatedFaceIds.length,
    timings,
  });

  return {
    success: true,
    reason: null,
    message: `Filled a ${stats.classification} hole with ${insertion.newVertexIds.length} new vertices and ${insertion.newFaceIds.length} new faces.`,
    patch: {
      newVertexIds: insertion.newVertexIds,
      newFaceIds: insertion.newFaceIds,
      boundaryVertexIds: boundaryLoop.slice(),
      interiorVertexIds: insertion.newVertexIds.slice(),
      seamAdjacentVertexIds: dirtyRegion.seamAdjacentVertexIds,
      dirtyRegion,
    },
    dirtyRegion,
    stats,
    timings,
  };
}

export function createPlanarHoleCase(size = 10, holeRadius = 2): SyntheticHoleCase {
  return createGridHoleCase('plane', size, (x, y) => [x, 0, y], holeRadius);
}

export function createSphereHoleCase(size = 12, holeRadius = 2): SyntheticHoleCase {
  return createGridHoleCase(
    'sphere',
    size,
    (x, y) => {
      const radius = size * 0.58;
      const zSq = Math.max(radius * radius - x * x - y * y, 0);
      return [x, Math.sqrt(zSq) - radius * 0.55, y];
    },
    holeRadius,
  );
}

export function createCylinderHoleCase(size = 16, holeRadius = 2): SyntheticHoleCase {
  return createGridHoleCase(
    'cylinder',
    size,
    (x, y) => {
      const radius = size * 0.38;
      const angle = x / Math.max(size * 0.5, 1);
      return [Math.sin(angle) * radius, y * 0.65, Math.cos(angle) * radius];
    },
    holeRadius,
  );
}

export function createUndulatingHoleCase(size = 12, holeRadius = 2): SyntheticHoleCase {
  return createGridHoleCase(
    'undulating',
    size,
    (x, y) => [x, Math.sin(x * 0.45) * 0.55 + Math.cos(y * 0.42) * 0.45, y],
    holeRadius,
  );
}

export function createInvalidNonManifoldCase(): SyntheticHoleCase {
  const base = createPlanarHoleCase(8, 1);
  const mesh = base.mesh;
  const loop = base.boundaryLoops[0];
  const extraVertex = appendVertex(mesh, add(getVertex(mesh, loop[0]), [0, 0.25, 0]));
  appendFace(mesh, loop[0], loop[1], extraVertex);
  recomputeLocalNormals(
    mesh,
    createSequentialIds(mesh.indices.length / 3),
    createSequentialIds(mesh.positions.length / 3),
  );
  return {
    name: 'invalid-non-manifold',
    mesh,
    boundaryLoops: detectBoundaryLoops(mesh),
  };
}

function createInitialPatch(
  mesh: Mesh,
  boundaryLoop: number[],
  triangles: number[],
  boundaryUv: Vec2[],
): PatchWork {
  const vertices: PatchVertex[] = [];
  for (let i = 0; i < boundaryLoop.length; i += 1) {
    vertices.push({
      uv: boundaryUv[i],
      position: getVertex(mesh, boundaryLoop[i]),
      isBoundary: true,
      meshVertexId: boundaryLoop[i],
    });
  }

  const patchTriangles: PatchTriangle[] = [];
  for (let i = 0; i < triangles.length; i += 3) {
    patchTriangles.push({
      a: triangles[i],
      b: triangles[i + 1],
      c: triangles[i + 2],
    });
  }

  return {
    vertices,
    triangles: patchTriangles,
    boundaryVertexCount: boundaryLoop.length,
    adjacency: buildPatchAdjacency(vertices.length, patchTriangles),
  };
}

function createSampledPatch(
  mesh: Mesh,
  boundaryLoop: number[],
  boundaryUv: Vec2[],
  targetEdgeLength: number,
): PatchWork {
  const patch = createInitialPatch(mesh, boundaryLoop, [], boundaryUv);
  const sampleSpacing = Math.max(targetEdgeLength * 0.68, 1e-4);
  const interiorSamples = generateInteriorSamplePoints(boundaryUv, sampleSpacing);
  for (let i = 0; i < interiorSamples.length; i += 1) {
    patch.vertices.push({
      uv: interiorSamples[i],
      position: [0, 0, 0],
      isBoundary: false,
      meshVertexId: null,
    });
  }

  const points = patch.vertices.map((vertex) => vertex.uv);
  const edges: Array<[number, number]> = [];
  for (let i = 0; i < boundaryLoop.length; i += 1) {
    edges.push([i, (i + 1) % boundaryLoop.length]);
  }

  const sampledTriangles = cdt2d(points, edges, {
    delaunay: true,
    interior: true,
    exterior: false,
  });

  if (sampledTriangles.length > 0) {
    patch.triangles = sampledTriangles.map(([a, b, c]) => ({ a, b, c }));
  } else {
    const fallback = triangulateHole2D(boundaryUv);
    patch.triangles = fallback.success
      ? chunkTriangles(fallback.triangles)
      : [];
  }

  patch.adjacency = buildPatchAdjacency(patch.vertices.length, patch.triangles);
  return patch;
}

function generateInteriorSamplePoints(polygon: Vec2[], sampleSpacing: number): Vec2[] {
  const samples: Vec2[] = [];
  const minMax = computeBounds2D(polygon);
  const rowStep = sampleSpacing * Math.sqrt(3) * 0.5;
  const edgeClearance = sampleSpacing * 0.1;
  const vertexClearance = sampleSpacing * 0.14;
  const polygonAreaAbs = Math.abs(polygonArea(polygon));

  let rowIndex = 0;
  for (let y = minMax.minY; y <= minMax.maxY + rowStep * 0.5; y += rowStep, rowIndex += 1) {
    const xOffset = (rowIndex & 1) === 0 ? 0 : sampleSpacing * 0.5;
    for (let x = minMax.minX; x <= minMax.maxX + sampleSpacing * 0.5; x += sampleSpacing) {
      const point: Vec2 = [x + xOffset, y];
      tryAddInteriorSample(
        samples,
        point,
        polygon,
        edgeClearance,
        vertexClearance,
        sampleSpacing * 0.72,
      );
    }
  }

  addSeamBiasedSamples(samples, polygon, sampleSpacing, edgeClearance, vertexClearance);

  const centroid = computePolygonCentroid2D(polygon);
  if (
    samples.length === 0 &&
    polygonAreaAbs > sampleSpacing * sampleSpacing * 0.8 &&
    tryAddInteriorSample(
      samples,
      centroid,
      polygon,
      edgeClearance,
      vertexClearance,
      sampleSpacing * 0.72,
    )
  ) {
    return samples;
  }

  return samples;
}

function addSeamBiasedSamples(
  samples: Vec2[],
  polygon: Vec2[],
  sampleSpacing: number,
  edgeClearance: number,
  vertexClearance: number,
): void {
  const winding = polygonArea(polygon) >= 0 ? 1 : -1;
  const inwardOffset = sampleSpacing * 0.44;

  for (let i = 0; i < polygon.length; i += 1) {
    const start = polygon[i];
    const end = polygon[(i + 1) % polygon.length];
    const edge = [end[0] - start[0], end[1] - start[1]] as Vec2;
    const edgeLength = Math.hypot(edge[0], edge[1]);
    if (edgeLength <= sampleSpacing * 1.1) {
      continue;
    }

    const inward: Vec2 =
      winding > 0
        ? [-edge[1] / edgeLength, edge[0] / edgeLength]
        : [edge[1] / edgeLength, -edge[0] / edgeLength];

    const segments = Math.max(1, Math.round(edgeLength / Math.max(sampleSpacing * 1.15, 1e-6)));
    for (let step = 0; step < segments; step += 1) {
      const t = (step + 0.5) / segments;
      const point: Vec2 = [
        start[0] + edge[0] * t + inward[0] * inwardOffset,
        start[1] + edge[1] * t + inward[1] * inwardOffset,
      ];
      tryAddInteriorSample(
        samples,
        point,
        polygon,
        edgeClearance * 0.6,
        vertexClearance * 0.75,
        sampleSpacing * 0.58,
      );
    }
  }
}

function tryAddInteriorSample(
  samples: Vec2[],
  point: Vec2,
  polygon: Vec2[],
  edgeClearance: number,
  vertexClearance: number,
  sampleClearance: number,
): boolean {
  if (!pointInPolygon2D(point, polygon)) {
    return false;
  }

  if (minDistanceToPolygonEdges2D(point, polygon) < edgeClearance) {
    return false;
  }

  if (minDistanceToPointSet2D(point, polygon) < vertexClearance) {
    return false;
  }

  if (minDistanceToPointSet2D(point, samples) < sampleClearance) {
    return false;
  }

  samples.push(point);
  return true;
}

function minDistanceToPolygonEdges2D(point: Vec2, polygon: Vec2[]): number {
  let minDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < polygon.length; i += 1) {
    minDistance = Math.min(
      minDistance,
      distancePointToSegment2D(point, polygon[i], polygon[(i + 1) % polygon.length]),
    );
  }

  return minDistance;
}

function minDistanceToPointSet2D(point: Vec2, points: Vec2[]): number {
  let minDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < points.length; i += 1) {
    minDistance = Math.min(minDistance, distance2D(point, points[i]));
  }

  return minDistance;
}

function distancePointToSegment2D(point: Vec2, start: Vec2, end: Vec2): number {
  const segmentX = end[0] - start[0];
  const segmentY = end[1] - start[1];
  const segmentLengthSq = segmentX * segmentX + segmentY * segmentY;
  if (segmentLengthSq <= 1e-12) {
    return distance2D(point, start);
  }

  const t = clamp(
    ((point[0] - start[0]) * segmentX + (point[1] - start[1]) * segmentY) / segmentLengthSq,
    0,
    1,
  );
  const projected: Vec2 = [start[0] + segmentX * t, start[1] + segmentY * t];
  return distance2D(point, projected);
}

function computeBounds2D(points: Vec2[]): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < points.length; i += 1) {
    minX = Math.min(minX, points[i][0]);
    minY = Math.min(minY, points[i][1]);
    maxX = Math.max(maxX, points[i][0]);
    maxY = Math.max(maxY, points[i][1]);
  }

  return { minX, minY, maxX, maxY };
}

function chunkTriangles(triangles: number[]): PatchTriangle[] {
  const patchTriangles: PatchTriangle[] = [];
  for (let i = 0; i < triangles.length; i += 3) {
    patchTriangles.push({
      a: triangles[i],
      b: triangles[i + 1],
      c: triangles[i + 2],
    });
  }

  return patchTriangles;
}

function computePerimeter2D(points: Vec2[]): number {
  let perimeter = 0;
  for (let i = 0; i < points.length; i += 1) {
    perimeter += distance2D(points[i], points[(i + 1) % points.length]);
  }

  return perimeter;
}

function initializeInteriorPatchPositions(
  patch: PatchWork,
  plane: PlaneFit,
  surfaceFit: QuadraticSurfaceFit,
): void {
  for (let vertexIndex = patch.boundaryVertexCount; vertexIndex < patch.vertices.length; vertexIndex += 1) {
    const uv = patch.vertices[vertexIndex].uv;
    // Interior vertices are lifted from the local support surface immediately so the fill
    // starts as a curvature-following patch instead of a flat planar lid.
    patch.vertices[vertexIndex].position = pointOnSurface(plane, surfaceFit, uv[0], uv[1]);
  }
}

function orientPatchTrianglesTowardNormal(patch: PatchWork, referenceNormal: Vec3): void {
  for (let triangleIndex = 0; triangleIndex < patch.triangles.length; triangleIndex += 1) {
    const triangle = patch.triangles[triangleIndex];
    const a = patch.vertices[triangle.a].position;
    const b = patch.vertices[triangle.b].position;
    const c = patch.vertices[triangle.c].position;
    const normal = computeTriangleNormal(a, b, c);
    if (dot(normal, referenceNormal) < 0) {
      const temp = triangle.b;
      triangle.b = triangle.c;
      triangle.c = temp;
    }
  }
}

function insertPatch(
  mesh: Mesh,
  patch: PatchWork,
  averageNormal: Vec3,
): { newVertexIds: number[]; newFaceIds: number[] } {
  const newVertexIds: number[] = [];
  const newFaceIds: number[] = [];

  for (let vertexIndex = patch.boundaryVertexCount; vertexIndex < patch.vertices.length; vertexIndex += 1) {
    const meshVertexId = appendVertex(mesh, patch.vertices[vertexIndex].position);
    patch.vertices[vertexIndex].meshVertexId = meshVertexId;
    newVertexIds.push(meshVertexId);
  }

  for (let triangleIndex = 0; triangleIndex < patch.triangles.length; triangleIndex += 1) {
    const triangle = patch.triangles[triangleIndex];
    const a = patch.vertices[triangle.a].meshVertexId!;
    let b = patch.vertices[triangle.b].meshVertexId!;
    let c = patch.vertices[triangle.c].meshVertexId!;
    const normal = computeTriangleNormal(getVertex(mesh, a), getVertex(mesh, b), getVertex(mesh, c));
    if (dot(normal, averageNormal) < 0) {
      const temp = b;
      b = c;
      c = temp;
    }

    const faceId = appendFace(mesh, a, b, c);
    newFaceIds.push(faceId);
  }

  return { newVertexIds, newFaceIds };
}

function validatePatchQuality(
  patch: PatchWork,
  averageNormal: Vec3,
  maxTriangleAspectRatio: number,
): boolean {
  return validatePatchQualityDetailed(patch, averageNormal, maxTriangleAspectRatio).valid;
}

function validatePatchQualityDetailed(
  patch: PatchWork,
  averageNormal: Vec3,
  maxTriangleAspectRatio: number,
): PatchQualityReport {
  for (let triangleIndex = 0; triangleIndex < patch.triangles.length; triangleIndex += 1) {
    const triangle = patch.triangles[triangleIndex];
    const a = patch.vertices[triangle.a].position;
    const b = patch.vertices[triangle.b].position;
    const c = patch.vertices[triangle.c].position;
    const normal = computeTriangleNormal(a, b, c);
    const areaTwice = length(cross(subtract(b, a), subtract(c, a)));
    if (areaTwice <= 1e-8) {
      return {
        valid: false,
        reason: 'zero_area',
        triangleIndex,
        areaTwice,
        normalDot: dot(normal, averageNormal),
        aspectRatio: triangleAspectRatio3D(a, b, c),
      };
    }

    const normalDot = dot(normal, averageNormal);
    if (normalDot <= 1e-5) {
      return {
        valid: false,
        reason: 'flipped',
        triangleIndex,
        areaTwice,
        normalDot,
        aspectRatio: triangleAspectRatio3D(a, b, c),
      };
    }

    const aspectRatio = triangleAspectRatio3D(a, b, c);
    if (!Number.isFinite(aspectRatio) || aspectRatio > maxTriangleAspectRatio) {
      return {
        valid: false,
        reason: 'aspect_ratio',
        triangleIndex,
        areaTwice,
        normalDot,
        aspectRatio,
      };
    }
  }

  return {
    valid: true,
    reason: null,
    triangleIndex: -1,
    areaTwice: 0,
    normalDot: 1,
    aspectRatio: 1,
  };
}

function describePatchQualityReason(reason: PatchQualityReport['reason'] | undefined): string {
  switch (reason) {
    case 'zero_area':
      return 'zero-area';
    case 'flipped':
      return 'flipped-triangle';
    case 'aspect_ratio':
      return 'aspect-ratio';
    default:
      return 'unknown';
  }
}

function measureSeamNormalDeviation(mesh: Mesh, boundaryLoop: number[], patch: PatchWork): number {
  let maxDeviation = 0;
  const patchFaceByEdge = new Map<string, PatchTriangle>();

  for (let triangleIndex = 0; triangleIndex < patch.triangles.length; triangleIndex += 1) {
    const triangle = patch.triangles[triangleIndex];
    const localIds = [triangle.a, triangle.b, triangle.c];
    for (let edge = 0; edge < 3; edge += 1) {
      const from = localIds[edge];
      const to = localIds[(edge + 1) % 3];
      if (!patch.vertices[from].isBoundary || !patch.vertices[to].isBoundary) {
        continue;
      }

      patchFaceByEdge.set(
        makeEdgeKey(patch.vertices[from].meshVertexId!, patch.vertices[to].meshVertexId!),
        triangle,
      );
    }
  }

  for (let i = 0; i < boundaryLoop.length; i += 1) {
    const a = boundaryLoop[i];
    const b = boundaryLoop[(i + 1) % boundaryLoop.length];
    const edgeKey = makeEdgeKey(a, b);
    const existingFaces = mesh.edgeToFaces.get(edgeKey);
    const patchFace = patchFaceByEdge.get(edgeKey);
    if (!existingFaces || existingFaces.length !== 1 || !patchFace) {
      continue;
    }

    const existingNormal = getFaceNormal(mesh, existingFaces[0]);
    const patchNormal = computeTriangleNormal(
      patch.vertices[patchFace.a].position,
      patch.vertices[patchFace.b].position,
      patch.vertices[patchFace.c].position,
    );
    const deviation = safeAcos(clamp(dot(existingNormal, patchNormal), -1, 1));
    maxDeviation = Math.max(maxDeviation, deviation);
  }

  return maxDeviation;
}

function estimateTargetEdgeLength(mesh: Mesh, boundaryLoop: number[], supportBand: SupportBand): number {
  const edgeLengths: number[] = [];

  for (let i = 0; i < boundaryLoop.length; i += 1) {
    const a = getVertex(mesh, boundaryLoop[i]);
    const b = getVertex(mesh, boundaryLoop[(i + 1) % boundaryLoop.length]);
    edgeLengths.push(distance(a, b));
  }

  const supportVertices = [...supportBand.boundaryVertices, ...supportBand.seamAdjacentVertices];
  const seenEdges = new Set<string>();

  for (let i = 0; i < supportVertices.length; i += 1) {
    const vertex = supportVertices[i];
    const neighbors = mesh.vertexNeighbors[vertex] ?? [];
    for (let neighborIndex = 0; neighborIndex < neighbors.length; neighborIndex += 1) {
      const neighbor = neighbors[neighborIndex];
      const edgeKey = makeEdgeKey(vertex, neighbor);
      if (seenEdges.has(edgeKey)) {
        continue;
      }

      seenEdges.add(edgeKey);
      edgeLengths.push(distance(getVertex(mesh, vertex), getVertex(mesh, neighbor)));
    }
  }

  return Math.max(median(edgeLengths), 1e-4);
}

function createGridHoleCase(
  name: string,
  size: number,
  surface: (x: number, y: number) => Vec3,
  holeRadius: number,
): SyntheticHoleCase {
  const positions: number[] = [];
  const indices: number[] = [];
  const dimension = size + 1;
  const offset = size / 2;

  for (let y = 0; y <= size; y += 1) {
    for (let x = 0; x <= size; x += 1) {
      const point = surface(x - offset, y - offset);
      positions.push(point[0], point[1], point[2]);
    }
  }

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const centerX = x + 0.5 - offset;
      const centerY = y + 0.5 - offset;
      if (Math.hypot(centerX, centerY) < holeRadius) {
        continue;
      }

      const a = y * dimension + x;
      const b = a + 1;
      const c = a + dimension;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const mesh = createMesh(positions, indices);
  return {
    name,
    mesh,
    boundaryLoops: detectBoundaryLoops(mesh),
  };
}

function orderBoundaryLoop(
  boundaryNeighbors: Map<number, number[]>,
  startVertex: number,
  firstNeighbor: number,
  visitedEdges: Set<string>,
): number[] {
  const loop = [startVertex];
  let previous = startVertex;
  let current = firstNeighbor;
  visitedEdges.add(makeEdgeKey(startVertex, firstNeighbor));

  for (let guard = 0; guard < boundaryNeighbors.size + 4; guard += 1) {
    loop.push(current);
    const neighbors = boundaryNeighbors.get(current);
    if (!neighbors || neighbors.length !== 2) {
      return [];
    }

    const next = neighbors[0] === previous ? neighbors[1] : neighbors[0];
    if (next === startVertex) {
      return loop;
    }

    const edgeKey = makeEdgeKey(current, next);
    if (visitedEdges.has(edgeKey)) {
      return [];
    }

    visitedEdges.add(edgeKey);
    previous = current;
    current = next;
  }

  return [];
}

function dedupeLoops(loops: number[][]): number[][] {
  const seen = new Set<string>();
  const deduped: number[][] = [];

  for (let i = 0; i < loops.length; i += 1) {
    const normalized = normalizeLoopStart(loops[i]);
    const key = normalized.join(',');
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    deduped.push(normalized);
  }

  return deduped;
}

function normalizeLoopStart(loop: number[]): number[] {
  let minIndex = 0;
  for (let i = 1; i < loop.length; i += 1) {
    if (loop[i] < loop[minIndex]) {
      minIndex = i;
    }
  }

  const rotated = loop.slice(minIndex).concat(loop.slice(0, minIndex));
  const reversed = [rotated[0], ...rotated.slice(1).reverse()];
  return reversed.join(',') < rotated.join(',') ? reversed : rotated;
}

function buildPatchAdjacency(vertexCount: number, triangles: PatchTriangle[]): number[][] {
  const adjacency = Array.from({ length: vertexCount }, () => new Set<number>());
  for (let i = 0; i < triangles.length; i += 1) {
    const triangle = triangles[i];
    adjacency[triangle.a].add(triangle.b);
    adjacency[triangle.a].add(triangle.c);
    adjacency[triangle.b].add(triangle.a);
    adjacency[triangle.b].add(triangle.c);
    adjacency[triangle.c].add(triangle.a);
    adjacency[triangle.c].add(triangle.b);
  }

  return adjacency.map((entry) => Array.from(entry));
}

function buildPatchIncidentTriangles(vertexCount: number, triangles: PatchTriangle[]): number[][] {
  const incident = Array.from({ length: vertexCount }, () => [] as number[]);
  for (let triangleIndex = 0; triangleIndex < triangles.length; triangleIndex += 1) {
    const triangle = triangles[triangleIndex];
    incident[triangle.a].push(triangleIndex);
    incident[triangle.b].push(triangleIndex);
    incident[triangle.c].push(triangleIndex);
  }

  return incident;
}

function computePatchBoundaryDepths(patch: PatchWork): number[] {
  const depths = new Array<number>(patch.vertices.length).fill(Number.POSITIVE_INFINITY);
  const queue: number[] = [];

  for (let vertexIndex = 0; vertexIndex < patch.vertices.length; vertexIndex += 1) {
    if (!patch.vertices[vertexIndex].isBoundary) {
      continue;
    }

    depths[vertexIndex] = 0;
    queue.push(vertexIndex);
  }

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const vertexIndex = queue[cursor];
    const nextDepth = depths[vertexIndex] + 1;
    const neighbors = patch.adjacency[vertexIndex];
    for (let neighborIndex = 0; neighborIndex < neighbors.length; neighborIndex += 1) {
      const neighbor = neighbors[neighborIndex];
      if (nextDepth >= depths[neighbor]) {
        continue;
      }

      depths[neighbor] = nextDepth;
      queue.push(neighbor);
    }
  }

  return depths;
}

function createMixedPatchPositionSource(
  patch: PatchWork,
  currentPositions: Vec3[],
  interiorTargets: Vec3[],
): Vec3[] {
  return Array.from({ length: patch.vertices.length }, (_, vertexIndex) =>
    vertexIndex < patch.boundaryVertexCount
      ? currentPositions[vertexIndex]
      : interiorTargets[vertexIndex],
  );
}

function computePatchSmoothTarget(
  patch: PatchWork,
  incidentTriangles: number[][],
  positions: Vec3[],
  vertexIndex: number,
  useCotangentWeights: boolean,
): Vec3 {
  if (!useCotangentWeights) {
    return computeUniformPatchAverage(patch.adjacency[vertexIndex], positions);
  }

  const weightByNeighbor = new Map<number, number>();
  const triangles = incidentTriangles[vertexIndex];
  for (let i = 0; i < triangles.length; i += 1) {
    const triangle = patch.triangles[triangles[i]];
    accumulateTriangleCotangentWeights(weightByNeighbor, triangle, vertexIndex, positions);
  }

  if (weightByNeighbor.size === 0) {
    return computeUniformPatchAverage(patch.adjacency[vertexIndex], positions);
  }

  let weightSum = 0;
  let x = 0;
  let y = 0;
  let z = 0;
  for (const [neighbor, weight] of weightByNeighbor) {
    if (!Number.isFinite(weight) || weight <= 1e-8) {
      continue;
    }

    weightSum += weight;
    x += positions[neighbor][0] * weight;
    y += positions[neighbor][1] * weight;
    z += positions[neighbor][2] * weight;
  }

  if (weightSum <= 1e-8) {
    return computeUniformPatchAverage(patch.adjacency[vertexIndex], positions);
  }

  return [x / weightSum, y / weightSum, z / weightSum];
}

function computeUniformPatchAverage(neighbors: number[], positions: Vec3[]): Vec3 {
  if (!neighbors || neighbors.length === 0) {
    return [0, 0, 0];
  }

  let x = 0;
  let y = 0;
  let z = 0;
  for (let i = 0; i < neighbors.length; i += 1) {
    x += positions[neighbors[i]][0];
    y += positions[neighbors[i]][1];
    z += positions[neighbors[i]][2];
  }

  const invCount = 1 / neighbors.length;
  return [x * invCount, y * invCount, z * invCount];
}

function accumulateTriangleCotangentWeights(
  weightByNeighbor: Map<number, number>,
  triangle: PatchTriangle,
  vertexIndex: number,
  positions: Vec3[],
): void {
  if (triangle.a === vertexIndex) {
    accumulateNeighborWeight(weightByNeighbor, triangle.b, cotangentAtVertex(positions[triangle.c], positions[triangle.a], positions[triangle.b]));
    accumulateNeighborWeight(weightByNeighbor, triangle.c, cotangentAtVertex(positions[triangle.b], positions[triangle.a], positions[triangle.c]));
    return;
  }

  if (triangle.b === vertexIndex) {
    accumulateNeighborWeight(weightByNeighbor, triangle.a, cotangentAtVertex(positions[triangle.c], positions[triangle.b], positions[triangle.a]));
    accumulateNeighborWeight(weightByNeighbor, triangle.c, cotangentAtVertex(positions[triangle.a], positions[triangle.b], positions[triangle.c]));
    return;
  }

  if (triangle.c === vertexIndex) {
    accumulateNeighborWeight(weightByNeighbor, triangle.a, cotangentAtVertex(positions[triangle.b], positions[triangle.c], positions[triangle.a]));
    accumulateNeighborWeight(weightByNeighbor, triangle.b, cotangentAtVertex(positions[triangle.a], positions[triangle.c], positions[triangle.b]));
  }
}

function accumulateNeighborWeight(
  weightByNeighbor: Map<number, number>,
  neighbor: number,
  weight: number,
): void {
  const clampedWeight = clamp(weight, 0, 12);
  if (clampedWeight <= 1e-8) {
    return;
  }

  weightByNeighbor.set(neighbor, (weightByNeighbor.get(neighbor) ?? 0) + clampedWeight);
}

function cotangentAtVertex(vertex: Vec3, pointA: Vec3, pointB: Vec3): number {
  const edgeA = subtract(pointA, vertex);
  const edgeB = subtract(pointB, vertex);
  const crossLength = length(cross(edgeA, edgeB));
  if (crossLength <= 1e-10) {
    return 0;
  }

  return dot(edgeA, edgeB) / crossLength;
}

function computeSeamPlaneTarget(
  candidate: Vec3,
  neighbors: number[],
  patch: PatchWork,
  boundaryFrameByMeshVertex: Map<number, BoundaryFrame>,
): Vec3 | null {
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  let count = 0;

  for (let i = 0; i < neighbors.length; i += 1) {
    const neighbor = patch.vertices[neighbors[i]];
    if (!neighbor.isBoundary || neighbor.meshVertexId == null) {
      continue;
    }

    const frame = boundaryFrameByMeshVertex.get(neighbor.meshVertexId);
    if (!frame) {
      continue;
    }

    const delta = subtract(candidate, frame.position);
    const projected = subtract(candidate, scale(frame.normal, dot(delta, frame.normal)));
    sumX += projected[0];
    sumY += projected[1];
    sumZ += projected[2];
    count += 1;
  }

  if (count === 0) {
    return null;
  }

  return [sumX / count, sumY / count, sumZ / count];
}

function computeSeamRelaxTarget(
  candidate: Vec3,
  uv: Vec2,
  seamSupport: SeamSupportVertex[],
  targetEdgeLength: number,
): Vec3 | null {
  const radius = targetEdgeLength * 3.5;
  const radiusSq = radius * radius;
  let sumWeight = 0;
  let sumPosition: Vec3 = [0, 0, 0];
  let sumNormal: Vec3 = [0, 0, 0];

  for (let i = 0; i < seamSupport.length; i += 1) {
    const support = seamSupport[i];
    const dx = support.uv[0] - uv[0];
    const dy = support.uv[1] - uv[1];
    const distanceSq = dx * dx + dy * dy;
    if (distanceSq > radiusSq) {
      continue;
    }

    const weight = 1 / Math.max(distanceSq, targetEdgeLength * targetEdgeLength * 0.1);
    sumWeight += weight;
    sumPosition = add(sumPosition, scale(support.position, weight));
    sumNormal = add(sumNormal, scale(support.normal, weight));
  }

  if (sumWeight <= 1e-8) {
    return null;
  }

  const averagePosition = scale(sumPosition, 1 / sumWeight);
  const averageNormal = normalize(scale(sumNormal, 1 / sumWeight));
  const projected = subtract(
    candidate,
    scale(averageNormal, dot(subtract(candidate, averagePosition), averageNormal)),
  );
  return lerpVec3(projected, averagePosition, 0.25);
}

function stabilizeVertexMove(
  patch: PatchWork,
  vertexIndex: number,
  proposedPosition: Vec3,
  averageNormal: Vec3,
  maxTriangleAspectRatio: number,
): Vec3 {
  let candidate = proposedPosition;
  const current = patch.vertices[vertexIndex].position;

  for (let attempt = 0; attempt < 4; attempt += 1) {
    let valid = true;

    for (let triangleIndex = 0; triangleIndex < patch.triangles.length; triangleIndex += 1) {
      const triangle = patch.triangles[triangleIndex];
      if (triangle.a !== vertexIndex && triangle.b !== vertexIndex && triangle.c !== vertexIndex) {
        continue;
      }

      const a = triangle.a === vertexIndex ? candidate : patch.vertices[triangle.a].position;
      const b = triangle.b === vertexIndex ? candidate : patch.vertices[triangle.b].position;
      const c = triangle.c === vertexIndex ? candidate : patch.vertices[triangle.c].position;
      const normal = computeTriangleNormal(a, b, c);
      if (dot(normal, averageNormal) <= 1e-5) {
        valid = false;
        break;
      }

      if (triangleAspectRatio3D(a, b, c) > maxTriangleAspectRatio) {
        valid = false;
        break;
      }
    }

    if (valid) {
      return candidate;
    }

    candidate = lerpVec3(current, candidate, 0.5);
  }

  return current;
}

function estimateSeamBias(
  patch: PatchWork,
  neighbors: number[],
  boundaryFrameByMeshVertex: Map<number, BoundaryFrame>,
  targetEdgeLength: number,
): Vec3 {
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  let count = 0;

  for (let i = 0; i < neighbors.length; i += 1) {
    const neighbor = patch.vertices[neighbors[i]];
    if (!neighbor.isBoundary || neighbor.meshVertexId == null) {
      continue;
    }

    const frame = boundaryFrameByMeshVertex.get(neighbor.meshVertexId);
    if (!frame) {
      continue;
    }

    const bias = add(
      frame.position,
      add(scale(frame.inward, targetEdgeLength * 0.35), scale(frame.normal, targetEdgeLength * 0.08)),
    );
    sumX += bias[0];
    sumY += bias[1];
    sumZ += bias[2];
    count += 1;
  }

  if (count === 0) {
    return [0, 0, 0];
  }

  return [sumX / count, sumY / count, sumZ / count];
}

function addFaceConnectivity(mesh: Mesh, faceId: number, a: number, b: number, c: number): void {
  mesh.vertexFaces[a].push(faceId);
  mesh.vertexFaces[b].push(faceId);
  mesh.vertexFaces[c].push(faceId);
  ensureNeighborLink(mesh.vertexNeighbors[a], b);
  ensureNeighborLink(mesh.vertexNeighbors[a], c);
  ensureNeighborLink(mesh.vertexNeighbors[b], a);
  ensureNeighborLink(mesh.vertexNeighbors[b], c);
  ensureNeighborLink(mesh.vertexNeighbors[c], a);
  ensureNeighborLink(mesh.vertexNeighbors[c], b);
  addEdgeFace(mesh.edgeToFaces, a, b, faceId);
  addEdgeFace(mesh.edgeToFaces, b, c, faceId);
  addEdgeFace(mesh.edgeToFaces, c, a, faceId);
}

function appendVertex(mesh: Mesh, position: Vec3): number {
  const vertexId = mesh.positions.length / 3;
  mesh.positions.push(position[0], position[1], position[2]);
  mesh.vertexFaces.push([]);
  mesh.vertexNeighbors.push([]);
  mesh.vertexNormals.push(0, 0, 0);
  mesh.dirtyVertices.add(vertexId);
  return vertexId;
}

function appendFace(mesh: Mesh, a: number, b: number, c: number): number {
  const faceId = mesh.indices.length / 3;
  mesh.indices.push(a, b, c);
  mesh.faceNormals.push(0, 0, 0);
  addFaceConnectivity(mesh, faceId, a, b, c);
  mesh.dirtyFaces.add(faceId);
  mesh.dirtyVertices.add(a);
  mesh.dirtyVertices.add(b);
  mesh.dirtyVertices.add(c);
  return faceId;
}

function addEdgeFace(edgeToFaces: Map<string, number[]>, a: number, b: number, faceId: number): void {
  const key = makeEdgeKey(a, b);
  const faces = edgeToFaces.get(key);
  if (faces) {
    faces.push(faceId);
    return;
  }

  edgeToFaces.set(key, [faceId]);
}

function pushBoundaryNeighbor(boundaryNeighbors: Map<number, number[]>, a: number, b: number): void {
  const neighbors = boundaryNeighbors.get(a);
  if (neighbors) {
    neighbors.push(b);
    return;
  }

  boundaryNeighbors.set(a, [b]);
}

function ensureNeighborLink(neighbors: number[], target: number): void {
  if (!neighbors.includes(target)) {
    neighbors.push(target);
  }
}

function pointOnSurface(plane: PlaneFit, fit: QuadraticSurfaceFit, u: number, v: number): Vec3 {
  const w = evaluateQuadraticSurface(fit.coefficients, u, v);
  return add(
    add(plane.origin, scale(plane.uAxis, u)),
    add(scale(plane.vAxis, v), scale(plane.normal, w)),
  );
}

function evaluateQuadraticSurface(
  coefficients: QuadraticSurfaceFit['coefficients'],
  u: number,
  v: number,
): number {
  return (
    coefficients[0] * u * u +
    coefficients[1] * u * v +
    coefficients[2] * v * v +
    coefficients[3] * u +
    coefficients[4] * v +
    coefficients[5]
  );
}

function getVertex(mesh: Mesh, vertexId: number): Vec3 {
  const offset = vertexId * 3;
  return [mesh.positions[offset], mesh.positions[offset + 1], mesh.positions[offset + 2]];
}

function getVertexNormal(mesh: Mesh, vertexId: number): Vec3 {
  const offset = vertexId * 3;
  return normalize([
    mesh.vertexNormals[offset],
    mesh.vertexNormals[offset + 1],
    mesh.vertexNormals[offset + 2],
  ]);
}

function getFaceNormal(mesh: Mesh, faceId: number): Vec3 {
  const offset = faceId * 3;
  return normalize([
    mesh.faceNormals[offset],
    mesh.faceNormals[offset + 1],
    mesh.faceNormals[offset + 2],
  ]);
}

function averagePoints(points: Vec3[]): Vec3 {
  let x = 0;
  let y = 0;
  let z = 0;
  for (let i = 0; i < points.length; i += 1) {
    x += points[i][0];
    y += points[i][1];
    z += points[i][2];
  }

  const invCount = 1 / Math.max(points.length, 1);
  return [x * invCount, y * invCount, z * invCount];
}

function averageVertexNormals(mesh: Mesh, vertices: number[]): Vec3 {
  let nx = 0;
  let ny = 0;
  let nz = 0;
  for (let i = 0; i < vertices.length; i += 1) {
    const offset = vertices[i] * 3;
    nx += mesh.vertexNormals[offset];
    ny += mesh.vertexNormals[offset + 1];
    nz += mesh.vertexNormals[offset + 2];
  }

  return normalize([nx, ny, nz]);
}

function orientNormalToReference(normal: Vec3, reference: Vec3): Vec3 {
  return dot(normal, reference) < 0 ? scale(normal, -1) : normalize(normal);
}

function computeTriangleNormal(a: Vec3, b: Vec3, c: Vec3): Vec3 {
  return normalize(cross(subtract(b, a), subtract(c, a)));
}

function projectPointToPlane(point: Vec3, plane: PlaneFit): [number, number, number] {
  const relative = subtract(point, plane.origin);
  return [
    dot(relative, plane.uAxis),
    dot(relative, plane.vAxis),
    dot(relative, plane.normal),
  ];
}

function polygonArea(points: Vec2[]): number {
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    sum += current[0] * next[1] - next[0] * current[1];
  }

  return sum * 0.5;
}

function computePolygonCentroid2D(points: Vec2[]): Vec2 {
  const area = polygonArea(points);
  if (Math.abs(area) <= 1e-10) {
    return averagePoints2D(points);
  }

  let centroidX = 0;
  let centroidY = 0;
  for (let i = 0; i < points.length; i += 1) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    const crossValue = current[0] * next[1] - next[0] * current[1];
    centroidX += (current[0] + next[0]) * crossValue;
    centroidY += (current[1] + next[1]) * crossValue;
  }

  const scaleFactor = 1 / (6 * area);
  return [centroidX * scaleFactor, centroidY * scaleFactor];
}

function averagePoints2D(points: Vec2[]): Vec2 {
  let x = 0;
  let y = 0;
  for (let i = 0; i < points.length; i += 1) {
    x += points[i][0];
    y += points[i][1];
  }

  const invCount = 1 / Math.max(points.length, 1);
  return [x * invCount, y * invCount];
}

function isValidFanCenter(points: Vec2[], center: Vec2): boolean {
  if (!pointInPolygon2D(center, points)) {
    return false;
  }

  const winding = polygonArea(points) >= 0 ? 1 : -1;
  const edgeTolerance = Math.max(estimateAverageEdgeLength2D(points) * 1e-5, 1e-9);
  for (let i = 0; i < points.length; i += 1) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    const signedDistance = orient2D(current, next, center);
    if (winding > 0 ? signedDistance <= edgeTolerance : signedDistance >= -edgeTolerance) {
      return false;
    }
  }

  return true;
}

function isConvexCorner(a: Vec2, b: Vec2, c: Vec2, winding: number): boolean {
  const crossZ = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
  return winding > 0 ? crossZ > 1e-10 : crossZ < -1e-10;
}

function containsOtherPolygonPoint(
  points: Vec2[],
  remaining: number[],
  a: number,
  b: number,
  c: number,
): boolean {
  for (let i = 0; i < remaining.length; i += 1) {
    const candidate = remaining[i];
    if (candidate === a || candidate === b || candidate === c) {
      continue;
    }

    if (pointInTriangle2D(points[candidate], points[a], points[b], points[c])) {
      return true;
    }
  }

  return false;
}

function pointInTriangle2D(point: Vec2, a: Vec2, b: Vec2, c: Vec2): boolean {
  const area = orient2D(a, b, c);
  const w0 = orient2D(point, b, c);
  const w1 = orient2D(a, point, c);
  const w2 = orient2D(a, b, point);
  if (area < 0) {
    return w0 <= 1e-10 && w1 <= 1e-10 && w2 <= 1e-10;
  }

  return w0 >= -1e-10 && w1 >= -1e-10 && w2 >= -1e-10;
}

function pointInPolygon2D(point: Vec2, polygon: Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const current = polygon[i];
    const previous = polygon[j];
    const intersects =
      (current[1] > point[1]) !== (previous[1] > point[1]) &&
      point[0] <
        ((previous[0] - current[0]) * (point[1] - current[1])) /
          Math.max(previous[1] - current[1], 1e-12) +
          current[0];
    if (intersects) {
      inside = !inside;
    }
  }

  return inside;
}

function clipPolygonToHalfPlane(polygon: Vec2[], a: Vec2, b: Vec2, winding: number): Vec2[] {
  if (polygon.length === 0) {
    return polygon;
  }

  const clipped: Vec2[] = [];
  let previous = polygon[polygon.length - 1];
  let previousInside = isPointInsideHalfPlane(previous, a, b, winding);

  for (let i = 0; i < polygon.length; i += 1) {
    const current = polygon[i];
    const currentInside = isPointInsideHalfPlane(current, a, b, winding);

    if (currentInside !== previousInside) {
      clipped.push(intersectSegmentWithLine(previous, current, a, b));
    }

    if (currentInside) {
      clipped.push(current);
    }

    previous = current;
    previousInside = currentInside;
  }

  return clipped;
}

function isPointInsideHalfPlane(point: Vec2, a: Vec2, b: Vec2, winding: number): boolean {
  const signedDistance = orient2D(a, b, point);
  return winding > 0 ? signedDistance >= -1e-9 : signedDistance <= 1e-9;
}

function intersectSegmentWithLine(start: Vec2, end: Vec2, a: Vec2, b: Vec2): Vec2 {
  const startSide = orient2D(a, b, start);
  const endSide = orient2D(a, b, end);
  const denominator = startSide - endSide;
  if (Math.abs(denominator) <= 1e-12) {
    return [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5];
  }

  const t = clamp(startSide / denominator, 0, 1);
  return [
    start[0] + (end[0] - start[0]) * t,
    start[1] + (end[1] - start[1]) * t,
  ];
}

function hasPolygonSelfIntersection(points: Vec2[]): boolean {
  for (let i = 0; i < points.length; i += 1) {
    const a0 = points[i];
    const a1 = points[(i + 1) % points.length];
    for (let j = i + 1; j < points.length; j += 1) {
      if (Math.abs(i - j) <= 1 || (i === 0 && j === points.length - 1)) {
        continue;
      }

      const b0 = points[j];
      const b1 = points[(j + 1) % points.length];
      if (segmentsIntersect2D(a0, a1, b0, b1)) {
        return true;
      }
    }
  }

  return false;
}

function segmentsIntersect2D(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2): boolean {
  const o1 = orient2D(a0, a1, b0);
  const o2 = orient2D(a0, a1, b1);
  const o3 = orient2D(b0, b1, a0);
  const o4 = orient2D(b0, b1, a1);
  return (o1 > 0) !== (o2 > 0) && (o3 > 0) !== (o4 > 0);
}

function orient2D(a: Vec2, b: Vec2, c: Vec2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function triangleAspectRatio3D(a: Vec3, b: Vec3, c: Vec3): number {
  const ab = distance(a, b);
  const bc = distance(b, c);
  const ca = distance(c, a);
  const longest = Math.max(ab, bc, ca);
  const shortest = Math.max(Math.min(ab, bc, ca), 1e-8);
  return longest / shortest;
}

function diagonalizeSymmetric3x3(matrix: number[][]): { values: number[]; vectors: Vec3[] } {
  const a = matrix.map((row) => row.slice());
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  for (let iteration = 0; iteration < 12; iteration += 1) {
    let p = 0;
    let q = 1;
    let maxValue = Math.abs(a[0][1]);

    if (Math.abs(a[0][2]) > maxValue) {
      p = 0;
      q = 2;
      maxValue = Math.abs(a[0][2]);
    }

    if (Math.abs(a[1][2]) > maxValue) {
      p = 1;
      q = 2;
      maxValue = Math.abs(a[1][2]);
    }

    if (maxValue < 1e-10) {
      break;
    }

    const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
    const tangent = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const cosine = 1 / Math.sqrt(tangent * tangent + 1);
    const sine = tangent * cosine;

    const app = a[p][p];
    const aqq = a[q][q];
    const apq = a[p][q];
    a[p][p] = app - tangent * apq;
    a[q][q] = aqq + tangent * apq;
    a[p][q] = 0;
    a[q][p] = 0;

    for (let r = 0; r < 3; r += 1) {
      if (r === p || r === q) {
        continue;
      }

      const arp = a[r][p];
      const arq = a[r][q];
      a[r][p] = cosine * arp - sine * arq;
      a[p][r] = a[r][p];
      a[r][q] = sine * arp + cosine * arq;
      a[q][r] = a[r][q];
    }

    for (let r = 0; r < 3; r += 1) {
      const vrp = v[r][p];
      const vrq = v[r][q];
      v[r][p] = cosine * vrp - sine * vrq;
      v[r][q] = sine * vrp + cosine * vrq;
    }
  }

  return {
    values: [a[0][0], a[1][1], a[2][2]],
    vectors: [
      [v[0][0], v[1][0], v[2][0]],
      [v[0][1], v[1][1], v[2][1]],
      [v[0][2], v[1][2], v[2][2]],
    ],
  };
}

function sortEigenPairs(values: number[], vectors: Vec3[]): { values: number[]; vectors: Vec3[] } {
  const pairs = values.map((value, index) => ({ value, vector: vectors[index] }));
  pairs.sort((a, b) => a.value - b.value);
  return {
    values: pairs.map((pair) => pair.value),
    vectors: pairs.map((pair) => pair.vector),
  };
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] | null {
  const n = rhs.length;
  const a = matrix.map((row, rowIndex) => row.slice().concat(rhs[rowIndex]));

  for (let pivot = 0; pivot < n; pivot += 1) {
    let maxRow = pivot;
    let maxValue = Math.abs(a[pivot][pivot]);
    for (let row = pivot + 1; row < n; row += 1) {
      const candidate = Math.abs(a[row][pivot]);
      if (candidate > maxValue) {
        maxValue = candidate;
        maxRow = row;
      }
    }

    if (maxValue < 1e-10) {
      return null;
    }

    if (maxRow !== pivot) {
      const temp = a[pivot];
      a[pivot] = a[maxRow];
      a[maxRow] = temp;
    }

    const pivotValue = a[pivot][pivot];
    for (let column = pivot; column <= n; column += 1) {
      a[pivot][column] /= pivotValue;
    }

    for (let row = 0; row < n; row += 1) {
      if (row === pivot) {
        continue;
      }

      const factor = a[row][pivot];
      if (Math.abs(factor) < 1e-12) {
        continue;
      }

      for (let column = pivot; column <= n; column += 1) {
        a[row][column] -= factor * a[pivot][column];
      }
    }
  }

  return a.map((row) => row[n]);
}

function createMatrix(rows: number, columns: number): number[][] {
  return Array.from({ length: rows }, () => new Array(columns).fill(0));
}

function createSequentialIds(count: number): number[] {
  return Array.from({ length: count }, (_, index) => index);
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) * 0.5
    : sorted[middle];
}

function now(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }

  return Date.now();
}

function makeEdgeKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function parseEdgeKey(key: string): [number, number] {
  const split = key.indexOf(':');
  return [Number(key.slice(0, split)), Number(key.slice(split + 1))];
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a: Vec3, scalar: number): Vec3 {
  return [a[0] * scalar, a[1] * scalar, a[2] * scalar];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function normalize(a: Vec3): Vec3 {
  const len = length(a);
  if (len <= 1e-12) {
    return [0, 1, 0];
  }

  return [a[0] / len, a[1] / len, a[2] / len];
}

function distance(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function distanceSquared(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

function lerpVec3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function safeAcos(value: number): number {
  return Math.acos(clamp(value, -1, 1));
}

function distance2D(a: Vec2, b: Vec2): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function estimateAverageEdgeLength2D(points: Vec2[]): number {
  if (points.length === 0) {
    return 0;
  }

  let lengthSum = 0;
  for (let i = 0; i < points.length; i += 1) {
    lengthSum += distance2D(points[i], points[(i + 1) % points.length]);
  }

  return lengthSum / points.length;
}
