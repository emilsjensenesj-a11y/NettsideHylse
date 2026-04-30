export interface SelectionEditResult {
  positions: Float32Array;
  indices: Uint32Array;
  referencePositions: Float32Array;
  selectedTriangleMask: Uint8Array;
  triangleSourceIds: Int32Array;
}

interface LaplacianSmoothOptions {
  preserveOpenBoundaryVertices?: boolean;
}

export function refineSelectedTriangles(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  referencePositions: ArrayLike<number>,
  selectedTriangleMask: Uint8Array,
): SelectionEditResult | null {
  if (selectedTriangleMask.length === 0) {
    return null;
  }

  const nextPositions = Array.from(positions);
  const nextReferencePositions = Array.from(referencePositions);
  const splitEdges = new Set<string>();
  const midpointMap = new Map<string, number>();

  for (let triangle = 0; triangle < selectedTriangleMask.length; triangle += 1) {
    if (selectedTriangleMask[triangle] === 0) {
      continue;
    }

    const triOffset = triangle * 3;
    const a = indices[triOffset];
    const b = indices[triOffset + 1];
    const c = indices[triOffset + 2];
    splitEdges.add(makeEdgeKey(a, b));
    splitEdges.add(makeEdgeKey(b, c));
    splitEdges.add(makeEdgeKey(c, a));
  }

  if (splitEdges.size === 0) {
    return null;
  }

  const nextIndices: number[] = [];
  const nextSelectionMask: number[] = [];
  const nextTriangleSources: number[] = [];
  const selectedChildTriangles: number[] = [];

  for (let triangle = 0; triangle < indices.length / 3; triangle += 1) {
    const triOffset = triangle * 3;
    const a = indices[triOffset];
    const b = indices[triOffset + 1];
    const c = indices[triOffset + 2];
    const selected = selectedTriangleMask[triangle] !== 0;

    const mab = splitEdges.has(makeEdgeKey(a, b))
      ? getOrCreateMidpoint(a, b, nextPositions, nextReferencePositions, midpointMap, positions, referencePositions)
      : -1;
    const mbc = splitEdges.has(makeEdgeKey(b, c))
      ? getOrCreateMidpoint(b, c, nextPositions, nextReferencePositions, midpointMap, positions, referencePositions)
      : -1;
    const mca = splitEdges.has(makeEdgeKey(c, a))
      ? getOrCreateMidpoint(c, a, nextPositions, nextReferencePositions, midpointMap, positions, referencePositions)
      : -1;

    const splitCount = (mab >= 0 ? 1 : 0) + (mbc >= 0 ? 1 : 0) + (mca >= 0 ? 1 : 0);
    if (splitCount === 0) {
      pushTriangle(
        nextIndices,
        nextSelectionMask,
        nextTriangleSources,
        selectedChildTriangles,
        a,
        b,
        c,
        selected,
        triangle,
      );
      continue;
    }

    if (splitCount === 1) {
      if (mab >= 0) {
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          a,
          mab,
          c,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          mab,
          b,
          c,
          selected,
          triangle,
        );
      } else if (mbc >= 0) {
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          a,
          b,
          mbc,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          a,
          mbc,
          c,
          selected,
          triangle,
        );
      } else {
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          a,
          b,
          mca,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          mca,
          b,
          c,
          selected,
          triangle,
        );
      }

      continue;
    }

    if (splitCount === 2) {
      if (mab >= 0 && mbc >= 0) {
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          a,
          mab,
          c,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          mab,
          mbc,
          c,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          mab,
          b,
          mbc,
          selected,
          triangle,
        );
      } else if (mbc >= 0 && mca >= 0) {
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          a,
          b,
          mca,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          b,
          mbc,
          mca,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          mbc,
          c,
          mca,
          selected,
          triangle,
        );
      } else {
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          a,
          mab,
          mca,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          mab,
          b,
          c,
          selected,
          triangle,
        );
        pushTriangle(
          nextIndices,
          nextSelectionMask,
          nextTriangleSources,
          selectedChildTriangles,
          mab,
          c,
          mca,
          selected,
          triangle,
        );
      }

      continue;
    }

    pushTriangle(
      nextIndices,
      nextSelectionMask,
      nextTriangleSources,
      selectedChildTriangles,
      a,
      mab,
      mca,
      selected,
      triangle,
    );
    pushTriangle(
      nextIndices,
      nextSelectionMask,
      nextTriangleSources,
      selectedChildTriangles,
      mab,
      b,
      mbc,
      selected,
      triangle,
    );
    pushTriangle(
      nextIndices,
      nextSelectionMask,
      nextTriangleSources,
      selectedChildTriangles,
      mca,
      mbc,
      c,
      selected,
      triangle,
    );
    pushTriangle(
      nextIndices,
      nextSelectionMask,
      nextTriangleSources,
      selectedChildTriangles,
      mab,
      mbc,
      mca,
      selected,
      triangle,
    );
  }

  nextSelectionMask.fill(0);
  for (let i = 0; i < selectedChildTriangles.length; i += 1) {
    nextSelectionMask[selectedChildTriangles[i]] = 1;
  }

  return {
    positions: new Float32Array(nextPositions),
    indices: new Uint32Array(nextIndices),
    referencePositions: new Float32Array(nextReferencePositions),
    selectedTriangleMask: Uint8Array.from(nextSelectionMask),
    triangleSourceIds: Int32Array.from(nextTriangleSources),
  };
}

export function laplacianSmoothSelected(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  referencePositions: ArrayLike<number>,
  selectedTriangleMask: Uint8Array,
  intensity: number,
  iterations: number,
  options: LaplacianSmoothOptions = {},
): SelectionEditResult | null {
  if (selectedTriangleMask.length === 0) {
    return null;
  }

  const selectedVertices = new Uint8Array(Math.ceil(positions.length / 3));
  let selectedVertexCount = 0;
  for (let triangle = 0; triangle < selectedTriangleMask.length; triangle += 1) {
    if (selectedTriangleMask[triangle] === 0) {
      continue;
    }

    const triOffset = triangle * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = indices[triOffset + corner];
      if (selectedVertices[vertex] === 0) {
        selectedVertices[vertex] = 1;
        selectedVertexCount += 1;
      }
    }
  }

  if (selectedVertexCount === 0) {
    return null;
  }

  const resolvedIntensity = clamp(intensity, 0, 1);
  const resolvedIterations = Math.max(1, Math.round(iterations));
  let current = new Float32Array(positions);
  let next = new Float32Array(current.length);
  const neighborLists = buildNeighborLists(indices, selectedVertices.length, selectedTriangleMask);
  const vertexFaces = buildVertexFaceLists(indices, selectedVertices.length);
  const openBoundaryVertices = options.preserveOpenBoundaryVertices
    ? buildOpenBoundaryVertexMask(indices, selectedVertices.length)
    : null;
  const movableVertices = new Uint8Array(selectedVertices.length);
  for (let vertex = 0; vertex < selectedVertices.length; vertex += 1) {
    if (selectedVertices[vertex] === 0) {
      continue;
    }

    const faces = vertexFaces[vertex];
    let movable = faces.length > 0;
    for (let i = 0; i < faces.length; i += 1) {
      if (selectedTriangleMask[faces[i]] === 0) {
        movable = false;
        break;
      }
    }

    if (movable && openBoundaryVertices?.[vertex] === 1) {
      movable = false;
    }

    if (movable) {
      movableVertices[vertex] = 1;
    }
  }

  for (let iteration = 0; iteration < resolvedIterations; iteration += 1) {
    next.set(current);
    for (let vertex = 0; vertex < selectedVertices.length; vertex += 1) {
      if (movableVertices[vertex] === 0) {
        continue;
      }

      const neighbors = neighborLists[vertex];
      if (neighbors.length === 0) {
        continue;
      }

      const offset = vertex * 3;
      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      for (let i = 0; i < neighbors.length; i += 1) {
        const neighborOffset = neighbors[i] * 3;
        sumX += current[neighborOffset];
        sumY += current[neighborOffset + 1];
        sumZ += current[neighborOffset + 2];
      }

      const invCount = 1 / neighbors.length;
      next[offset] = current[offset] + (sumX * invCount - current[offset]) * resolvedIntensity;
      next[offset + 1] = current[offset + 1] + (sumY * invCount - current[offset + 1]) * resolvedIntensity;
      next[offset + 2] = current[offset + 2] + (sumZ * invCount - current[offset + 2]) * resolvedIntensity;
    }

    const swap = current;
    current = next;
    next = swap;
  }

  return {
    positions: current,
    indices: new Uint32Array(indices),
    referencePositions: new Float32Array(referencePositions),
    selectedTriangleMask: selectedTriangleMask.slice(),
    triangleSourceIds: createIdentitySources(selectedTriangleMask.length),
  };
}

export function laplacianSmoothSelectionBoundary(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  referencePositions: ArrayLike<number>,
  selectedTriangleMask: Uint8Array,
  intensity: number,
  iterations: number,
): SelectionEditResult | null {
  if (selectedTriangleMask.length === 0) {
    return null;
  }

  const vertexCount = Math.ceil(positions.length / 3);
  const boundaryNeighbors = buildSelectionBoundaryNeighborLists(indices, vertexCount, selectedTriangleMask);
  const boundaryNormals = buildVertexNormals(indices, positions, vertexCount);
  const boundaryVertices: number[] = [];
  for (let vertex = 0; vertex < boundaryNeighbors.length; vertex += 1) {
    if (boundaryNeighbors[vertex].length > 0) {
      boundaryVertices.push(vertex);
    }
  }

  if (boundaryVertices.length === 0) {
    return null;
  }

  const resolvedIntensity = clamp(intensity, 0, 1);
  const resolvedIterations = Math.max(1, Math.round(iterations));
  let current = new Float32Array(positions);
  let next = new Float32Array(current.length);
  let movedVertices = 0;

  for (let iteration = 0; iteration < resolvedIterations; iteration += 1) {
    next.set(current);
    let movedThisIteration = 0;
    for (let i = 0; i < boundaryVertices.length; i += 1) {
      const vertex = boundaryVertices[i];
      const neighbors = boundaryNeighbors[vertex];
      // A clean boundary curve vertex should have exactly two boundary neighbors.
      // Endpoints and junctions are left in place to avoid collapsing corners.
      if (neighbors.length !== 2) {
        continue;
      }

      const offset = vertex * 3;
      const neighborA = neighbors[0] * 3;
      const neighborB = neighbors[1] * 3;
      const targetX = (current[neighborA] + current[neighborB]) * 0.5;
      const targetY = (current[neighborA + 1] + current[neighborB + 1]) * 0.5;
      const targetZ = (current[neighborA + 2] + current[neighborB + 2]) * 0.5;

      let deltaX = targetX - current[offset];
      let deltaY = targetY - current[offset + 1];
      let deltaZ = targetZ - current[offset + 2];

      // Keep most of the motion tangential to the current surface so the
      // boundary relaxes as a curve instead of drifting strongly in or out.
      const nx = boundaryNormals[offset];
      const ny = boundaryNormals[offset + 1];
      const nz = boundaryNormals[offset + 2];
      const normalLength = Math.hypot(nx, ny, nz);
      if (normalLength > 1e-8) {
        const invNormalLength = 1 / normalLength;
        const normalizedX = nx * invNormalLength;
        const normalizedY = ny * invNormalLength;
        const normalizedZ = nz * invNormalLength;
        const normalComponent =
          deltaX * normalizedX + deltaY * normalizedY + deltaZ * normalizedZ;
        deltaX -= normalComponent * normalizedX;
        deltaY -= normalComponent * normalizedY;
        deltaZ -= normalComponent * normalizedZ;
      }

      next[offset] = current[offset] + deltaX * resolvedIntensity;
      next[offset + 1] = current[offset + 1] + deltaY * resolvedIntensity;
      next[offset + 2] = current[offset + 2] + deltaZ * resolvedIntensity;
      movedThisIteration += 1;
    }

    movedVertices += movedThisIteration;
    const swap = current;
    current = next;
    next = swap;
  }

  if (movedVertices === 0) {
    return null;
  }

  return {
    positions: current,
    indices: new Uint32Array(indices),
    referencePositions: new Float32Array(referencePositions),
    selectedTriangleMask: selectedTriangleMask.slice(),
    triangleSourceIds: createIdentitySources(selectedTriangleMask.length),
  };
}

function getOrCreateMidpoint(
  a: number,
  b: number,
  positionsOut: number[],
  referencePositionsOut: number[],
  midpointMap: Map<string, number>,
  positions: ArrayLike<number>,
  referencePositions: ArrayLike<number>,
): number {
  const key = makeEdgeKey(a, b);
  const existing = midpointMap.get(key);
  if (existing != null) {
    return existing;
  }

  const aOffset = a * 3;
  const bOffset = b * 3;
  const vertex = positionsOut.length / 3;
  positionsOut.push(
    (positions[aOffset] + positions[bOffset]) * 0.5,
    (positions[aOffset + 1] + positions[bOffset + 1]) * 0.5,
    (positions[aOffset + 2] + positions[bOffset + 2]) * 0.5,
  );
  referencePositionsOut.push(
    (referencePositions[aOffset] + referencePositions[bOffset]) * 0.5,
    (referencePositions[aOffset + 1] + referencePositions[bOffset + 1]) * 0.5,
    (referencePositions[aOffset + 2] + referencePositions[bOffset + 2]) * 0.5,
  );
  midpointMap.set(key, vertex);
  return vertex;
}

function pushTriangle(
  indexTarget: number[],
  selectionTarget: number[],
  sourceTarget: number[],
  selectedChildTarget: number[],
  a: number,
  b: number,
  c: number,
  selected: boolean,
  sourceTriangle: number,
): void {
  if (a === b || b === c || c === a) {
    return;
  }

  const triangleIndex = indexTarget.length / 3;
  indexTarget.push(a, b, c);
  selectionTarget.push(0);
  sourceTarget.push(sourceTriangle);
  if (selected) {
    selectedChildTarget.push(triangleIndex);
  }
}

function buildNeighborLists(
  indices: ArrayLike<number>,
  vertexCount: number,
  selectedTriangleMask: Uint8Array,
): Uint32Array[] {
  const neighbors = Array.from({ length: vertexCount }, () => new Set<number>());
  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    if (selectedTriangleMask[triangle / 3] === 0) {
      continue;
    }

    const a = indices[triangle];
    const b = indices[triangle + 1];
    const c = indices[triangle + 2];
    neighbors[a].add(b);
    neighbors[a].add(c);
    neighbors[b].add(a);
    neighbors[b].add(c);
    neighbors[c].add(a);
    neighbors[c].add(b);
  }

  return neighbors.map((entry) => Uint32Array.from(entry));
}

function buildSelectionBoundaryNeighborLists(
  indices: ArrayLike<number>,
  vertexCount: number,
  selectedTriangleMask: Uint8Array,
): Uint32Array[] {
  const edgeMap = new Map<string, { a: number; b: number; total: number; selected: number }>();
  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    const faceIndex = triangle / 3;
    const selected = selectedTriangleMask[faceIndex] !== 0;
    trackBoundaryEdge(edgeMap, indices[triangle], indices[triangle + 1], selected);
    trackBoundaryEdge(edgeMap, indices[triangle + 1], indices[triangle + 2], selected);
    trackBoundaryEdge(edgeMap, indices[triangle + 2], indices[triangle], selected);
  }

  const neighbors = Array.from({ length: vertexCount }, () => new Set<number>());
  for (const edge of edgeMap.values()) {
    const crossesSelectionBoundary =
      (edge.selected > 0 && edge.selected < edge.total) || (edge.selected === 1 && edge.total === 1);
    if (!crossesSelectionBoundary) {
      continue;
    }

    neighbors[edge.a].add(edge.b);
    neighbors[edge.b].add(edge.a);
  }

  return neighbors.map((entry) => Uint32Array.from(entry));
}

function buildOpenBoundaryVertexMask(
  indices: ArrayLike<number>,
  vertexCount: number,
): Uint8Array {
  const edgeMap = new Map<string, { a: number; b: number; total: number; selected: number }>();
  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    trackBoundaryEdge(edgeMap, indices[triangle], indices[triangle + 1], true);
    trackBoundaryEdge(edgeMap, indices[triangle + 1], indices[triangle + 2], true);
    trackBoundaryEdge(edgeMap, indices[triangle + 2], indices[triangle], true);
  }

  const mask = new Uint8Array(vertexCount);
  for (const edge of edgeMap.values()) {
    if (edge.total !== 1) {
      continue;
    }

    mask[edge.a] = 1;
    mask[edge.b] = 1;
  }

  return mask;
}

function buildVertexFaceLists(indices: ArrayLike<number>, vertexCount: number): Uint32Array[] {
  const vertexFaces = Array.from({ length: vertexCount }, () => [] as number[]);
  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    const face = triangle / 3;
    vertexFaces[indices[triangle]].push(face);
    vertexFaces[indices[triangle + 1]].push(face);
    vertexFaces[indices[triangle + 2]].push(face);
  }

  return vertexFaces.map((entry) => Uint32Array.from(entry));
}

function createIdentitySources(count: number): Int32Array {
  const ids = new Int32Array(count);
  for (let i = 0; i < count; i += 1) {
    ids[i] = i;
  }

  return ids;
}

function makeEdgeKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function trackBoundaryEdge(
  edgeMap: Map<string, { a: number; b: number; total: number; selected: number }>,
  a: number,
  b: number,
  selected: boolean,
): void {
  const key = makeEdgeKey(a, b);
  const entry = edgeMap.get(key);
  if (entry) {
    entry.total += 1;
    entry.selected += selected ? 1 : 0;
    return;
  }

  edgeMap.set(key, {
    a,
    b,
    total: 1,
    selected: selected ? 1 : 0,
  });
}

function buildVertexNormals(
  indices: ArrayLike<number>,
  positions: ArrayLike<number>,
  vertexCount: number,
): Float32Array {
  const normals = new Float32Array(vertexCount * 3);
  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    const aOffset = indices[triangle] * 3;
    const bOffset = indices[triangle + 1] * 3;
    const cOffset = indices[triangle + 2] * 3;

    const abx = positions[bOffset] - positions[aOffset];
    const aby = positions[bOffset + 1] - positions[aOffset + 1];
    const abz = positions[bOffset + 2] - positions[aOffset + 2];
    const acx = positions[cOffset] - positions[aOffset];
    const acy = positions[cOffset + 1] - positions[aOffset + 1];
    const acz = positions[cOffset + 2] - positions[aOffset + 2];

    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;

    normals[aOffset] += nx;
    normals[aOffset + 1] += ny;
    normals[aOffset + 2] += nz;
    normals[bOffset] += nx;
    normals[bOffset + 1] += ny;
    normals[bOffset + 2] += nz;
    normals[cOffset] += nx;
    normals[cOffset + 1] += ny;
    normals[cOffset + 2] += nz;
  }

  return normals;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
