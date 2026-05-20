export interface HoleLoop {
  segmentVertexPairs: Uint32Array;
  edgeCount: number;
  boundaryEdgeCount: number;
  orderedVertexIds: Uint32Array | null;
  isBoundaryLoop: boolean;
}

export interface HoleLoopSet {
  loops: HoleLoop[];
  edgeCount: number;
}

interface EdgeRecord {
  a: number;
  b: number;
  count: number;
}

export function buildHoleLoopSet(indices: Uint32Array, positions?: Float32Array): HoleLoopSet {
  const edgeIndexByKey = new Map<string, number>();
  const edgeRecords: EdgeRecord[] = [];
  const canonicalByVertex = positions ? buildCanonicalVerticesByPosition(positions) : null;

  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    recordEdge(indices[triangle], indices[triangle + 1], edgeIndexByKey, edgeRecords, canonicalByVertex);
    recordEdge(indices[triangle + 1], indices[triangle + 2], edgeIndexByKey, edgeRecords, canonicalByVertex);
    recordEdge(indices[triangle + 2], indices[triangle], edgeIndexByKey, edgeRecords, canonicalByVertex);
  }

  const specialEdges: EdgeRecord[] = [];
  const vertexToSpecialEdges = new Map<number, number[]>();

  for (let i = 0; i < edgeRecords.length; i += 1) {
    const edge = edgeRecords[i];
    if (edge.count === 2) {
      continue;
    }

    const specialIndex = specialEdges.length;
    specialEdges.push(edge);
    addVertexEdgeLink(vertexToSpecialEdges, edge.a, specialIndex);
    addVertexEdgeLink(vertexToSpecialEdges, edge.b, specialIndex);
  }

  if (specialEdges.length === 0) {
    return {
      loops: [],
      edgeCount: 0,
    };
  }

  const visited = new Uint8Array(specialEdges.length);
  const edgeStack = new Uint32Array(specialEdges.length);
  const loops: HoleLoop[] = [];

  for (let start = 0; start < specialEdges.length; start += 1) {
    if (visited[start] !== 0) {
      continue;
    }

    let stackSize = 0;
    edgeStack[stackSize] = start;
    stackSize += 1;
    visited[start] = 1;

    const componentPairs: number[] = [];
    let boundaryEdgeCount = 0;

    while (stackSize > 0) {
      stackSize -= 1;
      const edgeIndex = edgeStack[stackSize];
      const edge = specialEdges[edgeIndex];
      componentPairs.push(edge.a, edge.b);
      if (edge.count === 1) {
        boundaryEdgeCount += 1;
      }

      const aConnections = vertexToSpecialEdges.get(edge.a);
      if (aConnections) {
        for (let i = 0; i < aConnections.length; i += 1) {
          const neighborEdgeIndex = aConnections[i];
          if (visited[neighborEdgeIndex] !== 0) {
            continue;
          }

          visited[neighborEdgeIndex] = 1;
          edgeStack[stackSize] = neighborEdgeIndex;
          stackSize += 1;
        }
      }

      const bConnections = vertexToSpecialEdges.get(edge.b);
      if (bConnections) {
        for (let i = 0; i < bConnections.length; i += 1) {
          const neighborEdgeIndex = bConnections[i];
          if (visited[neighborEdgeIndex] !== 0) {
            continue;
          }

          visited[neighborEdgeIndex] = 1;
          edgeStack[stackSize] = neighborEdgeIndex;
          stackSize += 1;
        }
      }
    }

    const orderedVertexIds =
      boundaryEdgeCount === componentPairs.length / 2
        ? orderBoundaryComponent(componentPairs)
        : null;

    loops.push({
      segmentVertexPairs: new Uint32Array(componentPairs),
      edgeCount: componentPairs.length / 2,
      boundaryEdgeCount,
      orderedVertexIds,
      isBoundaryLoop: orderedVertexIds !== null,
    });
  }

  const visibleBoundaryLoops = loops
    .filter((loop) => loop.edgeCount >= 2 && (loop.edgeCount > 10 || loop.isBoundaryLoop))
    .sort((a, b) => b.edgeCount - a.edgeCount);

  return {
    loops: visibleBoundaryLoops,
    edgeCount: visibleBoundaryLoops.reduce((sum, loop) => sum + loop.edgeCount, 0),
  };
}

export function buildOpenBoundaryLoopCandidates(indices: Uint32Array, positions?: Float32Array): HoleLoop[] {
  const edgeIndexByKey = new Map<string, number>();
  const edgeRecords: EdgeRecord[] = [];
  const canonicalByVertex = positions ? buildCanonicalVerticesByPosition(positions) : null;

  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    recordEdge(indices[triangle], indices[triangle + 1], edgeIndexByKey, edgeRecords, canonicalByVertex);
    recordEdge(indices[triangle + 1], indices[triangle + 2], edgeIndexByKey, edgeRecords, canonicalByVertex);
    recordEdge(indices[triangle + 2], indices[triangle], edgeIndexByKey, edgeRecords, canonicalByVertex);
  }

  const boundaryEdges = edgeRecords.filter((edge) => edge.count === 1);
  if (boundaryEdges.length === 0) {
    return [];
  }

  const vertexToBoundaryEdges = new Map<number, number[]>();
  for (let i = 0; i < boundaryEdges.length; i += 1) {
    addVertexEdgeLink(vertexToBoundaryEdges, boundaryEdges[i].a, i);
    addVertexEdgeLink(vertexToBoundaryEdges, boundaryEdges[i].b, i);
  }

  const visited = new Uint8Array(boundaryEdges.length);
  const edgeStack = new Uint32Array(boundaryEdges.length);
  const loops: HoleLoop[] = [];

  for (let start = 0; start < boundaryEdges.length; start += 1) {
    if (visited[start] !== 0) {
      continue;
    }

    let stackSize = 0;
    edgeStack[stackSize] = start;
    stackSize += 1;
    visited[start] = 1;

    const componentPairs: number[] = [];
    while (stackSize > 0) {
      stackSize -= 1;
      const edgeIndex = edgeStack[stackSize];
      const edge = boundaryEdges[edgeIndex];
      componentPairs.push(edge.a, edge.b);

      const aConnections = vertexToBoundaryEdges.get(edge.a);
      if (aConnections) {
        for (let i = 0; i < aConnections.length; i += 1) {
          const neighborEdgeIndex = aConnections[i];
          if (visited[neighborEdgeIndex] !== 0) {
            continue;
          }
          visited[neighborEdgeIndex] = 1;
          edgeStack[stackSize] = neighborEdgeIndex;
          stackSize += 1;
        }
      }

      const bConnections = vertexToBoundaryEdges.get(edge.b);
      if (bConnections) {
        for (let i = 0; i < bConnections.length; i += 1) {
          const neighborEdgeIndex = bConnections[i];
          if (visited[neighborEdgeIndex] !== 0) {
            continue;
          }
          visited[neighborEdgeIndex] = 1;
          edgeStack[stackSize] = neighborEdgeIndex;
          stackSize += 1;
        }
      }
    }

    const orderedVertexIds = orderBoundaryComponent(componentPairs);
    loops.push({
      segmentVertexPairs: new Uint32Array(componentPairs),
      edgeCount: componentPairs.length / 2,
      boundaryEdgeCount: componentPairs.length / 2,
      orderedVertexIds,
      isBoundaryLoop: orderedVertexIds !== null,
    });
  }

  return loops.sort((a, b) => b.edgeCount - a.edgeCount);
}

export function countNonManifoldEdges(indices: Uint32Array, positions?: Float32Array): number {
  const edgeIndexByKey = new Map<string, number>();
  const edgeRecords: EdgeRecord[] = [];
  const canonicalByVertex = positions ? buildCanonicalVerticesByPosition(positions) : null;

  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    recordEdge(indices[triangle], indices[triangle + 1], edgeIndexByKey, edgeRecords, canonicalByVertex);
    recordEdge(indices[triangle + 1], indices[triangle + 2], edgeIndexByKey, edgeRecords, canonicalByVertex);
    recordEdge(indices[triangle + 2], indices[triangle], edgeIndexByKey, edgeRecords, canonicalByVertex);
  }

  let edgeCount = 0;
  for (let i = 0; i < edgeRecords.length; i += 1) {
    if (edgeRecords[i].count !== 2) {
      edgeCount += 1;
    }
  }

  return edgeCount;
}

function recordEdge(
  a: number,
  b: number,
  edgeIndexByKey: Map<string, number>,
  edgeRecords: EdgeRecord[],
  canonicalByVertex: Uint32Array | null,
): void {
  if (a === b) {
    return;
  }

  const canonicalA = canonicalByVertex?.[a] ?? a;
  const canonicalB = canonicalByVertex?.[b] ?? b;
  if (canonicalA === canonicalB) {
    return;
  }

  const low = Math.min(canonicalA, canonicalB);
  const high = Math.max(canonicalA, canonicalB);
  const key = `${low}:${high}`;
  const existingIndex = edgeIndexByKey.get(key);

  if (existingIndex == null) {
    edgeIndexByKey.set(key, edgeRecords.length);
    edgeRecords.push({
      a: low,
      b: high,
      count: 1,
    });
    return;
  }

  edgeRecords[existingIndex].count += 1;
}

function buildCanonicalVerticesByPosition(positions: Float32Array): Uint32Array {
  const vertexCount = positions.length / 3;
  const canonicalByVertex = new Uint32Array(vertexCount);
  const canonicalByPosition = new Map<string, number>();

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const offset = vertex * 3;
    const key = `${positions[offset].toFixed(5)},${positions[offset + 1].toFixed(5)},${positions[offset + 2].toFixed(5)}`;
    const existing = canonicalByPosition.get(key);
    if (existing !== undefined) {
      canonicalByVertex[vertex] = existing;
      continue;
    }

    canonicalByPosition.set(key, vertex);
    canonicalByVertex[vertex] = vertex;
  }

  return canonicalByVertex;
}

function addVertexEdgeLink(
  vertexToSpecialEdges: Map<number, number[]>,
  vertex: number,
  edgeIndex: number,
): void {
  const edges = vertexToSpecialEdges.get(vertex);
  if (edges) {
    edges.push(edgeIndex);
    return;
  }

  vertexToSpecialEdges.set(vertex, [edgeIndex]);
}

function orderBoundaryComponent(componentPairs: number[]): Uint32Array | null {
  const neighbors = new Map<number, number[]>();
  for (let i = 0; i < componentPairs.length; i += 2) {
    const a = componentPairs[i];
    const b = componentPairs[i + 1];
    pushNeighbor(neighbors, a, b);
    pushNeighbor(neighbors, b, a);
  }

  for (const degree of neighbors.values()) {
    if (degree.length !== 2) {
      return null;
    }
  }

  const vertices = Array.from(neighbors.keys()).sort((a, b) => a - b);
  if (vertices.length === 0) {
    return null;
  }

  const start = vertices[0];
  let previous = -1;
  let current = start;
  const ordered: number[] = [];

  for (let guard = 0; guard < componentPairs.length + 2; guard += 1) {
    ordered.push(current);
    const linked = neighbors.get(current);
    if (!linked || linked.length !== 2) {
      return null;
    }

    const next = linked[0] === previous ? linked[1] : linked[0];
    previous = current;
    current = next;
    if (current === start) {
      return new Uint32Array(ordered);
    }
  }

  return null;
}

function pushNeighbor(neighbors: Map<number, number[]>, from: number, to: number): void {
  const linked = neighbors.get(from);
  if (linked) {
    linked.push(to);
    return;
  }

  neighbors.set(from, [to]);
}
