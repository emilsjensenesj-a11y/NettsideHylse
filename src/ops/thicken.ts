import { BufferAttribute, BufferGeometry, Vector3 } from 'three';

import type { EditableMeshData } from '../sculpt/editable-mesh';

export interface ThickenResult {
  geometry: BufferGeometry;
}

interface BoundaryEdgeRecord {
  a: number;
  b: number;
  count: number;
}

const positionA = new Vector3();
const positionB = new Vector3();
const positionC = new Vector3();
const edgeDirection = new Vector3();
const averagedNormal = new Vector3();
const targetNormal = new Vector3();
const candidateNormal = new Vector3();

export function thickenMesh(editableMesh: EditableMeshData, thickness: number): ThickenResult {
  if (!Number.isFinite(thickness) || Math.abs(thickness) <= 1e-6) {
    throw new Error('Thickness must be greater than zero.');
  }

  const { positions, normals, indices, vertexCount } = editableMesh;
  const shellOffset = vertexCount;
  const thickenedPositions = new Float32Array(vertexCount * 2 * 3);
  thickenedPositions.set(positions, 0);

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const src = vertex * 3;
    const dst = src + vertexCount * 3;
    thickenedPositions[dst] = positions[src] + normals[src] * thickness;
    thickenedPositions[dst + 1] = positions[src + 1] + normals[src + 1] * thickness;
    thickenedPositions[dst + 2] = positions[src + 2] + normals[src + 2] * thickness;
  }

  const boundaryEdges = collectBoundaryEdges(indices);
  const thickenedIndices: number[] = [];

  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    const a = indices[triangle];
    const b = indices[triangle + 1];
    const c = indices[triangle + 2];

    // Keep the original shell as the inner surface and reverse it, like Blender's solidify shell.
    thickenedIndices.push(a, c, b);

    // The offset shell becomes the outer surface with the original winding.
    thickenedIndices.push(a + shellOffset, b + shellOffset, c + shellOffset);
  }

  for (let i = 0; i < boundaryEdges.length; i += 1) {
    const edge = boundaryEdges[i];
    const a = edge.a;
    const b = edge.b;
    const outerA = a + shellOffset;
    const outerB = b + shellOffset;

    edgeDirection.set(
      positions[b * 3] - positions[a * 3],
      positions[b * 3 + 1] - positions[a * 3 + 1],
      positions[b * 3 + 2] - positions[a * 3 + 2],
    ).normalize();

    averagedNormal.set(
      normals[a * 3] + normals[b * 3],
      normals[a * 3 + 1] + normals[b * 3 + 1],
      normals[a * 3 + 2] + normals[b * 3 + 2],
    );
    if (averagedNormal.lengthSq() <= 1e-12) {
      averagedNormal.set(0, 0, 1);
    } else {
      averagedNormal.normalize();
    }

    targetNormal.crossVectors(edgeDirection, averagedNormal);
    if (thickness < 0) {
      targetNormal.multiplyScalar(-1);
    }
    if (targetNormal.lengthSq() <= 1e-12) {
      targetNormal.copy(averagedNormal);
    } else {
      targetNormal.normalize();
    }

    appendOrientedTriangle(thickenedIndices, thickenedPositions, a, b, outerB, targetNormal);
    appendOrientedTriangle(thickenedIndices, thickenedPositions, a, outerB, outerA, targetNormal);
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(thickenedPositions, 3));
  geometry.setIndex(new BufferAttribute(new Uint32Array(thickenedIndices), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return { geometry };
}

function collectBoundaryEdges(indices: Uint32Array): BoundaryEdgeRecord[] {
  const edgeMap = new Map<string, BoundaryEdgeRecord>();

  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    trackEdge(edgeMap, indices[triangle], indices[triangle + 1]);
    trackEdge(edgeMap, indices[triangle + 1], indices[triangle + 2]);
    trackEdge(edgeMap, indices[triangle + 2], indices[triangle]);
  }

  const boundaryEdges: BoundaryEdgeRecord[] = [];
  for (const edge of edgeMap.values()) {
    if (edge.count === 1) {
      boundaryEdges.push(edge);
    }
  }

  return boundaryEdges;
}

function trackEdge(edgeMap: Map<string, BoundaryEdgeRecord>, a: number, b: number): void {
  const key = a < b ? `${a}:${b}` : `${b}:${a}`;
  const existing = edgeMap.get(key);
  if (existing) {
    existing.count += 1;
    return;
  }

  edgeMap.set(key, {
    a,
    b,
    count: 1,
  });
}

function appendOrientedTriangle(
  target: number[],
  positions: Float32Array,
  a: number,
  b: number,
  c: number,
  referenceNormal: Vector3,
): void {
  setPosition(positionA, positions, a);
  setPosition(positionB, positions, b);
  setPosition(positionC, positions, c);

  candidateNormal
    .subVectors(positionB, positionA)
    .cross(positionC.clone().sub(positionA));

  if (candidateNormal.dot(referenceNormal) < 0) {
    target.push(a, c, b);
  } else {
    target.push(a, b, c);
  }
}

function setPosition(target: Vector3, positions: Float32Array, vertex: number): void {
  const offset = vertex * 3;
  target.set(positions[offset], positions[offset + 1], positions[offset + 2]);
}
