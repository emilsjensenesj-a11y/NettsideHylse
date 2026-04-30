import { BufferAttribute, BufferGeometry, Vector3 } from 'three';

import type { EditableMeshData } from '../sculpt/editable-mesh';

export interface BoundaryExtrudeResult {
  geometry: BufferGeometry;
  outerVertexIds: Uint32Array;
}

export interface BoundaryExtrudeDirection {
  x: number;
  y: number;
  z: number;
}

const positionA = new Vector3();
const positionB = new Vector3();
const positionC = new Vector3();
const edgeDirection = new Vector3();
const averagedNormal = new Vector3();
const targetNormal = new Vector3();
const candidateNormal = new Vector3();

export function extrudeBoundaryLoop(
  editableMesh: EditableMeshData,
  orderedVertexIds: Uint32Array,
  distance: number,
): BoundaryExtrudeResult {
  if (!Number.isFinite(distance) || Math.abs(distance) <= 1e-6) {
    throw new Error('Extrude distance must be greater than zero.');
  }

  if (orderedVertexIds.length < 3) {
    throw new Error('A positive socket needs a clean boundary loop with at least 3 vertices.');
  }

  const { positions, normals, indices, vertexCount } = editableMesh;
  const coherentNormals = computeCoherentBoundaryNormals(normals, orderedVertexIds);
  const outerVertexIds = new Uint32Array(orderedVertexIds.length);
  const extrudedPositions = new Float32Array((vertexCount + orderedVertexIds.length) * 3);
  extrudedPositions.set(positions, 0);

  for (let i = 0; i < orderedVertexIds.length; i += 1) {
    const sourceVertex = orderedVertexIds[i];
    const outerVertex = vertexCount + i;
    outerVertexIds[i] = outerVertex;

    const sourceOffset = sourceVertex * 3;
    const outerOffset = outerVertex * 3;
    const normalOffset = i * 3;
    extrudedPositions[outerOffset] = positions[sourceOffset] + coherentNormals[normalOffset] * distance;
    extrudedPositions[outerOffset + 1] = positions[sourceOffset + 1] + coherentNormals[normalOffset + 1] * distance;
    extrudedPositions[outerOffset + 2] = positions[sourceOffset + 2] + coherentNormals[normalOffset + 2] * distance;
  }

  const extrudedIndices = Array.from(indices);
  for (let i = 0; i < orderedVertexIds.length; i += 1) {
    const next = (i + 1) % orderedVertexIds.length;
    const a = orderedVertexIds[i];
    const b = orderedVertexIds[next];
    const outerA = outerVertexIds[i];
    const outerB = outerVertexIds[next];

    edgeDirection.set(
      positions[b * 3] - positions[a * 3],
      positions[b * 3 + 1] - positions[a * 3 + 1],
      positions[b * 3 + 2] - positions[a * 3 + 2],
    ).normalize();

    averagedNormal.set(
      coherentNormals[i * 3] + coherentNormals[next * 3],
      coherentNormals[i * 3 + 1] + coherentNormals[next * 3 + 1],
      coherentNormals[i * 3 + 2] + coherentNormals[next * 3 + 2],
    );
    if (averagedNormal.lengthSq() <= 1e-12) {
      averagedNormal.set(0, 0, 1);
    } else {
      averagedNormal.normalize();
    }

    targetNormal.crossVectors(edgeDirection, averagedNormal);
    if (distance < 0) {
      targetNormal.multiplyScalar(-1);
    }
    if (targetNormal.lengthSq() <= 1e-12) {
      targetNormal.copy(averagedNormal);
    } else {
      targetNormal.normalize();
    }

    appendOrientedTriangle(extrudedIndices, extrudedPositions, a, b, outerB, targetNormal);
    appendOrientedTriangle(extrudedIndices, extrudedPositions, a, outerB, outerA, targetNormal);
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(extrudedPositions, 3));
  geometry.setIndex(new BufferAttribute(new Uint32Array(extrudedIndices), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return {
    geometry,
    outerVertexIds,
  };
}

export function extrudeBoundaryLoopAlongVector(
  editableMesh: EditableMeshData,
  orderedVertexIds: Uint32Array,
  direction: BoundaryExtrudeDirection,
  distance: number,
): BoundaryExtrudeResult {
  if (!Number.isFinite(distance) || Math.abs(distance) <= 1e-6) {
    throw new Error('Extrude distance must be greater than zero.');
  }

  if (orderedVertexIds.length < 3) {
    throw new Error('A directional boundary extrude needs a clean boundary loop with at least 3 vertices.');
  }

  targetNormal.set(direction.x, direction.y, direction.z);
  if (targetNormal.lengthSq() <= 1e-12) {
    throw new Error('Directional boundary extrude needs a valid tangent direction.');
  }
  targetNormal.normalize();

  const { positions, indices, vertexCount } = editableMesh;
  const outerVertexIds = new Uint32Array(orderedVertexIds.length);
  const extrudedPositions = new Float32Array((vertexCount + orderedVertexIds.length) * 3);
  extrudedPositions.set(positions, 0);

  for (let i = 0; i < orderedVertexIds.length; i += 1) {
    const sourceVertex = orderedVertexIds[i];
    const outerVertex = vertexCount + i;
    outerVertexIds[i] = outerVertex;

    const sourceOffset = sourceVertex * 3;
    const outerOffset = outerVertex * 3;
    extrudedPositions[outerOffset] = positions[sourceOffset] + targetNormal.x * distance;
    extrudedPositions[outerOffset + 1] = positions[sourceOffset + 1] + targetNormal.y * distance;
    extrudedPositions[outerOffset + 2] = positions[sourceOffset + 2] + targetNormal.z * distance;
  }

  const extrudedIndices = Array.from(indices);
  for (let i = 0; i < orderedVertexIds.length; i += 1) {
    const next = (i + 1) % orderedVertexIds.length;
    const a = orderedVertexIds[i];
    const b = orderedVertexIds[next];
    const outerA = outerVertexIds[i];
    const outerB = outerVertexIds[next];

    edgeDirection.set(
      positions[b * 3] - positions[a * 3],
      positions[b * 3 + 1] - positions[a * 3 + 1],
      positions[b * 3 + 2] - positions[a * 3 + 2],
    ).normalize();

    averagedNormal.crossVectors(edgeDirection, targetNormal);
    if (averagedNormal.lengthSq() <= 1e-12) {
      averagedNormal.set(0, 0, 1);
    } else {
      averagedNormal.normalize();
    }

    appendOrientedTriangle(extrudedIndices, extrudedPositions, a, b, outerB, averagedNormal);
    appendOrientedTriangle(extrudedIndices, extrudedPositions, a, outerB, outerA, averagedNormal);
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(extrudedPositions, 3));
  geometry.setIndex(new BufferAttribute(new Uint32Array(extrudedIndices), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return {
    geometry,
    outerVertexIds,
  };
}

export function computeCoherentBoundaryNormals(
  normals: Float32Array,
  orderedVertexIds: Uint32Array,
): Float32Array {
  const count = orderedVertexIds.length;
  const coherent = new Float32Array(count * 3);
  if (count === 0) {
    return coherent;
  }

  let seedIndex = 0;
  for (let i = 0; i < count; i += 1) {
    const offset = orderedVertexIds[i] * 3;
    if (
      normals[offset] * normals[offset] +
        normals[offset + 1] * normals[offset + 1] +
        normals[offset + 2] * normals[offset + 2] >
      1e-10
    ) {
      seedIndex = i;
      break;
    }
  }

  const seedOffset = orderedVertexIds[seedIndex] * 3;
  writeNormalized(
    coherent,
    seedIndex * 3,
    normals[seedOffset],
    normals[seedOffset + 1],
    normals[seedOffset + 2],
  );

  for (let step = 1; step < count; step += 1) {
    const i = (seedIndex + step) % count;
    const previous = (i - 1 + count) % count;
    const sourceOffset = orderedVertexIds[i] * 3;
    const previousOffset = previous * 3;
    let nx = normals[sourceOffset];
    let ny = normals[sourceOffset + 1];
    let nz = normals[sourceOffset + 2];
    if (nx * nx + ny * ny + nz * nz <= 1e-10) {
      nx = coherent[previousOffset];
      ny = coherent[previousOffset + 1];
      nz = coherent[previousOffset + 2];
    }

    const dot =
      nx * coherent[previousOffset] +
      ny * coherent[previousOffset + 1] +
      nz * coherent[previousOffset + 2];
    if (dot < 0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }

    writeNormalized(coherent, i * 3, nx, ny, nz);
  }

  const smoothed = new Float32Array(coherent.length);
  let current = coherent;
  let next = smoothed;
  for (let iteration = 0; iteration < 2; iteration += 1) {
    for (let i = 0; i < count; i += 1) {
      const previous = (i - 1 + count) % count;
      const following = (i + 1) % count;
      const currentOffset = i * 3;
      const previousOffset = previous * 3;
      const followingOffset = following * 3;

      let prevX = current[previousOffset];
      let prevY = current[previousOffset + 1];
      let prevZ = current[previousOffset + 2];
      if (
        prevX * current[currentOffset] +
          prevY * current[currentOffset + 1] +
          prevZ * current[currentOffset + 2] <
        0
      ) {
        prevX = -prevX;
        prevY = -prevY;
        prevZ = -prevZ;
      }

      let nextX = current[followingOffset];
      let nextY = current[followingOffset + 1];
      let nextZ = current[followingOffset + 2];
      if (
        nextX * current[currentOffset] +
          nextY * current[currentOffset + 1] +
          nextZ * current[currentOffset + 2] <
        0
      ) {
        nextX = -nextX;
        nextY = -nextY;
        nextZ = -nextZ;
      }

      writeNormalized(
        next,
        currentOffset,
        current[currentOffset] * 2 + prevX + nextX,
        current[currentOffset + 1] * 2 + prevY + nextY,
        current[currentOffset + 2] * 2 + prevZ + nextZ,
      );
    }

    const swap = current;
    current = next;
    next = swap;
  }

  return current;
}

function writeNormalized(
  target: Float32Array,
  offset: number,
  x: number,
  y: number,
  z: number,
): void {
  const length = Math.hypot(x, y, z);
  if (length <= 1e-10) {
    target[offset] = 0;
    target[offset + 1] = 0;
    target[offset + 2] = 1;
    return;
  }

  const invLength = 1 / length;
  target[offset] = x * invLength;
  target[offset + 1] = y * invLength;
  target[offset + 2] = z * invLength;
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
