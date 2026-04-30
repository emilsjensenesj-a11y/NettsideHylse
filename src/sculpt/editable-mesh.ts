import {
  BufferAttribute,
  BufferGeometry,
  DynamicDrawUsage,
  type TypedArray,
} from 'three';

export interface EditableMeshData {
  geometry: BufferGeometry;
  positionAttribute: BufferAttribute;
  normalAttribute: BufferAttribute;
  colorAttribute: BufferAttribute;
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  referencePositions: Float32Array;
  indices: Uint32Array;
  faceNormals: Float32Array;
  vertexCount: number;
  triangleCount: number;
  boundsRadius: number;
  vertexFaceOffsets: Uint32Array;
  vertexFaces: Uint32Array;
  vertexNeighborOffsets: Uint32Array;
  vertexNeighbors: Uint32Array;
  triangleNeighbors: Int32Array;
  regionQueue: Uint32Array;
  regionTriangles: Uint32Array;
  regionVertices: Uint32Array;
  regionTriangleMarks: Uint32Array;
  regionVertexMarks: Uint32Array;
  dirtyFaces: Uint32Array;
  dirtyFaceMarks: Uint32Array;
  dirtyVertices: Uint32Array;
  dirtyVertexMarks: Uint32Array;
}

interface EditableMeshOptions {
  referencePositions?: ArrayLike<number>;
}

// Imported mesh units are treated as millimeters for the displacement color ramp.
const DISPLACEMENT_COLOR_LIMIT_MM = 5;
const POSITIVE_DISPLACEMENT_COLOR: [number, number, number] = [0.9, 1.0, 0.38];
const NEGATIVE_DISPLACEMENT_COLOR: [number, number, number] = [0.9, 0.42, 1.0];

export function createEditableMeshData(
  geometry: BufferGeometry,
  options: EditableMeshOptions = {},
): EditableMeshData {
  const positionAttribute = normalizePositionAttribute(geometry);
  const indexAttribute = normalizeIndexAttribute(geometry);
  const positions = positionAttribute.array as Float32Array;
  const indices = indexAttribute.array as Uint32Array;
  const vertexCount = positionAttribute.count;
  const triangleCount = indices.length / 3;
  const referencePositions = normalizeReferencePositions(options.referencePositions, positions);

  const normals = new Float32Array(vertexCount * 3);
  const normalAttribute = new BufferAttribute(normals, 3);
  const colors = new Float32Array(vertexCount * 3);
  const colorAttribute = new BufferAttribute(colors, 3);
  positionAttribute.setUsage(DynamicDrawUsage);
  normalAttribute.setUsage(DynamicDrawUsage);
  colorAttribute.setUsage(DynamicDrawUsage);
  geometry.setAttribute('normal', normalAttribute);
  geometry.setAttribute('color', colorAttribute);

  const { offsets: vertexFaceOffsets, values: vertexFaces } = buildVertexFaceAdjacency(
    indices,
    vertexCount,
    triangleCount,
  );
  const triangleNeighbors = buildTriangleAdjacency(
    indices,
    triangleCount,
    vertexFaceOffsets,
    vertexFaces,
  );
  const { offsets: vertexNeighborOffsets, values: vertexNeighbors } =
    buildVertexNeighborAdjacency(indices, vertexCount, vertexFaceOffsets, vertexFaces);

  const faceNormals = new Float32Array(triangleCount * 3);
  recomputeAllNormals(positions, indices, faceNormals, normals, vertexFaceOffsets, vertexFaces);
  recomputeDisplacementColorsRange(
    positions,
    referencePositions,
    normals,
    colors,
    0,
    vertexCount,
  );

  geometry.computeBoundingSphere();
  const boundsRadius = geometry.boundingSphere?.radius ?? 1;

  return {
    geometry,
    positionAttribute,
    normalAttribute,
    colorAttribute,
    positions,
    normals,
    colors,
    referencePositions,
    indices,
    faceNormals,
    vertexCount,
    triangleCount,
    boundsRadius,
    vertexFaceOffsets,
    vertexFaces,
    vertexNeighborOffsets,
    vertexNeighbors,
    triangleNeighbors,
    regionQueue: new Uint32Array(triangleCount),
    regionTriangles: new Uint32Array(triangleCount),
    regionVertices: new Uint32Array(vertexCount),
    regionTriangleMarks: new Uint32Array(triangleCount),
    regionVertexMarks: new Uint32Array(vertexCount),
    dirtyFaces: new Uint32Array(triangleCount),
    dirtyFaceMarks: new Uint32Array(triangleCount),
    dirtyVertices: new Uint32Array(vertexCount),
    dirtyVertexMarks: new Uint32Array(vertexCount),
  };
}

export function recomputeDisplacementColorsRange(
  positions: Float32Array,
  referencePositions: Float32Array,
  normals: Float32Array,
  colors: Float32Array,
  startVertex: number,
  vertexCount: number,
): void {
  for (let vertex = startVertex; vertex < startVertex + vertexCount; vertex += 1) {
    writeDisplacementColor(positions, referencePositions, normals, colors, vertex);
  }
}

export function recomputeDisplacementColorsForVertices(
  positions: Float32Array,
  referencePositions: Float32Array,
  normals: Float32Array,
  colors: Float32Array,
  vertexIds: Uint32Array,
  count: number,
): void {
  for (let i = 0; i < count; i += 1) {
    writeDisplacementColor(positions, referencePositions, normals, colors, vertexIds[i]);
  }
}

export function recomputeAllNormals(
  positions: Float32Array,
  indices: Uint32Array,
  faceNormals: Float32Array,
  normals: Float32Array,
  vertexFaceOffsets: Uint32Array,
  vertexFaces: Uint32Array,
): void {
  recomputeFaceNormalsRange(positions, indices, faceNormals, 0, indices.length / 3);
  recomputeVertexNormalsRange(normals, faceNormals, vertexFaceOffsets, vertexFaces, 0, normals.length / 3);
}

export function recomputeFaceNormalsRange(
  positions: Float32Array,
  indices: Uint32Array,
  faceNormals: Float32Array,
  startTriangle: number,
  triangleCount: number,
): void {
  for (let triangle = startTriangle; triangle < startTriangle + triangleCount; triangle += 1) {
    const triOffset = triangle * 3;
    const ia = indices[triOffset] * 3;
    const ib = indices[triOffset + 1] * 3;
    const ic = indices[triOffset + 2] * 3;

    const abx = positions[ib] - positions[ia];
    const aby = positions[ib + 1] - positions[ia + 1];
    const abz = positions[ib + 2] - positions[ia + 2];
    const acx = positions[ic] - positions[ia];
    const acy = positions[ic + 1] - positions[ia + 1];
    const acz = positions[ic + 2] - positions[ia + 2];

    faceNormals[triOffset] = aby * acz - abz * acy;
    faceNormals[triOffset + 1] = abz * acx - abx * acz;
    faceNormals[triOffset + 2] = abx * acy - aby * acx;
  }
}

export function recomputeVertexNormalsRange(
  normals: Float32Array,
  faceNormals: Float32Array,
  vertexFaceOffsets: Uint32Array,
  vertexFaces: Uint32Array,
  startVertex: number,
  vertexCount: number,
): void {
  for (let vertex = startVertex; vertex < startVertex + vertexCount; vertex += 1) {
    let nx = 0;
    let ny = 0;
    let nz = 0;

    for (
      let faceOffset = vertexFaceOffsets[vertex];
      faceOffset < vertexFaceOffsets[vertex + 1];
      faceOffset += 1
    ) {
      const face = vertexFaces[faceOffset] * 3;
      nx += faceNormals[face];
      ny += faceNormals[face + 1];
      nz += faceNormals[face + 2];
    }

    const normalOffset = vertex * 3;
    const length = Math.hypot(nx, ny, nz);
    if (length > 1e-12) {
      normals[normalOffset] = nx / length;
      normals[normalOffset + 1] = ny / length;
      normals[normalOffset + 2] = nz / length;
    } else {
      normals[normalOffset] = 0;
      normals[normalOffset + 1] = 0;
      normals[normalOffset + 2] = 1;
    }
  }
}

function normalizePositionAttribute(geometry: BufferGeometry): BufferAttribute {
  const attribute = geometry.getAttribute('position');
  if (!attribute) {
    throw new Error('The mesh is missing a position attribute.');
  }

  if (attribute.itemSize !== 3) {
    throw new Error('The mesh position attribute must contain XYZ vertices.');
  }

  if (!(attribute instanceof BufferAttribute) || !(attribute.array instanceof Float32Array)) {
    const array = copyIntoTypedArray(Float32Array, attribute.array as ArrayLike<number>);
    const floatAttribute = new BufferAttribute(array, 3);
    geometry.setAttribute('position', floatAttribute);
    return floatAttribute;
  }

  return attribute as BufferAttribute;
}

function normalizeIndexAttribute(geometry: BufferGeometry): BufferAttribute {
  const index = geometry.getIndex();
  if (!index) {
    throw new Error('The mesh is not indexed.');
  }

  if (!(index instanceof BufferAttribute) || !(index.array instanceof Uint32Array)) {
    const array = copyIntoTypedArray(Uint32Array, index.array as ArrayLike<number>);
    const uintAttribute = new BufferAttribute(array, 1);
    geometry.setIndex(uintAttribute);
    return uintAttribute;
  }

  return index as BufferAttribute;
}

function normalizeReferencePositions(
  source: ArrayLike<number> | undefined,
  positions: Float32Array,
): Float32Array {
  if (!source || source.length !== positions.length) {
    return positions.slice();
  }

  return copyIntoTypedArray(Float32Array, source);
}

function writeDisplacementColor(
  positions: Float32Array,
  referencePositions: Float32Array,
  normals: Float32Array,
  colors: Float32Array,
  vertex: number,
): void {
  const offset = vertex * 3;
  const dx = positions[offset] - referencePositions[offset];
  const dy = positions[offset + 1] - referencePositions[offset + 1];
  const dz = positions[offset + 2] - referencePositions[offset + 2];
  const nx = normals[offset];
  const ny = normals[offset + 1];
  const nz = normals[offset + 2];
  const normalLength = Math.hypot(nx, ny, nz);

  let colorMix = 0;
  let targetColor = POSITIVE_DISPLACEMENT_COLOR;
  if (normalLength > 1e-12) {
    const signedDisplacement = (dx * nx + dy * ny + dz * nz) / normalLength;
    const normalizedDisplacement = Math.min(
      Math.max(signedDisplacement / DISPLACEMENT_COLOR_LIMIT_MM, -1),
      1,
    );
    colorMix = smoothColorMix(Math.abs(normalizedDisplacement));
    targetColor =
      normalizedDisplacement >= 0 ? POSITIVE_DISPLACEMENT_COLOR : NEGATIVE_DISPLACEMENT_COLOR;
  }

  colors[offset] = 1 + (targetColor[0] - 1) * colorMix;
  colors[offset + 1] = 1 + (targetColor[1] - 1) * colorMix;
  colors[offset + 2] = 1 + (targetColor[2] - 1) * colorMix;
}

function smoothColorMix(value: number): number {
  const x = Math.min(Math.max(value, 0), 1);
  return x * x * (3 - 2 * x);
}

function buildVertexFaceAdjacency(
  indices: Uint32Array,
  vertexCount: number,
  triangleCount: number,
): { offsets: Uint32Array; values: Uint32Array } {
  const counts = new Uint32Array(vertexCount);
  for (let i = 0; i < indices.length; i += 1) {
    counts[indices[i]] += 1;
  }

  const offsets = new Uint32Array(vertexCount + 1);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    offsets[vertex + 1] = offsets[vertex] + counts[vertex];
  }

  const values = new Uint32Array(indices.length);
  const cursor = offsets.slice(0);

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const triOffset = triangle * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = indices[triOffset + corner];
      values[cursor[vertex]] = triangle;
      cursor[vertex] += 1;
    }
  }

  return { offsets, values };
}

function buildVertexNeighborAdjacency(
  indices: Uint32Array,
  vertexCount: number,
  vertexFaceOffsets: Uint32Array,
  vertexFaces: Uint32Array,
): { offsets: Uint32Array; values: Uint32Array } {
  const counts = new Uint32Array(vertexCount);
  const marks = new Uint32Array(vertexCount);
  let stamp = 1;

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    stamp = nextStamp(marks, stamp);
    for (
      let faceOffset = vertexFaceOffsets[vertex];
      faceOffset < vertexFaceOffsets[vertex + 1];
      faceOffset += 1
    ) {
      const triangle = vertexFaces[faceOffset] * 3;
      for (let corner = 0; corner < 3; corner += 1) {
        const neighbor = indices[triangle + corner];
        if (neighbor === vertex || marks[neighbor] === stamp) {
          continue;
        }

        marks[neighbor] = stamp;
        counts[vertex] += 1;
      }
    }
  }

  const offsets = new Uint32Array(vertexCount + 1);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    offsets[vertex + 1] = offsets[vertex] + counts[vertex];
  }

  const values = new Uint32Array(offsets[offsets.length - 1]);
  const cursor = offsets.slice(0);
  marks.fill(0);
  stamp = 1;

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    stamp = nextStamp(marks, stamp);
    for (
      let faceOffset = vertexFaceOffsets[vertex];
      faceOffset < vertexFaceOffsets[vertex + 1];
      faceOffset += 1
    ) {
      const triangle = vertexFaces[faceOffset] * 3;
      for (let corner = 0; corner < 3; corner += 1) {
        const neighbor = indices[triangle + corner];
        if (neighbor === vertex || marks[neighbor] === stamp) {
          continue;
        }

        marks[neighbor] = stamp;
        values[cursor[vertex]] = neighbor;
        cursor[vertex] += 1;
      }
    }
  }

  return { offsets, values };
}

function buildTriangleAdjacency(
  indices: Uint32Array,
  triangleCount: number,
  vertexFaceOffsets: Uint32Array,
  vertexFaces: Uint32Array,
): Int32Array {
  const neighbors = new Int32Array(triangleCount * 3);
  neighbors.fill(-1);

  const marks = new Uint32Array(triangleCount);
  let stamp = 1;

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const triOffset = triangle * 3;

    for (let edge = 0; edge < 3; edge += 1) {
      const neighborSlot = triOffset + edge;
      if (neighbors[neighborSlot] !== -1) {
        continue;
      }

      const a = indices[triOffset + edge];
      const b = indices[triOffset + ((edge + 1) % 3)];
      stamp = nextStamp(marks, stamp);

      for (
        let faceOffset = vertexFaceOffsets[a];
        faceOffset < vertexFaceOffsets[a + 1];
        faceOffset += 1
      ) {
        marks[vertexFaces[faceOffset]] = stamp;
      }

      let match = -1;
      for (
        let faceOffset = vertexFaceOffsets[b];
        faceOffset < vertexFaceOffsets[b + 1];
        faceOffset += 1
      ) {
        const face = vertexFaces[faceOffset];
        if (face !== triangle && marks[face] === stamp) {
          match = face;
          break;
        }
      }

      if (match === -1) {
        continue;
      }

      neighbors[neighborSlot] = match;
      const matchOffset = match * 3;
      for (let matchEdge = 0; matchEdge < 3; matchEdge += 1) {
        const ma = indices[matchOffset + matchEdge];
        const mb = indices[matchOffset + ((matchEdge + 1) % 3)];
        if ((ma === a && mb === b) || (ma === b && mb === a)) {
          neighbors[matchOffset + matchEdge] = triangle;
          break;
        }
      }
    }
  }

  return neighbors;
}

function copyIntoTypedArray<T extends TypedArray>(
  TypedArrayConstructor: new (length: number) => T,
  source: ArrayLike<number>,
): T {
  const array = new TypedArrayConstructor(source.length);
  for (let i = 0; i < source.length; i += 1) {
    array[i] = source[i];
  }

  return array;
}

function nextStamp(marks: Uint32Array, current: number): number {
  const next = current + 1;
  if (next !== 0) {
    return next;
  }

  marks.fill(0);
  return 1;
}
