import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  Group,
  Mesh,
  Vector3,
} from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import {
  mergeGeometries,
  mergeVertices,
} from 'three/examples/jsm/utils/BufferGeometryUtils.js';

const box = new Box3();
const center = new Vector3();

export type MeshUnit = 'mm' | 'cm' | 'm';

export interface LoadedMeshAsset {
  geometry: BufferGeometry;
  filename: string;
  extension: string;
  vertexCount: number;
  triangleCount: number;
  boundsRadius: number;
  importUnit: MeshUnit;
}

export async function loadMeshFile(file: File, importUnit: MeshUnit = 'mm'): Promise<LoadedMeshAsset> {
  const extension = getFileExtension(file.name);

  let parsedGeometry: BufferGeometry;
  if (extension === 'obj') {
    parsedGeometry = normalizeGeometry(mergeObjectMeshes(await parseObj(file)), importUnit);
  } else if (extension === 'stl') {
    const buffer = await file.arrayBuffer();
    parsedGeometry = normalizeGeometry(new STLLoader().parse(buffer), importUnit);
  } else if (extension === 'ply') {
    const buffer = await file.arrayBuffer();
    parsedGeometry = normalizeGeometry(new PLYLoader().parse(buffer), importUnit);
  } else {
    throw new Error('Unsupported file type. Choose an STL, OBJ, or PLY mesh.');
  }

  const position = parsedGeometry.getAttribute('position');
  const triangleCount = parsedGeometry.getIndex()!.count / 3;
  parsedGeometry.computeBoundingSphere();
  const boundsRadius = parsedGeometry.boundingSphere?.radius ?? 1;

  return {
    geometry: parsedGeometry,
    filename: file.name,
    extension,
    vertexCount: position.count,
    triangleCount,
    boundsRadius,
    importUnit,
  };
}

function getFileExtension(filename: string): string {
  const dotIndex = filename.lastIndexOf('.');
  return dotIndex >= 0 ? filename.slice(dotIndex + 1).toLowerCase() : '';
}

async function parseObj(file: File): Promise<Group> {
  const text = await file.text();
  return new OBJLoader().parse(text);
}

function mergeObjectMeshes(root: Group): BufferGeometry {
  const geometries: BufferGeometry[] = [];
  root.updateMatrixWorld(true);

  root.traverse((child) => {
    if (!(child instanceof Mesh)) {
      return;
    }

    const childGeometry = child.geometry?.clone();
    if (!childGeometry) {
      return;
    }

    childGeometry.applyMatrix4(child.matrixWorld);
    geometries.push(childGeometry);
  });

  if (geometries.length === 0) {
    throw new Error('No mesh data was found in the OBJ file.');
  }

  if (geometries.length === 1) {
    return geometries[0];
  }

  const merged = mergeGeometries(geometries, false);
  if (!merged) {
    throw new Error('The OBJ mesh parts could not be merged into one editable mesh.');
  }

  geometries.forEach((geometry) => geometry.dispose());
  return merged;
}

function normalizeGeometry(source: BufferGeometry, importUnit: MeshUnit): BufferGeometry {
  const positionAttr = source.getAttribute('position');
  if (!positionAttr || positionAttr.count < 3) {
    throw new Error('The selected file does not contain a valid triangle mesh.');
  }

  const geometry = new BufferGeometry();
  const sourcePositions = positionAttr.array as ArrayLike<number>;
  const positions = new Float32Array(sourcePositions.length);
  positions.set(sourcePositions);
  geometry.setAttribute('position', new BufferAttribute(positions, 3));

  const index = source.getIndex();
  if (index) {
    const srcIndex = index.array as ArrayLike<number>;
    const normalizedIndex = new Uint32Array(srcIndex.length);
    for (let i = 0; i < srcIndex.length; i += 1) {
      normalizedIndex[i] = srcIndex[i];
    }

    geometry.setIndex(new BufferAttribute(normalizedIndex, 1));
  } else {
    geometry.setIndex(new BufferAttribute(createSequentialIndex(positionAttr.count), 1));
  }

  const importScale = getMillimeterScale(importUnit);
  if (importScale !== 1) {
    geometry.scale(importScale, importScale, importScale);
  }

  geometry.computeBoundingBox();
  if (!geometry.boundingBox) {
    throw new Error('Failed to compute mesh bounds.');
  }

  box.copy(geometry.boundingBox);
  box.getCenter(center);
  geometry.translate(-center.x, -center.y, -center.z);

  box.setFromBufferAttribute(geometry.getAttribute('position') as BufferAttribute);
  const diagonal = box.min.distanceTo(box.max);
  const weldTolerance = Math.max(diagonal * 1e-6, 1e-7);
  const welded = mergeVertices(geometry, weldTolerance);

  if (!welded.getIndex()) {
    welded.setIndex(new BufferAttribute(createSequentialIndex(welded.getAttribute('position').count), 1));
  }

  const weldedIndex = welded.getIndex()!;
  const weldedAttr = welded.getAttribute('position');
  if (weldedIndex.count % 3 !== 0 || weldedAttr.count < 3) {
    throw new Error('The mesh could not be converted into indexed triangles.');
  }

  if (!(weldedAttr.array instanceof Float32Array)) {
    const array = new Float32Array((weldedAttr.array as ArrayLike<number>).length);
    array.set(weldedAttr.array as ArrayLike<number>);
    welded.setAttribute('position', new BufferAttribute(array, 3));
  }

  if (!(weldedIndex.array instanceof Uint32Array)) {
    const array = new Uint32Array((weldedIndex.array as ArrayLike<number>).length);
    array.set(weldedIndex.array as ArrayLike<number>);
    welded.setIndex(new BufferAttribute(array, 1));
  }

  welded.deleteAttribute('normal');
  welded.deleteAttribute('color');
  welded.deleteAttribute('uv');
  welded.clearGroups();
  welded.computeBoundingBox();
  welded.computeBoundingSphere();

  source.dispose();

  if ((welded.boundingSphere?.radius ?? 0) <= 0) {
    throw new Error('The mesh bounds are invalid after normalization.');
  }

  return welded;
}

function getMillimeterScale(unit: MeshUnit): number {
  switch (unit) {
    case 'cm':
      return 10;
    case 'm':
      return 1000;
    case 'mm':
    default:
      return 1;
  }
}

function createSequentialIndex(vertexCount: number): Uint32Array {
  const indices = new Uint32Array(vertexCount);
  for (let i = 0; i < vertexCount; i += 1) {
    indices[i] = i;
  }

  return indices;
}
