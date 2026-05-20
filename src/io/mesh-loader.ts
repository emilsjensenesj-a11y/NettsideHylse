import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  Group,
  LoadingManager,
  type Material,
  Mesh,
  SRGBColorSpace,
  TextureLoader,
  type Texture,
  Vector3,
} from 'three';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import {
  mergeGeometries,
  mergeVertices,
} from 'three/examples/jsm/utils/BufferGeometryUtils.js';

const box = new Box3();
const center = new Vector3();
const MERGE_BY_DISTANCE_TOLERANCE_MM = 0.01;

export type MeshUnit = 'mm' | 'cm' | 'm';

export interface LoadedMeshAsset {
  geometry: BufferGeometry;
  filename: string;
  extension: string;
  vertexCount: number;
  triangleCount: number;
  boundsRadius: number;
  importUnit: MeshUnit;
  texture: Texture | null;
  textureFile: File | null;
}

export async function loadMeshFile(fileOrFiles: File | readonly File[], importUnit: MeshUnit = 'mm'): Promise<LoadedMeshAsset> {
  const files = Array.isArray(fileOrFiles) ? fileOrFiles : [fileOrFiles];
  const file = findMeshFile(files);
  const extension = getFileExtension(file.name);

  let parsedGeometry: BufferGeometry;
  let texture: Texture | null = null;
  let textureFile: File | null = null;
  if (extension === 'obj') {
    const selectedTexture = await loadSelectedObjTexture(file, files);
    texture = selectedTexture?.texture ?? null;
    textureFile = selectedTexture?.file ?? null;
    const parsedObj = await parseObj(file, files);
    texture ??= extractFirstTexture(parsedObj);
    parsedGeometry = normalizeGeometry(mergeObjectMeshes(parsedObj), importUnit);
  } else if (extension === 'stl') {
    const buffer = await file.arrayBuffer();
    parsedGeometry = normalizeGeometry(new STLLoader().parse(buffer), importUnit);
  } else {
    throw new Error('Unsupported file type. Choose an STL or OBJ mesh.');
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
    texture,
    textureFile,
  };
}

function findMeshFile(files: readonly File[]): File {
  const meshFile = files.find((file) => {
    const extension = getFileExtension(file.name);
    return extension === 'obj' || extension === 'stl';
  });
  if (!meshFile) {
    throw new Error('Choose an STL or OBJ mesh.');
  }

  return meshFile;
}

function getFileExtension(filename: string): string {
  const dotIndex = filename.lastIndexOf('.');
  return dotIndex >= 0 ? filename.slice(dotIndex + 1).toLowerCase() : '';
}

async function parseObj(file: File, files: readonly File[]): Promise<Group> {
  const text = await file.text();
  const loader = new OBJLoader(createLocalFileManager(files));
  const mtlFile = findReferencedMtlFile(text, files);
  if (mtlFile) {
    const materials = new MTLLoader(loader.manager).parse(await mtlFile.text(), '');
    materials.preload();
    loader.setMaterials(materials);
  }

  return loader.parse(text);
}

async function loadSelectedObjTexture(
  objFile: File,
  files: readonly File[],
): Promise<{ texture: Texture; file: File } | null> {
  const objText = await objFile.text();
  const mtlFile = findReferencedMtlFile(objText, files);
  const mtlText = mtlFile ? await mtlFile.text() : '';
  const textureFile = findReferencedTextureFile(mtlText, files);
  if (!textureFile) {
    return null;
  }

  const objectUrl = URL.createObjectURL(textureFile);
  return new Promise((resolve) => {
    new TextureLoader().load(
      objectUrl,
      (texture) => {
        texture.colorSpace = SRGBColorSpace;
        resolve({ texture, file: textureFile });
      },
      undefined,
      () => resolve(null),
    );
  });
}

function createLocalFileManager(files: readonly File[]): LoadingManager {
  const manager = new LoadingManager();
  const objectUrls = new Map<string, string>();
  for (const file of files) {
    objectUrls.set(normalizeFileKey(file.name), URL.createObjectURL(file));
  }

  manager.setURLModifier((url) => {
    const key = normalizeFileKey(url.split(/[\\/]/).pop() ?? url);
    return objectUrls.get(key) ?? url;
  });
  return manager;
}

function findReferencedMtlFile(objText: string, files: readonly File[]): File | null {
  const mtlMatch = objText.match(/^mtllib\s+(.+)$/im);
  const referencedName = mtlMatch?.[1]?.trim();
  if (referencedName) {
    const referencedKey = normalizeFileKey(referencedName.split(/[\\/]/).pop() ?? referencedName);
    const referencedFile = files.find((file) => normalizeFileKey(file.name) === referencedKey);
    if (referencedFile) {
      return referencedFile;
    }
  }

  return files.find((file) => getFileExtension(file.name) === 'mtl') ?? null;
}

function findReferencedTextureFile(mtlText: string, files: readonly File[]): File | null {
  const textureFiles = files.filter((file) => isTextureExtension(getFileExtension(file.name)));
  if (textureFiles.length === 0) {
    return null;
  }

  const textureLines = mtlText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^map_Kd\s+/i.test(line))
    .map(normalizeFileKey);

  for (const textureFile of textureFiles) {
    const textureKey = normalizeFileKey(textureFile.name);
    const textureBase = normalizeFileKey(textureFile.name.split(/[\\/]/).pop() ?? textureFile.name);
    if (textureLines.some((line) => line.includes(textureKey) || line.includes(textureBase))) {
      return textureFile;
    }
  }

  return textureFiles[0] ?? null;
}

function isTextureExtension(extension: string): boolean {
  return ['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(extension);
}

function extractFirstTexture(root: Group): Texture | null {
  let texture: Texture | null = null;
  root.traverse((child) => {
    if (texture || !(child instanceof Mesh)) {
      return;
    }

    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      const candidate = (material as Material & { map?: Texture | null }).map ?? null;
      if (candidate) {
        candidate.colorSpace = SRGBColorSpace;
        texture = candidate;
        return;
      }
    }
  });

  return texture;
}

function normalizeFileKey(path: string): string {
  return decodeURIComponent(path).replace(/^\.?\//, '').toLowerCase();
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
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      const firstMap = materials
        .map((material) => (material as Material & { map?: Texture | null }).map ?? null)
        .find((map): map is Texture => map !== null);
      if (firstMap && childGeometry.getAttribute('uv')) {
        childGeometry.userData.texture = firstMap;
      }
    }
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
  const uvAttr = source.getAttribute('uv');
  if (uvAttr) {
    const sourceUvs = uvAttr.array as ArrayLike<number>;
    const uvs = new Float32Array(sourceUvs.length);
    uvs.set(sourceUvs);
    geometry.setAttribute('uv', new BufferAttribute(uvs, 2));
  }
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
  applyNouraScannerOrientation(geometry);

  geometry.computeBoundingBox();
  if (!geometry.boundingBox) {
    throw new Error('Failed to compute mesh bounds.');
  }

  box.copy(geometry.boundingBox);
  box.getCenter(center);
  geometry.translate(-center.x, -center.y, -center.z);

  const welded = mergeVertices(geometry, MERGE_BY_DISTANCE_TOLERANCE_MM);

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

  flipInwardFacingWinding(welded);

  welded.deleteAttribute('normal');
  welded.deleteAttribute('color');
  welded.clearGroups();
  welded.computeBoundingBox();
  welded.computeBoundingSphere();

  source.dispose();

  if ((welded.boundingSphere?.radius ?? 0) <= 0) {
    throw new Error('The mesh bounds are invalid after normalization.');
  }

  return welded;
}

function applyNouraScannerOrientation(geometry: BufferGeometry): void {
  geometry.scale(1, -1, 1);
  reverseTriangleWinding(geometry);
}

function reverseTriangleWinding(geometry: BufferGeometry): void {
  const index = geometry.getIndex();
  if (!index) {
    return;
  }

  const array = index.array as Uint16Array | Uint32Array;
  for (let triangle = 0; triangle < array.length; triangle += 3) {
    const swap = array[triangle + 1];
    array[triangle + 1] = array[triangle + 2];
    array[triangle + 2] = swap;
  }

  index.needsUpdate = true;
}

function flipInwardFacingWinding(geometry: BufferGeometry): void {
  const positionAttr = geometry.getAttribute('position');
  const index = geometry.getIndex();
  if (!positionAttr || !index) {
    return;
  }

  const positions = positionAttr.array as ArrayLike<number>;
  const indices = index.array as ArrayLike<number>;
  const meshCenter = computeAreaWeightedCenter(positions, indices);
  let inwardCount = 0;
  let outwardCount = 0;

  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    const a = indices[triangle];
    const b = indices[triangle + 1];
    const c = indices[triangle + 2];
    const aOffset = a * 3;
    const bOffset = b * 3;
    const cOffset = c * 3;

    const abX = positions[bOffset] - positions[aOffset];
    const abY = positions[bOffset + 1] - positions[aOffset + 1];
    const abZ = positions[bOffset + 2] - positions[aOffset + 2];
    const acX = positions[cOffset] - positions[aOffset];
    const acY = positions[cOffset + 1] - positions[aOffset + 1];
    const acZ = positions[cOffset + 2] - positions[aOffset + 2];
    const normalX = abY * acZ - abZ * acY;
    const normalY = abZ * acX - abX * acZ;
    const normalZ = abX * acY - abY * acX;
    const normalLengthSq = normalX * normalX + normalY * normalY + normalZ * normalZ;
    if (normalLengthSq <= 1e-16) {
      continue;
    }

    const centroidX = (positions[aOffset] + positions[bOffset] + positions[cOffset]) / 3;
    const centroidY = (positions[aOffset + 1] + positions[bOffset + 1] + positions[cOffset + 1]) / 3;
    const centroidZ = (positions[aOffset + 2] + positions[bOffset + 2] + positions[cOffset + 2]) / 3;
    const directionX = centroidX - meshCenter.x;
    const directionY = centroidY - meshCenter.y;
    const directionZ = centroidZ - meshCenter.z;
    const dot = normalX * directionX + normalY * directionY + normalZ * directionZ;
    if (dot < 0) {
      inwardCount += 1;
    } else if (dot > 0) {
      outwardCount += 1;
    }
  }

  if (inwardCount > outwardCount) {
    reverseTriangleWinding(geometry);
  }
}

function computeAreaWeightedCenter(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
): Vector3 {
  const result = new Vector3();
  let areaSum = 0;

  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    const aOffset = indices[triangle] * 3;
    const bOffset = indices[triangle + 1] * 3;
    const cOffset = indices[triangle + 2] * 3;
    const abX = positions[bOffset] - positions[aOffset];
    const abY = positions[bOffset + 1] - positions[aOffset + 1];
    const abZ = positions[bOffset + 2] - positions[aOffset + 2];
    const acX = positions[cOffset] - positions[aOffset];
    const acY = positions[cOffset + 1] - positions[aOffset + 1];
    const acZ = positions[cOffset + 2] - positions[aOffset + 2];
    const crossX = abY * acZ - abZ * acY;
    const crossY = abZ * acX - abX * acZ;
    const crossZ = abX * acY - abY * acX;
    const area = Math.hypot(crossX, crossY, crossZ) * 0.5;
    if (area <= 1e-12) {
      continue;
    }

    result.x += ((positions[aOffset] + positions[bOffset] + positions[cOffset]) / 3) * area;
    result.y += ((positions[aOffset + 1] + positions[bOffset + 1] + positions[cOffset + 1]) / 3) * area;
    result.z += ((positions[aOffset + 2] + positions[bOffset + 2] + positions[cOffset + 2]) / 3) * area;
    areaSum += area;
  }

  if (areaSum > 0) {
    result.multiplyScalar(1 / areaSum);
    return result;
  }

  let vertexCount = 0;
  for (let vertex = 0; vertex < positions.length; vertex += 3) {
    result.x += positions[vertex];
    result.y += positions[vertex + 1];
    result.z += positions[vertex + 2];
    vertexCount += 1;
  }

  if (vertexCount > 0) {
    result.multiplyScalar(1 / vertexCount);
  }
  return result;
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
