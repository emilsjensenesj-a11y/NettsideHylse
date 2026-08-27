import { BufferAttribute, Triangle, Vector3 } from 'three';

import { recomputeDisplacementColorsForVertices } from './editable-mesh';
import type { EditableMeshData } from './editable-mesh';
import type {
  BrushStamp,
  HistoryState,
  SculptHistorySnapshot,
  StrokeRecord,
  Vec3Like,
} from './types';

const DEFAULT_HISTORY_LIMIT = 12;
const SMOOTH_LAMBDA = 1.35;
const SMOOTH_MU = -0.42;
const NORMAL_DISPLACEMENT_SCALE = 0.18;
const FLATTEN_SCALE = 0.9;

export class SculptEngine {
  readonly data: EditableMeshData;
  preserveVertexColors = false;

  private readonly historyLimit: number;
  private readonly weightByVertex: Float32Array;
  private readonly activeVertices: Uint32Array;
  private readonly sortedVertices: Uint32Array;
  private readonly smoothBufferA: Float32Array;
  private readonly smoothBufferB: Float32Array;
  private readonly boundaryNeighborOffsets: Uint32Array;
  private readonly boundaryNeighbors: Uint32Array;
  private readonly strokeVertexMarks: Uint32Array;
  private readonly activeVertexMarks: Uint32Array;
  private readonly supportVertexMarks: Uint32Array;

  private readonly triangleScratch = new Triangle();
  private readonly triA = new Vector3();
  private readonly triB = new Vector3();
  private readonly triC = new Vector3();
  private readonly closestPoint = new Vector3();
  private readonly brushCenter = new Vector3();
  private undoStack: StrokeRecord[] = [];
  private redoStack: StrokeRecord[] = [];
  private strokeActive = false;
  private strokeVertexIds: number[] = [];
  private strokeBeforePositions: number[] = [];
  private strokeVertexStamp = 1;
  private activeVertexStamp = 1;
  private supportVertexStamp = 1;
  private regionTriangleStamp = 1;
  private regionVertexStamp = 1;
  private dirtyFaceStamp = 1;
  private dirtyVertexStamp = 1;

  constructor(data: EditableMeshData, historyLimit = DEFAULT_HISTORY_LIMIT) {
    this.data = data;
    this.historyLimit = historyLimit;
    this.weightByVertex = new Float32Array(data.vertexCount);
    this.activeVertices = new Uint32Array(data.vertexCount);
    this.sortedVertices = new Uint32Array(data.vertexCount);
    this.smoothBufferA = new Float32Array(data.positions.length);
    this.smoothBufferB = new Float32Array(data.positions.length);
    const boundaryNeighbors = buildBoundaryNeighborAdjacency(data.indices, data.positions, data.vertexCount);
    this.boundaryNeighborOffsets = boundaryNeighbors.offsets;
    this.boundaryNeighbors = boundaryNeighbors.values;
    this.strokeVertexMarks = new Uint32Array(data.vertexCount);
    this.activeVertexMarks = new Uint32Array(data.vertexCount);
    this.supportVertexMarks = new Uint32Array(data.vertexCount);
  }

  getHistoryState(): HistoryState {
    return {
      canUndo: this.undoStack.length > 0,
      canRedo: this.redoStack.length > 0,
      undoCount: this.undoStack.length,
      redoCount: this.redoStack.length,
    };
  }

  exportHistorySnapshot(): SculptHistorySnapshot {
    return {
      undoStack: this.undoStack.map(cloneStrokeRecord),
      redoStack: this.redoStack.map(cloneStrokeRecord),
    };
  }

  importHistorySnapshot(snapshot: SculptHistorySnapshot | null): void {
    if (!snapshot) {
      this.undoStack = [];
      this.redoStack = [];
      this.strokeActive = false;
      this.strokeVertexIds.length = 0;
      this.strokeBeforePositions.length = 0;
      return;
    }

    this.undoStack = snapshot.undoStack.map(cloneStrokeRecord);
    this.redoStack = snapshot.redoStack.map(cloneStrokeRecord);
    this.strokeActive = false;
    this.strokeVertexIds.length = 0;
    this.strokeBeforePositions.length = 0;
  }

  discardRedoHistory(): void {
    this.redoStack = [];
  }

  beginStroke(): void {
    this.strokeActive = true;
    this.strokeVertexIds.length = 0;
    this.strokeBeforePositions.length = 0;
    this.strokeVertexStamp = nextStamp(this.strokeVertexMarks, this.strokeVertexStamp);
  }

  endStroke(): StrokeRecord | null {
    if (!this.strokeActive) {
      return null;
    }

    this.strokeActive = false;
    if (this.strokeVertexIds.length === 0) {
      return null;
    }

    const vertexIds = Uint32Array.from(this.strokeVertexIds);
    const beforePositions = Float32Array.from(this.strokeBeforePositions);
    const afterPositions = new Float32Array(vertexIds.length * 3);
    this.captureCurrentPositions(vertexIds, afterPositions);

    const record: StrokeRecord = {
      vertexIds,
      beforePositions,
      afterPositions,
    };

    this.undoStack.push(record);
    if (this.undoStack.length > this.historyLimit) {
      this.undoStack.shift();
    }

    this.redoStack = [];
    return record;
  }

  applyStamp(stamp: BrushStamp): boolean {
    if (!this.strokeActive) {
      this.beginStroke();
    }

    if (stamp.faceIndex < 0 || stamp.faceIndex >= this.data.triangleCount) {
      return false;
    }

    const region = this.collectRegion(stamp.faceIndex, stamp.pointLocal, stamp.radius);
    if (region.vertexCount === 0) {
      return false;
    }

    let activeCount = this.populateActiveVertices(stamp, region.vertexCount);
    let trimlineBoundaryCount = activeCount;
    if (stamp.type === 'smooth' && stamp.smoothOnlyTrimline) {
      const trimlineCounts = this.partitionTrimlineBrushVertices(activeCount);
      trimlineBoundaryCount = trimlineCounts.boundaryCount;
      activeCount = trimlineCounts.totalCount;
    }
    if (activeCount === 0) {
      return false;
    }

    this.captureStrokeBefore(activeCount);

    switch (stamp.type) {
      case 'smooth':
        if (stamp.smoothOnlyTrimline) {
          this.applyTrimlineSmooth(stamp, trimlineBoundaryCount);
          this.applyTrimlineSupportSmooth(stamp, trimlineBoundaryCount, activeCount);
        } else {
          this.applySmooth(stamp, activeCount);
        }
        break;
      case 'inflate':
        this.applyNormalDisplacement(stamp, activeCount, 1);
        break;
      case 'carve':
        this.applyNormalDisplacement(stamp, activeCount, -1);
        break;
      case 'flatten':
        this.applyFlatten(stamp, activeCount);
        break;
    }
    if (!(stamp.type === 'smooth' && stamp.smoothOnlyTrimline)) {
      this.synchronizeActivePositionGroups(activeCount);
    }

    const dirtyVertexCount = this.recomputeLocalNormals(activeCount);
    this.markAttributeRanges(this.data.positionAttribute, this.activeVertices, activeCount);
    this.markAttributeRanges(this.data.normalAttribute, this.data.dirtyVertices, dirtyVertexCount);
    if (!this.preserveVertexColors) {
      recomputeDisplacementColorsForVertices(
        this.data.positions,
        this.data.referencePositions,
        this.data.normals,
        this.data.colors,
        this.data.dirtyVertices,
        dirtyVertexCount,
      );
      this.markAttributeRanges(this.data.colorAttribute, this.data.dirtyVertices, dirtyVertexCount);
    }
    this.data.geometry.boundsTree?.refit();
    return true;
  }

  collectTrianglesInSphere(faceIndex: number, pointLocal: Vec3Like, radius: number): number {
    return this.collectRegion(faceIndex, pointLocal, radius).triangleCount;
  }

  undo(): boolean {
    const record = this.undoStack.pop();
    if (!record) {
      return false;
    }

    this.applyStrokePositions(record.vertexIds, record.beforePositions);
    this.redoStack.push(record);
    return true;
  }

  redo(): boolean {
    const record = this.redoStack.pop();
    if (!record) {
      return false;
    }

    this.applyStrokePositions(record.vertexIds, record.afterPositions);
    this.undoStack.push(record);
    return true;
  }

  getFaceNormal(faceIndex: number, target: Vector3): Vector3 {
    const offset = faceIndex * 3;
    const x = this.data.faceNormals[offset];
    const y = this.data.faceNormals[offset + 1];
    const z = this.data.faceNormals[offset + 2];
    target.set(x, y, z);
    return target.normalize();
  }

  private collectRegion(
    faceIndex: number,
    pointLocal: Vec3Like,
    radius: number,
  ): { triangleCount: number; vertexCount: number } {
    const { regionQueue, regionTriangles, regionVertices, regionTriangleMarks, regionVertexMarks } =
      this.data;
    const { triangleNeighbors, indices } = this.data;

    this.regionTriangleStamp = nextStamp(regionTriangleMarks, this.regionTriangleStamp);
    this.regionVertexStamp = nextStamp(regionVertexMarks, this.regionVertexStamp);

    const radiusSq = radius * radius;
    this.brushCenter.set(pointLocal.x, pointLocal.y, pointLocal.z);

    let queueStart = 0;
    let queueEnd = 1;
    let triangleCount = 0;
    let vertexCount = 0;

    regionQueue[0] = faceIndex;
    regionTriangleMarks[faceIndex] = this.regionTriangleStamp;

    while (queueStart < queueEnd) {
      const triangle = regionQueue[queueStart];
      queueStart += 1;

      if (!this.triangleIntersectsSphere(triangle, radiusSq)) {
        continue;
      }

      regionTriangles[triangleCount] = triangle;
      triangleCount += 1;

      const triOffset = triangle * 3;
      for (let corner = 0; corner < 3; corner += 1) {
        const vertex = indices[triOffset + corner];
        if (regionVertexMarks[vertex] !== this.regionVertexStamp) {
          regionVertexMarks[vertex] = this.regionVertexStamp;
          regionVertices[vertexCount] = vertex;
          vertexCount += 1;
        }
      }

      for (let edge = 0; edge < 3; edge += 1) {
        const neighbor = triangleNeighbors[triOffset + edge];
        if (neighbor === -1 || regionTriangleMarks[neighbor] === this.regionTriangleStamp) {
          continue;
        }

        regionTriangleMarks[neighbor] = this.regionTriangleStamp;
        regionQueue[queueEnd] = neighbor;
        queueEnd += 1;
      }
    }

    return { triangleCount, vertexCount };
  }

  private populateActiveVertices(stamp: BrushStamp, candidateCount: number): number {
    const { regionVertices, positions, positionGroupOffsets, positionGroupVertices } = this.data;
    const radius = stamp.radius;
    const invRadius = 1 / radius;
    const cx = stamp.pointLocal.x;
    const cy = stamp.pointLocal.y;
    const cz = stamp.pointLocal.z;

    this.activeVertexStamp = nextStamp(this.activeVertexMarks, this.activeVertexStamp);
    let activeCount = 0;
    for (let i = 0; i < candidateCount; i += 1) {
      const vertex = regionVertices[i];
      const offset = vertex * 3;
      const dx = positions[offset] - cx;
      const dy = positions[offset + 1] - cy;
      const dz = positions[offset + 2] - cz;
      const distance = Math.hypot(dx, dy, dz);
      if (distance >= radius) {
        this.weightByVertex[vertex] = 0;
        continue;
      }

      const falloff = smootherStep(1 - distance * invRadius);
      if (falloff <= 1e-6) {
        this.weightByVertex[vertex] = 0;
        continue;
      }

      for (
        let groupOffset = positionGroupOffsets[vertex];
        groupOffset < positionGroupOffsets[vertex + 1];
        groupOffset += 1
      ) {
        const groupVertex = positionGroupVertices[groupOffset];
        if (this.activeVertexMarks[groupVertex] === this.activeVertexStamp) {
          continue;
        }

        this.activeVertexMarks[groupVertex] = this.activeVertexStamp;
        this.weightByVertex[groupVertex] = falloff;
        this.activeVertices[activeCount] = groupVertex;
        activeCount += 1;
      }
    }

    return activeCount;
  }

  private captureStrokeBefore(activeCount: number): void {
    const { positions } = this.data;
    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      if (this.strokeVertexMarks[vertex] === this.strokeVertexStamp) {
        continue;
      }

      this.strokeVertexMarks[vertex] = this.strokeVertexStamp;
      this.strokeVertexIds.push(vertex);
      const offset = vertex * 3;
      this.strokeBeforePositions.push(
        positions[offset],
        positions[offset + 1],
        positions[offset + 2],
      );
    }
  }

  private partitionTrimlineBrushVertices(activeCount: number): { boundaryCount: number; totalCount: number } {
    let boundaryCount = 0;
    let supportCount = 0;
    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      if (this.boundaryNeighborOffsets[vertex + 1] === this.boundaryNeighborOffsets[vertex]) {
        this.sortedVertices[supportCount] = vertex;
        supportCount += 1;
        continue;
      }

      this.activeVertices[boundaryCount] = vertex;
      boundaryCount += 1;
    }

    if (boundaryCount === 0) {
      for (let i = 0; i < supportCount; i += 1) {
        this.weightByVertex[this.sortedVertices[i]] = 0;
      }

      return { boundaryCount: 0, totalCount: 0 };
    }

    this.activeVertices.set(this.sortedVertices.subarray(0, supportCount), boundaryCount);
    return { boundaryCount, totalCount: boundaryCount + supportCount };
  }

  private applyNormalDisplacement(stamp: BrushStamp, activeCount: number, direction: 1 | -1): void {
    const { positions, normals } = this.data;
    const amplitude = stamp.radius * stamp.strength * NORMAL_DISPLACEMENT_SCALE * direction;

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      const weight = this.weightByVertex[vertex];
      const delta = amplitude * weight;
      positions[offset] += normals[offset] * delta;
      positions[offset + 1] += normals[offset + 1] * delta;
      positions[offset + 2] += normals[offset + 2] * delta;
    }
  }

  private applySmooth(stamp: BrushStamp, activeCount: number): void {
    const { positions, vertexNeighborOffsets, vertexNeighbors, regionVertexMarks } = this.data;

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      this.smoothBufferA[offset] = positions[offset];
      this.smoothBufferA[offset + 1] = positions[offset + 1];
      this.smoothBufferA[offset + 2] = positions[offset + 2];
    }

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      const weight = Math.min(this.weightByVertex[vertex] * stamp.strength * SMOOTH_LAMBDA, 0.95);

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let count = 0;

      for (
        let neighborOffset = vertexNeighborOffsets[vertex];
        neighborOffset < vertexNeighborOffsets[vertex + 1];
        neighborOffset += 1
      ) {
        const neighbor = vertexNeighbors[neighborOffset];
        const neighborPosition = neighbor * 3;
        sumX += positions[neighborPosition];
        sumY += positions[neighborPosition + 1];
        sumZ += positions[neighborPosition + 2];
        count += 1;
      }

      if (count === 0) {
        this.smoothBufferB[offset] = this.smoothBufferA[offset];
        this.smoothBufferB[offset + 1] = this.smoothBufferA[offset + 1];
        this.smoothBufferB[offset + 2] = this.smoothBufferA[offset + 2];
        continue;
      }

      const invCount = 1 / count;
      this.smoothBufferB[offset] =
        this.smoothBufferA[offset] + (sumX * invCount - this.smoothBufferA[offset]) * weight;
      this.smoothBufferB[offset + 1] =
        this.smoothBufferA[offset + 1] + (sumY * invCount - this.smoothBufferA[offset + 1]) * weight;
      this.smoothBufferB[offset + 2] =
        this.smoothBufferA[offset + 2] + (sumZ * invCount - this.smoothBufferA[offset + 2]) * weight;
    }

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      const weight = Math.min(this.weightByVertex[vertex] * stamp.strength, 0.95);
      const mu = SMOOTH_MU * weight;

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let count = 0;

      for (
        let neighborOffset = vertexNeighborOffsets[vertex];
        neighborOffset < vertexNeighborOffsets[vertex + 1];
        neighborOffset += 1
      ) {
        const neighbor = vertexNeighbors[neighborOffset];
        const source =
          regionVertexMarks[neighbor] === this.regionVertexStamp &&
          this.weightByVertex[neighbor] > 0
            ? this.smoothBufferB
            : positions;
        const neighborPosition = neighbor * 3;
        sumX += source[neighborPosition];
        sumY += source[neighborPosition + 1];
        sumZ += source[neighborPosition + 2];
        count += 1;
      }

      if (count === 0) {
        positions[offset] = this.smoothBufferB[offset];
        positions[offset + 1] = this.smoothBufferB[offset + 1];
        positions[offset + 2] = this.smoothBufferB[offset + 2];
        continue;
      }

      const invCount = 1 / count;
      positions[offset] =
        this.smoothBufferB[offset] + (sumX * invCount - this.smoothBufferB[offset]) * mu;
      positions[offset + 1] =
        this.smoothBufferB[offset + 1] + (sumY * invCount - this.smoothBufferB[offset + 1]) * mu;
      positions[offset + 2] =
        this.smoothBufferB[offset + 2] + (sumZ * invCount - this.smoothBufferB[offset + 2]) * mu;
    }

    this.relaxActiveBoundaryVertices(stamp, activeCount);
  }

  private applyTrimlineSmooth(stamp: BrushStamp, activeCount: number): void {
    const { positions, normals } = this.data;

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      this.smoothBufferA[offset] = positions[offset];
      this.smoothBufferA[offset + 1] = positions[offset + 1];
      this.smoothBufferA[offset + 2] = positions[offset + 2];
    }

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const start = this.boundaryNeighborOffsets[vertex];
      const end = this.boundaryNeighborOffsets[vertex + 1];
      const neighborCount = end - start;
      const offset = vertex * 3;
      if (neighborCount === 0) {
        continue;
      }

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      for (let neighborOffset = start; neighborOffset < end; neighborOffset += 1) {
        const neighbor = this.boundaryNeighbors[neighborOffset];
        const neighborPosition = neighbor * 3;
        const source =
          this.activeVertexMarks[neighbor] === this.activeVertexStamp && this.weightByVertex[neighbor] > 0
            ? this.smoothBufferA
            : positions;
        sumX += source[neighborPosition];
        sumY += source[neighborPosition + 1];
        sumZ += source[neighborPosition + 2];
      }

      const invCount = 1 / neighborCount;
      const targetX = sumX * invCount;
      const targetY = sumY * invCount;
      const targetZ = sumZ * invCount;
      const deltaX = targetX - this.smoothBufferA[offset];
      const deltaY = targetY - this.smoothBufferA[offset + 1];
      const deltaZ = targetZ - this.smoothBufferA[offset + 2];
      const normalX = normals[offset];
      const normalY = normals[offset + 1];
      const normalZ = normals[offset + 2];
      const normalLengthSq = normalX * normalX + normalY * normalY + normalZ * normalZ;
      let adjustedX = deltaX;
      let adjustedY = deltaY;
      let adjustedZ = deltaZ;
      if (normalLengthSq > 1e-12) {
        const invNormalLengthSq = 1 / normalLengthSq;
        const normalDelta = (deltaX * normalX + deltaY * normalY + deltaZ * normalZ) * invNormalLengthSq;
        const normalDeltaX = normalX * normalDelta;
        const normalDeltaY = normalY * normalDelta;
        const normalDeltaZ = normalZ * normalDelta;
        adjustedX = deltaX - normalDeltaX * 0.75;
        adjustedY = deltaY - normalDeltaY * 0.75;
        adjustedZ = deltaZ - normalDeltaZ * 0.75;
      }

      const weight = Math.min(this.weightByVertex[vertex] * stamp.strength * 1.45, 0.9);
      positions[offset] = this.smoothBufferA[offset] + adjustedX * weight;
      positions[offset + 1] = this.smoothBufferA[offset + 1] + adjustedY * weight;
      positions[offset + 2] = this.smoothBufferA[offset + 2] + adjustedZ * weight;
    }
  }

  private applyTrimlineSupportSmooth(stamp: BrushStamp, start: number, end: number): void {
    if (end <= start) {
      return;
    }

    const { positions, vertexNeighborOffsets, vertexNeighbors } = this.data;
    this.supportVertexStamp = nextStamp(this.supportVertexMarks, this.supportVertexStamp);

    for (let i = start; i < end; i += 1) {
      const vertex = this.activeVertices[i];
      this.supportVertexMarks[vertex] = this.supportVertexStamp;
      const offset = vertex * 3;
      this.smoothBufferA[offset] = positions[offset];
      this.smoothBufferA[offset + 1] = positions[offset + 1];
      this.smoothBufferA[offset + 2] = positions[offset + 2];
    }

    for (let i = start; i < end; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      const weight = Math.min(this.weightByVertex[vertex] * stamp.strength * SMOOTH_LAMBDA, 0.95);

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let count = 0;

      for (
        let neighborOffset = vertexNeighborOffsets[vertex];
        neighborOffset < vertexNeighborOffsets[vertex + 1];
        neighborOffset += 1
      ) {
        const neighbor = vertexNeighbors[neighborOffset];
        const neighborPosition = neighbor * 3;
        sumX += positions[neighborPosition];
        sumY += positions[neighborPosition + 1];
        sumZ += positions[neighborPosition + 2];
        count += 1;
      }

      if (count === 0) {
        this.smoothBufferB[offset] = this.smoothBufferA[offset];
        this.smoothBufferB[offset + 1] = this.smoothBufferA[offset + 1];
        this.smoothBufferB[offset + 2] = this.smoothBufferA[offset + 2];
        continue;
      }

      const invCount = 1 / count;
      this.smoothBufferB[offset] =
        this.smoothBufferA[offset] + (sumX * invCount - this.smoothBufferA[offset]) * weight;
      this.smoothBufferB[offset + 1] =
        this.smoothBufferA[offset + 1] + (sumY * invCount - this.smoothBufferA[offset + 1]) * weight;
      this.smoothBufferB[offset + 2] =
        this.smoothBufferA[offset + 2] + (sumZ * invCount - this.smoothBufferA[offset + 2]) * weight;
    }

    for (let i = start; i < end; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      const weight = Math.min(this.weightByVertex[vertex] * stamp.strength, 0.95);
      const mu = SMOOTH_MU * weight;

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let count = 0;

      for (
        let neighborOffset = vertexNeighborOffsets[vertex];
        neighborOffset < vertexNeighborOffsets[vertex + 1];
        neighborOffset += 1
      ) {
        const neighbor = vertexNeighbors[neighborOffset];
        const source = this.supportVertexMarks[neighbor] === this.supportVertexStamp ? this.smoothBufferB : positions;
        const neighborPosition = neighbor * 3;
        sumX += source[neighborPosition];
        sumY += source[neighborPosition + 1];
        sumZ += source[neighborPosition + 2];
        count += 1;
      }

      if (count === 0) {
        positions[offset] = this.smoothBufferB[offset];
        positions[offset + 1] = this.smoothBufferB[offset + 1];
        positions[offset + 2] = this.smoothBufferB[offset + 2];
        continue;
      }

      const invCount = 1 / count;
      positions[offset] =
        this.smoothBufferB[offset] + (sumX * invCount - this.smoothBufferB[offset]) * mu;
      positions[offset + 1] =
        this.smoothBufferB[offset + 1] + (sumY * invCount - this.smoothBufferB[offset + 1]) * mu;
      positions[offset + 2] =
        this.smoothBufferB[offset + 2] + (sumZ * invCount - this.smoothBufferB[offset + 2]) * mu;
    }
  }

  private relaxActiveBoundaryVertices(stamp: BrushStamp, activeCount: number): void {
    const { positions } = this.data;

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const start = this.boundaryNeighborOffsets[vertex];
      const end = this.boundaryNeighborOffsets[vertex + 1];
      const neighborCount = end - start;
      if (neighborCount < 2) {
        continue;
      }

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      for (let neighborOffset = start; neighborOffset < end; neighborOffset += 1) {
        const neighborPosition = this.boundaryNeighbors[neighborOffset] * 3;
        sumX += positions[neighborPosition];
        sumY += positions[neighborPosition + 1];
        sumZ += positions[neighborPosition + 2];
      }

      const offset = vertex * 3;
      const invCount = 1 / neighborCount;
      const weight = Math.min(this.weightByVertex[vertex] * stamp.strength * 1.45, 0.9);
      positions[offset] += (sumX * invCount - positions[offset]) * weight;
      positions[offset + 1] += (sumY * invCount - positions[offset + 1]) * weight;
      positions[offset + 2] += (sumZ * invCount - positions[offset + 2]) * weight;
    }
  }

  private applyFlatten(stamp: BrushStamp, activeCount: number): void {
    const { positions, normals } = this.data;
    let centroidX = 0;
    let centroidY = 0;
    let centroidZ = 0;
    let normalX = 0;
    let normalY = 0;
    let normalZ = 0;
    let weightSum = 0;

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      const weight = this.weightByVertex[vertex];
      centroidX += positions[offset] * weight;
      centroidY += positions[offset + 1] * weight;
      centroidZ += positions[offset + 2] * weight;
      normalX += normals[offset] * weight;
      normalY += normals[offset + 1] * weight;
      normalZ += normals[offset + 2] * weight;
      weightSum += weight;
    }

    if (weightSum <= 1e-6) {
      return;
    }

    centroidX /= weightSum;
    centroidY /= weightSum;
    centroidZ /= weightSum;

    let length = Math.hypot(normalX, normalY, normalZ);
    if (length <= 1e-6) {
      normalX = stamp.normalLocal.x;
      normalY = stamp.normalLocal.y;
      normalZ = stamp.normalLocal.z;
      length = Math.hypot(normalX, normalY, normalZ);
    }

    normalX /= length;
    normalY /= length;
    normalZ /= length;

    const orientation =
      normalX * stamp.normalLocal.x +
      normalY * stamp.normalLocal.y +
      normalZ * stamp.normalLocal.z;
    if (orientation < 0) {
      normalX *= -1;
      normalY *= -1;
      normalZ *= -1;
    }

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      const offset = vertex * 3;
      const weight = this.weightByVertex[vertex] * stamp.strength * FLATTEN_SCALE;
      const toPlaneX = positions[offset] - centroidX;
      const toPlaneY = positions[offset + 1] - centroidY;
      const toPlaneZ = positions[offset + 2] - centroidZ;
      const planeDistance = toPlaneX * normalX + toPlaneY * normalY + toPlaneZ * normalZ;

      positions[offset] -= normalX * planeDistance * weight;
      positions[offset + 1] -= normalY * planeDistance * weight;
      positions[offset + 2] -= normalZ * planeDistance * weight;
    }
  }

  private synchronizeActivePositionGroups(activeCount: number): void {
    const { positions, positionGroupOffsets, positionGroupVertices } = this.data;
    const groupStamp = nextStamp(this.activeVertexMarks, this.activeVertexStamp);
    this.activeVertexStamp = groupStamp;

    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      if (this.activeVertexMarks[vertex] === groupStamp) {
        continue;
      }

      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let count = 0;
      for (
        let groupOffset = positionGroupOffsets[vertex];
        groupOffset < positionGroupOffsets[vertex + 1];
        groupOffset += 1
      ) {
        const groupVertex = positionGroupVertices[groupOffset];
        this.activeVertexMarks[groupVertex] = groupStamp;
        const offset = groupVertex * 3;
        sumX += positions[offset];
        sumY += positions[offset + 1];
        sumZ += positions[offset + 2];
        count += 1;
      }

      if (count < 2) {
        continue;
      }

      const x = sumX / count;
      const y = sumY / count;
      const z = sumZ / count;
      for (
        let groupOffset = positionGroupOffsets[vertex];
        groupOffset < positionGroupOffsets[vertex + 1];
        groupOffset += 1
      ) {
        const groupVertex = positionGroupVertices[groupOffset];
        const offset = groupVertex * 3;
        positions[offset] = x;
        positions[offset + 1] = y;
        positions[offset + 2] = z;
      }
    }
  }

  private recomputeLocalNormals(activeCount: number): number {
    const {
      indices,
      positions,
      faceNormals,
      normals,
      vertexFaceOffsets,
      vertexFaces,
      dirtyFaces,
      dirtyFaceMarks,
      dirtyVertices,
      dirtyVertexMarks,
    } = this.data;

    this.dirtyFaceStamp = nextStamp(dirtyFaceMarks, this.dirtyFaceStamp);
    this.dirtyVertexStamp = nextStamp(dirtyVertexMarks, this.dirtyVertexStamp);

    let dirtyFaceCount = 0;
    for (let i = 0; i < activeCount; i += 1) {
      const vertex = this.activeVertices[i];
      for (
        let faceOffset = vertexFaceOffsets[vertex];
        faceOffset < vertexFaceOffsets[vertex + 1];
        faceOffset += 1
      ) {
        const face = vertexFaces[faceOffset];
        if (dirtyFaceMarks[face] === this.dirtyFaceStamp) {
          continue;
        }

        dirtyFaceMarks[face] = this.dirtyFaceStamp;
        dirtyFaces[dirtyFaceCount] = face;
        dirtyFaceCount += 1;
      }
    }

    for (let i = 0; i < dirtyFaceCount; i += 1) {
      const face = dirtyFaces[i];
      const triOffset = face * 3;
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

    let dirtyVertexCount = 0;
    for (let i = 0; i < dirtyFaceCount; i += 1) {
      const face = dirtyFaces[i] * 3;
      for (let corner = 0; corner < 3; corner += 1) {
        const vertex = indices[face + corner];
        if (dirtyVertexMarks[vertex] === this.dirtyVertexStamp) {
          continue;
        }

        dirtyVertexMarks[vertex] = this.dirtyVertexStamp;
        dirtyVertices[dirtyVertexCount] = vertex;
        dirtyVertexCount += 1;
      }
    }

    for (let i = 0; i < dirtyVertexCount; i += 1) {
      const vertex = dirtyVertices[i];
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

      const offset = vertex * 3;
      const length = Math.hypot(nx, ny, nz);
      if (length > 1e-12) {
        normals[offset] = nx / length;
        normals[offset + 1] = ny / length;
        normals[offset + 2] = nz / length;
      } else {
        normals[offset] = 0;
        normals[offset + 1] = 0;
        normals[offset + 2] = 1;
      }
    }

    return dirtyVertexCount;
  }

  private markAttributeRanges(attribute: BufferAttribute, vertexIds: Uint32Array, count: number): void {
    attribute.clearUpdateRanges();
    if (count === 0) {
      attribute.needsUpdate = true;
      return;
    }

    const sorted = this.sortedVertices.subarray(0, count);
    sorted.set(vertexIds.subarray(0, count));
    sorted.sort();

    let rangeStart = sorted[0];
    let previous = sorted[0];

    for (let i = 1; i < count; i += 1) {
      const current = sorted[i];
      if (current === previous || current === previous + 1) {
        previous = current;
        continue;
      }

      attribute.addUpdateRange(rangeStart * 3, (previous - rangeStart + 1) * 3);
      rangeStart = current;
      previous = current;
    }

    attribute.addUpdateRange(rangeStart * 3, (previous - rangeStart + 1) * 3);
    attribute.needsUpdate = true;
  }

  private captureCurrentPositions(vertexIds: Uint32Array, target: Float32Array): void {
    const { positions } = this.data;
    for (let i = 0; i < vertexIds.length; i += 1) {
      const vertex = vertexIds[i];
      const src = vertex * 3;
      const dst = i * 3;
      target[dst] = positions[src];
      target[dst + 1] = positions[src + 1];
      target[dst + 2] = positions[src + 2];
    }
  }

  private applyStrokePositions(vertexIds: Uint32Array, source: Float32Array): void {
    const { positions } = this.data;
    for (let i = 0; i < vertexIds.length; i += 1) {
      const vertex = vertexIds[i];
      const dst = vertex * 3;
      const src = i * 3;
      positions[dst] = source[src];
      positions[dst + 1] = source[src + 1];
      positions[dst + 2] = source[src + 2];
    }

    const dirtyVertexCount = this.recomputeNormalsForVertexSet(vertexIds);
    this.markAttributeRanges(this.data.positionAttribute, vertexIds, vertexIds.length);
    this.markAttributeRanges(this.data.normalAttribute, this.data.dirtyVertices, dirtyVertexCount);
    if (!this.preserveVertexColors) {
      recomputeDisplacementColorsForVertices(
        this.data.positions,
        this.data.referencePositions,
        this.data.normals,
        this.data.colors,
        this.data.dirtyVertices,
        dirtyVertexCount,
      );
      this.markAttributeRanges(this.data.colorAttribute, this.data.dirtyVertices, dirtyVertexCount);
    }
    this.data.geometry.boundsTree?.refit();
  }

  private recomputeNormalsForVertexSet(vertexIds: Uint32Array): number {
    const { activeVertices } = this;
    activeVertices.set(vertexIds, 0);
    return this.recomputeLocalNormals(vertexIds.length);
  }

  private triangleIntersectsSphere(triangleIndex: number, radiusSq: number): boolean {
    const { positions, indices } = this.data;
    const triOffset = triangleIndex * 3;

    setVectorFromPositions(this.triA, positions, indices[triOffset]);
    setVectorFromPositions(this.triB, positions, indices[triOffset + 1]);
    setVectorFromPositions(this.triC, positions, indices[triOffset + 2]);
    this.triangleScratch.set(this.triA, this.triB, this.triC);
    this.triangleScratch.closestPointToPoint(this.brushCenter, this.closestPoint);
    return this.closestPoint.distanceToSquared(this.brushCenter) <= radiusSq;
  }
}

function setVectorFromPositions(target: Vector3, positions: Float32Array, vertex: number): void {
  const offset = vertex * 3;
  target.set(positions[offset], positions[offset + 1], positions[offset + 2]);
}

function smootherStep(value: number): number {
  const x = Math.min(Math.max(value, 0), 1);
  return x * x * x * (x * (x * 6 - 15) + 10);
}

function nextStamp(marks: Uint32Array, current: number): number {
  const next = current + 1;
  if (next !== 0) {
    return next;
  }

  marks.fill(0);
  return 1;
}

function cloneStrokeRecord(record: StrokeRecord): StrokeRecord {
  return {
    vertexIds: record.vertexIds.slice(),
    beforePositions: record.beforePositions.slice(),
    afterPositions: record.afterPositions.slice(),
  };
}

function buildBoundaryNeighborAdjacency(
  indices: Uint32Array,
  positions: Float32Array,
  vertexCount: number,
): { offsets: Uint32Array; values: Uint32Array } {
  const positionGroups = buildCanonicalVertexGroups(positions, vertexCount);
  const edgeCounts = new Map<string, { a: number; b: number; count: number }>();
  for (let triangle = 0; triangle < indices.length; triangle += 3) {
    addEdge(indices[triangle], indices[triangle + 1], edgeCounts, positionGroups.canonicalByVertex);
    addEdge(indices[triangle + 1], indices[triangle + 2], edgeCounts, positionGroups.canonicalByVertex);
    addEdge(indices[triangle + 2], indices[triangle], edgeCounts, positionGroups.canonicalByVertex);
  }

  const neighborSets = Array.from({ length: vertexCount }, () => new Set<number>());
  for (const edge of edgeCounts.values()) {
    if (edge.count !== 1) {
      continue;
    }

    const groupA = positionGroups.groupsByCanonical.get(edge.a) ?? [edge.a];
    const groupB = positionGroups.groupsByCanonical.get(edge.b) ?? [edge.b];
    for (let aIndex = 0; aIndex < groupA.length; aIndex += 1) {
      const a = groupA[aIndex];
      for (let bIndex = 0; bIndex < groupB.length; bIndex += 1) {
        const b = groupB[bIndex];
        if (a === b) {
          continue;
        }

        neighborSets[a].add(b);
        neighborSets[b].add(a);
      }
    }
  }

  const offsets = new Uint32Array(vertexCount + 1);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    offsets[vertex + 1] = offsets[vertex] + neighborSets[vertex].size;
  }

  const values = new Uint32Array(offsets[vertexCount]);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    values.set(Array.from(neighborSets[vertex]), offsets[vertex]);
  }

  return { offsets, values };
}

function addEdge(
  a: number,
  b: number,
  edgeCounts: Map<string, { a: number; b: number; count: number }>,
  canonicalByVertex: Uint32Array,
): void {
  if (a === b) {
    return;
  }

  const canonicalA = canonicalByVertex[a];
  const canonicalB = canonicalByVertex[b];
  if (canonicalA === canonicalB) {
    return;
  }

  const low = Math.min(canonicalA, canonicalB);
  const high = Math.max(canonicalA, canonicalB);
  const key = `${low}:${high}`;
  const existing = edgeCounts.get(key);
  if (existing) {
    existing.count += 1;
    return;
  }

  edgeCounts.set(key, { a: low, b: high, count: 1 });
}

function buildCanonicalVertexGroups(
  positions: Float32Array,
  vertexCount: number,
): { canonicalByVertex: Uint32Array; groupsByCanonical: Map<number, number[]> } {
  const canonicalByVertex = new Uint32Array(vertexCount);
  const canonicalByPosition = new Map<string, number>();
  const groupsByCanonical = new Map<number, number[]>();

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const offset = vertex * 3;
    const key = makePositionKey(positions[offset], positions[offset + 1], positions[offset + 2]);
    const existing = canonicalByPosition.get(key);
    if (existing !== undefined) {
      canonicalByVertex[vertex] = existing;
      groupsByCanonical.get(existing)?.push(vertex);
      continue;
    }

    canonicalByPosition.set(key, vertex);
    canonicalByVertex[vertex] = vertex;
    groupsByCanonical.set(vertex, [vertex]);
  }

  return { canonicalByVertex, groupsByCanonical };
}

function makePositionKey(x: number, y: number, z: number): string {
  return `${x.toFixed(5)},${y.toFixed(5)},${z.toFixed(5)}`;
}
