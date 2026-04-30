export type BrushType = 'bump' | 'smooth' | 'flatten';
export type InteractionMode = 'sculpt' | 'select' | 'fill' | 'boundary' | 'positive' | 'remesh' | 'thicken';
export type SelectionTool = 'sphere' | 'box' | 'snip';

export interface Vec3Like {
  x: number;
  y: number;
  z: number;
}

export interface BrushStamp {
  pointLocal: Vec3Like;
  normalLocal: Vec3Like;
  faceIndex: number;
  radius: number;
  strength: number;
  type: BrushType;
}

export interface StrokeRecord {
  vertexIds: Uint32Array;
  beforePositions: Float32Array;
  afterPositions: Float32Array;
}

export interface SculptHistorySnapshot {
  undoStack: StrokeRecord[];
  redoStack: StrokeRecord[];
}

export interface HistoryState {
  canUndo: boolean;
  canRedo: boolean;
  undoCount: number;
  redoCount: number;
}

export interface SelectionState {
  selectedTriangleCount: number;
  canDelete: boolean;
}

export interface MeshStats {
  vertexCount: number;
  triangleCount: number;
  boundsRadius: number;
}

export interface HoleLoopSummary {
  loopCount: number;
  edgeCount: number;
}

export interface BoundaryWorkflowState {
  hasSelectedBoundary: boolean;
  selectedBoundaryEdgeCount: number;
  smoothCommitted: boolean;
  remeshApplied: boolean;
  thickenApplied: boolean;
  extrudeApplied: boolean;
  hasBoundaryGuide: boolean;
  canOffsetSelect: boolean;
  offsetApplied: boolean;
  selectedTriangleCount: number;
}
