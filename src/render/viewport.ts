import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  MOUSE,
  Mesh,
  MeshBasicMaterial,
  MeshMatcapMaterial,
  NoToneMapping,
  PerspectiveCamera,
  Raycaster,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  Vector2,
  Vector3,
  WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';

import {
  captureBoundaryGuide,
  selectTrianglesNearBoundaryGuide,
  smoothBoundaryLoopVertices,
} from '../ops/boundary-workflow';
import {
  computeCoherentBoundaryNormals,
  extrudeBoundaryLoop,
  extrudeBoundaryLoopAlongVector,
} from '../ops/boundary-extrude';
import { surfaceRemeshMesh } from '../ops/surface-remesh';
import type { RemeshBoundaryMode } from '../ops/surface-remesh';
import {
  laplacianSmoothSelected,
  laplacianSmoothSelectionBoundary,
  refineSelectedTriangles,
} from '../ops/selection-edit';
import { remeshSelectedTriangles } from '../ops/selection-remesh';
import { thickenMesh } from '../ops/thicken';
import {
  createEditableMeshData,
  recomputeAllNormals,
  recomputeDisplacementColorsRange,
} from '../sculpt/editable-mesh';
import type { EditableMeshData } from '../sculpt/editable-mesh';
import {
  createMesh as createHoleFillMesh,
  fillHole as executeHoleFill,
} from '../sculpt/hole-fill';
import { buildHoleLoopSet } from '../sculpt/hole-loops';
import { SculptEngine } from '../sculpt/sculpt-engine';
import type {
  BrushType,
  BoundaryWorkflowState,
  HistoryState,
  HoleLoopSummary,
  InteractionMode,
  MeshStats,
  SculptHistorySnapshot,
  SelectionState,
  SelectionTool,
} from '../sculpt/types';
import type { HoleLoop } from '../sculpt/hole-loops';

interface HoverHit {
  faceIndex: number;
  pointLocal: Vector3;
  normalLocal: Vector3;
}

interface ViewportCallbacks {
  onHistoryChange?: (state: HistoryState) => void;
  onSelectionChange?: (state: SelectionState) => void;
  onBoundaryWorkflowChange?: (state: BoundaryWorkflowState) => void;
  onBoundaryAction?: (result: { success: boolean; message: string }) => void;
  onMeshStatsChange?: (stats: MeshStats) => void;
  onHoleFill?: (result: { success: boolean; message: string }) => void;
}

interface ViewState {
  position: Vector3;
  target: Vector3;
  near: number;
  far: number;
  zoom: number;
}

interface SessionSnapshot {
  sessionId: number;
  positions: Float32Array | null;
  indices: Uint32Array | null;
  referencePositions: Float32Array | null;
  history: SculptHistorySnapshot | null;
  selectedTriangleMask: Uint8Array | null;
  selectedTriangleCount: number;
}

interface SessionInstallOptions {
  sessionId?: number;
  resetActionHistory?: boolean;
  resetView?: boolean;
  selectedTriangleMask?: Uint8Array | null;
  selectedTriangleCount?: number;
}

interface BoundarySessionState {
  guide: Float32Array | null;
  activeBoundaryVertexIds: Uint32Array | null;
  smoothCommitted: boolean;
  remeshApplied: boolean;
  thickenApplied: boolean;
  extrudeApplied: boolean;
  offsetApplied: boolean;
}

export interface ViewportActionResult {
  success: boolean;
  message: string;
  stats: MeshStats | null;
}

type ViewportHistoryAction =
  | {
      kind: 'stroke';
      sessionId: number;
    }
  | {
      kind: 'session';
      before: SessionSnapshot;
      after: SessionSnapshot;
    };

type SelectionOperation = 'replace' | 'add' | 'subtract';
type OrbitMouseAction = (typeof MOUSE)[keyof typeof MOUSE];

const DISABLED_MOUSE_ACTION = -1 as OrbitMouseAction;
const HOLE_LOOP_HOVER_DISTANCE_PX = 16;
const HOLE_FILL_DEBUG = true;
const ACTION_HISTORY_LIMIT = 12;
const SCULPT_HISTORY_LIMIT = 12;

export class ViewportController {
  private readonly container: HTMLElement;
  private readonly callbacks: ViewportCallbacks;
  private readonly renderer: WebGLRenderer;
  private readonly scene: Scene;
  private readonly camera: PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly raycaster = new Raycaster();
  private readonly sculptMatcapTexture: CanvasTexture;
  private readonly overlayCanvas: HTMLCanvasElement;
  private readonly overlayContext: CanvasRenderingContext2D;
  private readonly pointerNdc = new Vector2();
  private readonly pointerClient = new Vector2();
  private readonly selectionRayNdc = new Vector2();
  private readonly selectionStart = new Vector2();
  private readonly selectionCurrent = new Vector2();
  private readonly worldHitPoint = new Vector3();
  private readonly localHitPoint = new Vector3();
  private readonly localHitNormal = new Vector3();
  private readonly interpolatedPoint = new Vector3();
  private readonly interpolatedNormal = new Vector3();
  private readonly lastStampPoint = new Vector3();
  private readonly lastStampNormal = new Vector3();
  private readonly triangleCentroid = new Vector3();
  private readonly triangleWorldPoint = new Vector3();
  private readonly triangleWorldA = new Vector3();
  private readonly triangleWorldB = new Vector3();
  private readonly triangleWorldC = new Vector3();
  private readonly triangleNormal = new Vector3();
  private readonly triangleWorldNormal = new Vector3();
  private readonly cameraWorldPosition = new Vector3();
  private readonly viewDirection = new Vector3();
  private readonly projectedPoint = new Vector3();
  private readonly projectedPointA = new Vector3();
  private readonly projectedPointB = new Vector3();
  private readonly projectedPointC = new Vector3();
  private readonly resizeObserver: ResizeObserver;

  private editableMesh: EditableMeshData | null = null;
  private sculptEngine: SculptEngine | null = null;
  private mesh: Mesh | null = null;
  private meshMaterial: MeshMatcapMaterial | null = null;
  private cursor: Mesh | null = null;
  private selectionOverlay: Mesh | null = null;
  private selectionOverlayGeometry: BufferGeometry | null = null;
  private holeLoopOverlay: LineSegments2 | null = null;
  private holeLoopOverlayGeometry: LineSegmentsGeometry | null = null;
  private holeHoverOverlay: LineSegments2 | null = null;
  private holeHoverOverlayGeometry: LineSegmentsGeometry | null = null;
  private holeLoops: HoleLoop[] = [];
  private wireframeEnabled = false;
  private pointerInside = false;
  private pointerDown = false;
  private activeStroke = false;
  private selectionGestureActive = false;
  private holeFillMode = false;
  private hoveredHoleLoopIndex = -1;
  private activeBoundaryLoopIndex = -1;
  private activeBoundaryVertexIds: Uint32Array | null = null;
  private boundaryGuide: Float32Array | null = null;
  private boundaryPreviewBaseSnapshot: SessionSnapshot | null = null;
  private boundaryThickenPreviewBaseSnapshot: SessionSnapshot | null = null;
  private boundaryExtrudePreviewBaseSnapshot: SessionSnapshot | null = null;
  private boundaryFinalSmoothPreviewBaseSnapshot: SessionSnapshot | null = null;
  private boundaryDirectionalExtrudePreviewBaseSnapshot: SessionSnapshot | null = null;
  private boundarySmoothCommitted = false;
  private boundaryRemeshApplied = false;
  private boundaryThickenApplied = false;
  private boundaryExtrudeApplied = false;
  private boundaryOffsetApplied = false;
  private hoverHit: HoverHit | null = null;
  private interactionMode: InteractionMode = 'sculpt';
  private selectionTool: SelectionTool = 'sphere';
  private selectionOperation: SelectionOperation = 'replace';
  private selectionPath: Vector2[] = [];
  private selectedTriangleMask: Uint8Array | null = null;
  private selectedTriangleCount = 0;
  private selectionDirty = false;
  private brushType: BrushType = 'bump';
  private brushRadiusMm = 5;
  private brushStrength = 0.35;
  private selectionRadiusMm = 6;
  private nextSessionId = 1;
  private currentSessionId = 0;
  private historyUndoStack: ViewportHistoryAction[] = [];
  private historyRedoStack: ViewportHistoryAction[] = [];

  constructor(container: HTMLElement, callbacks: ViewportCallbacks = {}) {
    this.container = container;
    this.callbacks = callbacks;

    this.scene = new Scene();
    this.scene.background = new Color('#e9eef2');

    this.camera = new PerspectiveCamera(50, 1, 0.01, 1000);
    this.camera.position.set(2.8, 1.8, 3.4);
    this.scene.add(this.camera);

    this.renderer = new WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = NoToneMapping;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.domElement.className = 'viewport-canvas';
    this.renderer.domElement.style.touchAction = 'none';
    this.container.append(this.renderer.domElement);

    this.overlayCanvas = document.createElement('canvas');
    this.overlayCanvas.className = 'viewport-overlay';
    const overlayContext = this.overlayCanvas.getContext('2d');
    if (!overlayContext) {
      throw new Error('Failed to create the selection overlay canvas.');
    }

    this.overlayContext = overlayContext;
    this.container.append(this.overlayCanvas);
    this.sculptMatcapTexture = createStudioClayMatcapTexture();

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);
    this.controls.mouseButtons.LEFT = DISABLED_MOUSE_ACTION;
    this.controls.mouseButtons.MIDDLE = MOUSE.ROTATE;
    this.controls.mouseButtons.RIGHT = MOUSE.PAN;

    this.raycaster.firstHitOnly = true;

    this.attachEvents();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  setBrushType(type: BrushType): void {
    this.brushType = type;
  }

  setBrushRadiusMm(radiusMm: number): void {
    this.brushRadiusMm = radiusMm;
  }

  setBrushStrength(strength: number): void {
    this.brushStrength = strength;
  }

  setInteractionMode(mode: InteractionMode): void {
    this.interactionMode = mode;
    if (mode !== 'sculpt') {
      this.finishStroke();
    }

    if (mode !== 'select') {
      this.finishSelectionGesture();
    }

    const shouldEnableFill = mode === 'fill' || mode === 'boundary' || mode === 'positive';
    if (shouldEnableFill !== this.holeFillMode) {
      this.holeFillMode = shouldEnableFill;
      this.hoveredHoleLoopIndex = -1;
      if (shouldEnableFill && this.editableMesh && this.mesh) {
        this.rebuildHoleLoopOverlays();
      }

      this.updateHoleLoopOverlayVisibility();
      this.updateHoleHoverOverlay();
    }

    this.updateCursorVisuals();
    this.emitBoundaryWorkflow();
  }

  setSelectionTool(tool: SelectionTool): void {
    this.selectionTool = tool;
    this.finishSelectionGesture();
    this.updateCursorVisuals();
  }

  setSelectionRadiusMm(radiusMm: number): void {
    this.selectionRadiusMm = radiusMm;
  }

  setFillHoleMode(enabled: boolean): HoleLoopSummary | null {
    this.setInteractionMode(enabled ? 'fill' : 'sculpt');
    if (!enabled || !this.editableMesh || !this.mesh) {
      return null;
    }

    return {
      loopCount: this.holeLoops.length,
      edgeCount: this.holeLoops.reduce((sum, loop) => sum + loop.edgeCount, 0),
    };
  }

  setWireframe(enabled: boolean): void {
    this.wireframeEnabled = enabled;
    if (this.meshMaterial) {
      this.meshMaterial.wireframe = enabled;
    }
  }

  getHoleLoopSummary(): HoleLoopSummary | null {
    if (!this.holeFillMode) {
      return null;
    }

    return {
      loopCount: this.holeLoops.length,
      edgeCount: this.holeLoops.reduce((sum, loop) => sum + loop.edgeCount, 0),
    };
  }

  getBoundaryWorkflowState(): BoundaryWorkflowState {
    return {
      hasSelectedBoundary: this.activeBoundaryVertexIds !== null,
      selectedBoundaryEdgeCount: this.activeBoundaryVertexIds?.length ?? 0,
      smoothCommitted: this.boundarySmoothCommitted,
      remeshApplied: this.boundaryRemeshApplied,
      thickenApplied: this.boundaryThickenApplied,
      extrudeApplied: this.boundaryExtrudeApplied,
      hasBoundaryGuide: this.boundaryGuide !== null,
      canOffsetSelect:
        this.boundaryGuide !== null && (this.boundaryThickenApplied || this.boundaryExtrudeApplied),
      offsetApplied: this.boundaryOffsetApplied,
      selectedTriangleCount: this.selectedTriangleCount,
    };
  }

  selectHoveredBoundaryLoop(): ViewportActionResult {
    if (!this.editableMesh || this.hoveredHoleLoopIndex < 0) {
      return {
        success: false,
        message: 'Hover a clean boundary loop before targeting it.',
        stats: null,
      };
    }

    const loop = this.holeLoops[this.hoveredHoleLoopIndex];
    if (!loop || !loop.isBoundaryLoop || !loop.orderedVertexIds) {
      return {
        success: false,
        message: 'That highlighted contour is not a simple boundary loop.',
        stats: null,
      };
    }

    this.activeBoundaryLoopIndex = this.hoveredHoleLoopIndex;
    this.activeBoundaryVertexIds = loop.orderedVertexIds.slice();
    this.boundaryGuide = captureBoundaryGuide(this.editableMesh.positions, this.activeBoundaryVertexIds);
    this.clearSelection();
    this.boundaryPreviewBaseSnapshot = this.captureSessionSnapshot();
    this.boundaryThickenPreviewBaseSnapshot = null;
    this.boundaryExtrudePreviewBaseSnapshot = null;
    this.boundaryFinalSmoothPreviewBaseSnapshot = null;
    this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
    this.boundarySmoothCommitted = false;
    this.boundaryRemeshApplied = false;
    this.boundaryThickenApplied = false;
    this.boundaryExtrudeApplied = false;
    this.boundaryOffsetApplied = false;
    this.updateHoleHoverOverlay();
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: `Targeted a boundary loop with ${loop.edgeCount} edges. Adjust the boundary smooth sliders, then press Done Boundary Smooth.`,
      stats: {
        vertexCount: this.editableMesh.vertexCount,
        triangleCount: this.editableMesh.triangleCount,
        boundsRadius: this.editableMesh.boundsRadius,
      },
    };
  }

  previewBoundarySmooth(intensity: number, iterations: number): ViewportActionResult {
    if (!this.editableMesh || !this.activeBoundaryVertexIds || !this.boundaryPreviewBaseSnapshot?.positions) {
      return {
        success: false,
        message: 'Target a boundary loop before previewing boundary smooth.',
        stats: null,
      };
    }

    this.editableMesh.positions.set(this.boundaryPreviewBaseSnapshot.positions);
    recomputeAllNormals(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.faceNormals,
      this.editableMesh.normals,
      this.editableMesh.vertexFaceOffsets,
      this.editableMesh.vertexFaces,
    );
    const preview = smoothBoundaryLoopVertices(
      this.boundaryPreviewBaseSnapshot.positions,
      this.editableMesh.normals,
      this.activeBoundaryVertexIds,
      intensity,
      iterations,
    );
    if (!preview) {
      return {
        success: false,
        message: 'Boundary smooth preview could not be generated for that loop.',
        stats: null,
      };
    }

    this.applyPositionsInPlace(preview);
    this.boundarySmoothCommitted = false;
    this.boundaryRemeshApplied = false;
    this.updateHoleLoopBaseOverlay();
    this.updateHoleHoverOverlay();
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: `Previewing boundary smooth at ${intensity.toFixed(2)} for ${Math.max(1, Math.round(iterations))} iterations.`,
      stats: {
        vertexCount: this.editableMesh.vertexCount,
        triangleCount: this.editableMesh.triangleCount,
        boundsRadius: this.editableMesh.boundsRadius,
      },
    };
  }

  commitBoundarySmooth(): ViewportActionResult {
    if (!this.editableMesh || !this.activeBoundaryVertexIds) {
      return {
        success: false,
        message: 'Target a boundary loop before finishing the boundary smooth stage.',
        stats: null,
      };
    }

    const beforeSnapshot = this.boundaryPreviewBaseSnapshot;
    const afterSnapshot = this.captureSessionSnapshot();
    if (beforeSnapshot?.positions && !floatArraysEqual(beforeSnapshot.positions, afterSnapshot.positions)) {
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.emitHistory();
    }

    this.boundaryGuide = captureBoundaryGuide(this.editableMesh.positions, this.activeBoundaryVertexIds);
    this.boundaryPreviewBaseSnapshot = this.captureSessionSnapshot();
    this.boundaryThickenPreviewBaseSnapshot = null;
    this.boundaryExtrudePreviewBaseSnapshot = null;
    this.boundaryFinalSmoothPreviewBaseSnapshot = null;
    this.boundarySmoothCommitted = true;
    this.boundaryRemeshApplied = false;
    this.boundaryThickenApplied = false;
    this.boundaryExtrudeApplied = false;
    this.boundaryOffsetApplied = false;
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: 'Boundary smooth committed. You can now run the fixed-boundary remesh stage.',
      stats: {
        vertexCount: this.editableMesh.vertexCount,
        triangleCount: this.editableMesh.triangleCount,
        boundsRadius: this.editableMesh.boundsRadius,
      },
    };
  }

  applyBoundaryFixedRemesh(targetEdgeSize: number): ViewportActionResult {
    if (!this.boundaryGuide) {
      return {
        success: false,
        message: 'Target a boundary loop and finish the boundary smooth stage first.',
        stats: null,
      };
    }

    if (!this.boundarySmoothCommitted) {
      this.commitBoundarySmooth();
    }

    const guide = this.boundaryGuide.slice();
    const result = this.applySurfaceRemesh(targetEdgeSize, 'fixed');
    if (!result.success) {
      return result;
    }

    this.boundaryGuide = guide;
    this.boundaryPreviewBaseSnapshot = null;
    this.boundaryThickenPreviewBaseSnapshot = this.captureSessionSnapshot();
    this.boundaryExtrudePreviewBaseSnapshot = this.captureSessionSnapshot();
    this.boundaryFinalSmoothPreviewBaseSnapshot = null;
    this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
    this.activeBoundaryLoopIndex = -1;
    this.activeBoundaryVertexIds = this.editableMesh
      ? this.resolveBoundaryLoopVertexIds(this.editableMesh, guide)
      : null;
    if (this.editableMesh && this.activeBoundaryVertexIds) {
      this.boundaryGuide = captureBoundaryGuide(this.editableMesh.positions, this.activeBoundaryVertexIds);
    }
    this.boundarySmoothCommitted = true;
    this.boundaryRemeshApplied = true;
    this.boundaryThickenApplied = false;
    this.boundaryExtrudeApplied = false;
    this.boundaryOffsetApplied = false;
    this.updateHoleHoverOverlay();
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: `Boundary-stage remesh applied at ${targetEdgeSize.toFixed(3)} mm with fixed boundaries.`,
      stats: result.stats,
    };
  }

  previewBoundaryThicken(thickness: number): ViewportActionResult {
    if (!this.boundaryGuide || !this.boundaryRemeshApplied) {
      return {
        success: false,
        message: 'Run the fixed-boundary remesh before previewing boundary thicken.',
        stats: null,
      };
    }

    this.boundaryThickenPreviewBaseSnapshot ??= this.captureSessionSnapshot();
    const baseSnapshot = this.boundaryThickenPreviewBaseSnapshot;
    if (!baseSnapshot.positions || !baseSnapshot.indices || !baseSnapshot.referencePositions) {
      return {
        success: false,
        message: 'Boundary thicken preview is missing its base mesh snapshot.',
        stats: null,
      };
    }

    const viewState = this.captureViewState();
    const boundaryState = this.captureBoundarySessionState();
    try {
      const { editable } = this.createEditableFromSnapshot(baseSnapshot);
      const thickened = thickenMesh(editable, thickness);
      thickened.geometry.computeBoundsTree({
        maxLeafSize: 20,
        setBoundingBox: false,
        indirect: true,
      });

      const previewEditable = createEditableMeshData(thickened.geometry);
      const previewEngine = new SculptEngine(previewEditable, SCULPT_HISTORY_LIMIT);
      this.installSession(previewEditable, previewEngine, {
        sessionId: this.currentSessionId,
        resetActionHistory: false,
        resetView: false,
      });
      this.restoreBoundarySessionState({
        ...boundaryState,
        remeshApplied: true,
        thickenApplied: false,
        extrudeApplied: false,
        offsetApplied: false,
      });
      this.boundaryThickenPreviewBaseSnapshot = baseSnapshot;
      this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
      this.restoreViewState(viewState);
    } catch (error) {
      console.error(error);
      return {
        success: false,
        message:
          error instanceof Error ? error.message : 'Boundary thicken preview failed on the current mesh.',
        stats: null,
      };
    }

    return {
      success: true,
      message: `Previewing boundary thicken at ${thickness.toFixed(3)} mm.`,
      stats: {
        vertexCount: this.editableMesh?.vertexCount ?? 0,
        triangleCount: this.editableMesh?.triangleCount ?? 0,
        boundsRadius: this.editableMesh?.boundsRadius ?? 0,
      },
    };
  }

  commitBoundaryThicken(thickness: number): ViewportActionResult {
    const previewResult = this.previewBoundaryThicken(thickness);
    if (!previewResult.success) {
      return previewResult;
    }

    const beforeSnapshot = this.boundaryThickenPreviewBaseSnapshot;
    const afterSnapshot = this.captureSessionSnapshot();
    if (beforeSnapshot?.positions && !floatArraysEqual(beforeSnapshot.positions, afterSnapshot.positions)) {
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.emitHistory();
    }

    const guide = this.boundaryGuide?.slice() ?? null;
    this.boundaryGuide = guide;
    this.boundarySmoothCommitted = true;
    this.boundaryRemeshApplied = true;
    this.boundaryThickenApplied = true;
    this.boundaryExtrudeApplied = false;
    this.boundaryOffsetApplied = false;
    this.boundaryFinalSmoothPreviewBaseSnapshot = null;
    this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: `Boundary-stage thicken applied with ${thickness.toFixed(3)} mm of shell offset.`,
      stats: {
        vertexCount: this.editableMesh?.vertexCount ?? 0,
        triangleCount: this.editableMesh?.triangleCount ?? 0,
        boundsRadius: this.editableMesh?.boundsRadius ?? 0,
      },
    };
  }

  previewBoundaryExtrude(distance: number): ViewportActionResult {
    if (!this.boundaryGuide || !this.boundaryRemeshApplied) {
      return {
        success: false,
        message: 'Run the fixed-boundary remesh before previewing a positive socket extrusion.',
        stats: null,
      };
    }

    this.boundaryExtrudePreviewBaseSnapshot ??= this.captureSessionSnapshot();
    const baseSnapshot = this.boundaryExtrudePreviewBaseSnapshot;
    if (!baseSnapshot.positions || !baseSnapshot.indices || !baseSnapshot.referencePositions) {
      return {
        success: false,
        message: 'Positive socket extrusion is missing its base mesh snapshot.',
        stats: null,
      };
    }

    const viewState = this.captureViewState();
    try {
      const { editable, engine } = this.createEditableFromSnapshot(baseSnapshot);
      const loopVertexIds = this.resolveBoundaryLoopVertexIds(editable, this.boundaryGuide);
      if (!loopVertexIds) {
        return {
          success: false,
          message: 'The remeshed boundary could not be matched back to a clean loop for extrusion.',
          stats: null,
        };
      }

      const extruded = extrudeBoundaryLoop(editable, loopVertexIds, distance);
      extruded.geometry.computeBoundsTree({
        maxLeafSize: 20,
        setBoundingBox: false,
        indirect: true,
      });

      const previewEditable = createEditableMeshData(extruded.geometry);
      const previewEngine = new SculptEngine(previewEditable, SCULPT_HISTORY_LIMIT);
      this.installSession(previewEditable, previewEngine, {
        sessionId: this.currentSessionId,
        resetActionHistory: false,
        resetView: false,
      });
      this.activeBoundaryLoopIndex = -1;
      this.activeBoundaryVertexIds = extruded.outerVertexIds.slice();
      this.boundaryGuide = captureBoundaryGuide(previewEditable.positions, this.activeBoundaryVertexIds);
      this.boundarySmoothCommitted = true;
      this.boundaryRemeshApplied = true;
      this.boundaryThickenApplied = false;
      this.boundaryExtrudeApplied = false;
      this.boundaryOffsetApplied = false;
      this.boundaryExtrudePreviewBaseSnapshot = baseSnapshot;
      this.boundaryFinalSmoothPreviewBaseSnapshot = null;
      this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
      this.restoreViewState(viewState);
      this.updateHoleHoverOverlay();
      this.emitBoundaryWorkflow();
    } catch (error) {
      console.error(error);
      return {
        success: false,
        message:
          error instanceof Error ? error.message : 'Positive socket extrusion preview failed on the current mesh.',
        stats: null,
      };
    }

    return {
      success: true,
      message: `Previewing a positive socket extrusion at ${distance.toFixed(3)} mm.`,
      stats: {
        vertexCount: this.editableMesh?.vertexCount ?? 0,
        triangleCount: this.editableMesh?.triangleCount ?? 0,
        boundsRadius: this.editableMesh?.boundsRadius ?? 0,
      },
    };
  }

  commitBoundaryExtrude(distance: number): ViewportActionResult {
    const previewResult = this.previewBoundaryExtrude(distance);
    if (!previewResult.success) {
      return previewResult;
    }

    const beforeSnapshot = this.boundaryExtrudePreviewBaseSnapshot;
    const afterSnapshot = this.captureSessionSnapshot();
    if (beforeSnapshot?.positions && !floatArraysEqual(beforeSnapshot.positions, afterSnapshot.positions)) {
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.emitHistory();
    }

    this.boundarySmoothCommitted = true;
    this.boundaryRemeshApplied = true;
    this.boundaryThickenApplied = false;
    this.boundaryExtrudeApplied = true;
    this.boundaryOffsetApplied = false;
    this.boundaryFinalSmoothPreviewBaseSnapshot = null;
    this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: `Positive socket extrusion applied at ${distance.toFixed(3)} mm.`,
      stats: {
        vertexCount: this.editableMesh?.vertexCount ?? 0,
        triangleCount: this.editableMesh?.triangleCount ?? 0,
        boundsRadius: this.editableMesh?.boundsRadius ?? 0,
      },
    };
  }

  previewBoundaryBand(distanceMm: number): ViewportActionResult {
    if (
      !this.editableMesh ||
      !this.boundaryGuide ||
      (!this.boundaryThickenApplied && !this.boundaryExtrudeApplied)
    ) {
      return {
        success: false,
        message: 'Run the shell or positive extrusion stage before applying the offset stage.',
        stats: null,
      };
    }

    const nextSelectionMask = selectTrianglesNearBoundaryGuide(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.boundaryGuide,
      distanceMm,
    );
    const nextCount = countSelectedTriangles(nextSelectionMask);
    if (nextCount === 0) {
      return {
        success: false,
        message: 'No faces were found inside that boundary offset band.',
        stats: null,
      };
    }

    this.selectedTriangleMask = nextSelectionMask;
    this.selectedTriangleCount = nextCount;
    this.selectionDirty = true;
    this.boundaryOffsetApplied = false;
    this.emitSelection();
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: `Offset band preview selected ${nextCount} faces inside ${distanceMm.toFixed(3)} mm of the stored boundary.`,
      stats: {
        vertexCount: this.editableMesh.vertexCount,
        triangleCount: this.editableMesh.triangleCount,
        boundsRadius: this.editableMesh.boundsRadius,
      },
    };
  }

  selectBoundaryBand(distanceMm: number): ViewportActionResult {
    const preview = this.previewBoundaryBand(distanceMm);
    if (!preview.success) {
      return preview;
    }

    this.boundaryOffsetApplied = true;
    this.emitBoundaryWorkflow();
    return {
      ...preview,
      message: `Offset band applied with ${distanceMm.toFixed(3)} mm.`,
    };
  }

  previewBoundaryFinalSmooth(intensity: number): ViewportActionResult {
    if (!this.editableMesh || !this.selectedTriangleMask || this.selectedTriangleCount === 0) {
      return {
        success: false,
        message: 'Apply the offset stage before previewing the final smooth.',
        stats: null,
      };
    }

    this.boundaryFinalSmoothPreviewBaseSnapshot ??= this.captureSessionSnapshot();
    const baseSnapshot = this.boundaryFinalSmoothPreviewBaseSnapshot;
    if (!baseSnapshot.positions || !baseSnapshot.indices || !baseSnapshot.referencePositions || !baseSnapshot.selectedTriangleMask) {
      return {
        success: false,
        message: 'Final smooth preview is missing its selected-band snapshot.',
        stats: null,
      };
    }

    const viewState = this.captureViewState();
    const boundaryState = this.captureBoundarySessionState();
    const selectedTriangleCount = baseSnapshot.selectedTriangleCount;
    try {
      const { editable, engine } = this.createEditableFromSnapshot(baseSnapshot);
      const smoothed = laplacianSmoothSelected(
        editable.positions,
        editable.indices,
        editable.referencePositions,
        baseSnapshot.selectedTriangleMask,
        intensity,
        40,
        { preserveOpenBoundaryVertices: true },
      );
      if (!smoothed) {
        return {
          success: false,
          message: 'Final smooth preview could not find any selected band vertices to relax.',
          stats: null,
        };
      }

      const previewGeometry = createGeometryFromMeshArrays(smoothed.positions, smoothed.indices);
      previewGeometry.computeBoundsTree({
        maxLeafSize: 20,
        setBoundingBox: false,
        indirect: true,
      });
      const previewEditable = createEditableMeshData(previewGeometry, {
        referencePositions: smoothed.referencePositions,
      });
      const previewEngine = new SculptEngine(previewEditable, SCULPT_HISTORY_LIMIT);
      this.installSession(previewEditable, previewEngine, {
        sessionId: this.currentSessionId,
        resetActionHistory: false,
        resetView: false,
        selectedTriangleMask: smoothed.selectedTriangleMask,
        selectedTriangleCount,
      });
      this.restoreBoundarySessionState(boundaryState);
      this.boundaryFinalSmoothPreviewBaseSnapshot = baseSnapshot;
      this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
      this.restoreViewState(viewState);
    } catch (error) {
      console.error(error);
      return {
        success: false,
        message:
          error instanceof Error ? error.message : 'Final smooth preview failed on the selected band.',
        stats: null,
      };
    }

    return {
      success: true,
      message: `Previewing the final selected-band smooth at ${intensity.toFixed(2)} with 40 iterations.`,
      stats: {
        vertexCount: this.editableMesh?.vertexCount ?? 0,
        triangleCount: this.editableMesh?.triangleCount ?? 0,
        boundsRadius: this.editableMesh?.boundsRadius ?? 0,
      },
    };
  }

  commitBoundaryFinalSmooth(intensity: number): ViewportActionResult {
    const previewResult = this.previewBoundaryFinalSmooth(intensity);
    if (!previewResult.success) {
      return previewResult;
    }

    const beforeSnapshot = this.boundaryFinalSmoothPreviewBaseSnapshot;
    const afterSnapshot = this.captureSessionSnapshot();
    if (beforeSnapshot?.positions && !floatArraysEqual(beforeSnapshot.positions, afterSnapshot.positions)) {
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.emitHistory();
    }

    this.boundaryFinalSmoothPreviewBaseSnapshot = this.captureSessionSnapshot();
    this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: `Final selected-band smooth applied at ${intensity.toFixed(2)} with 40 iterations.`,
      stats: {
        vertexCount: this.editableMesh?.vertexCount ?? 0,
        triangleCount: this.editableMesh?.triangleCount ?? 0,
        boundsRadius: this.editableMesh?.boundsRadius ?? 0,
      },
    };
  }

  previewBoundaryDirectionalExtrude(rotateXDegrees: number, rotateYDegrees: number): ViewportActionResult {
    if (!this.boundaryGuide) {
      return {
        success: false,
        message: 'Finish the positive socket smoothing stages before previewing the final wall extrusion.',
        stats: null,
      };
    }

    this.boundaryDirectionalExtrudePreviewBaseSnapshot ??= this.captureSessionSnapshot();
    const baseSnapshot = this.boundaryDirectionalExtrudePreviewBaseSnapshot;
    if (!baseSnapshot.positions || !baseSnapshot.indices || !baseSnapshot.referencePositions) {
      return {
        success: false,
        message: 'Directional boundary extrusion is missing its base mesh snapshot.',
        stats: null,
      };
    }

    const viewState = this.captureViewState();
    const boundaryState = this.captureBoundarySessionState();
    try {
      const { editable } = this.createEditableFromSnapshot(baseSnapshot);
      const loopVertexIds = this.resolveBoundaryLoopVertexIds(editable, this.boundaryGuide);
      if (!loopVertexIds) {
        return {
          success: false,
          message: 'The current outer boundary could not be matched back to a clean loop for the final wall extrusion.',
          stats: null,
        };
      }

      const direction = this.computeBoundaryDirectionalExtrudeDirection(
        editable,
        loopVertexIds,
        rotateXDegrees,
        rotateYDegrees,
      );
      const distance = this.computeLargestBoundingBoxDimension(editable);
      const extruded = extrudeBoundaryLoopAlongVector(editable, loopVertexIds, direction, distance);
      extruded.geometry.computeBoundsTree({
        maxLeafSize: 20,
        setBoundingBox: false,
        indirect: true,
      });

      const previewEditable = createEditableMeshData(extruded.geometry);
      const previewEngine = new SculptEngine(previewEditable, SCULPT_HISTORY_LIMIT);
      this.installSession(previewEditable, previewEngine, {
        sessionId: this.currentSessionId,
        resetActionHistory: false,
        resetView: false,
      });
      this.activeBoundaryLoopIndex = -1;
      this.activeBoundaryVertexIds = extruded.outerVertexIds.slice();
      this.boundaryGuide = captureBoundaryGuide(previewEditable.positions, this.activeBoundaryVertexIds);
      this.boundarySmoothCommitted = true;
      this.boundaryRemeshApplied = true;
      this.boundaryThickenApplied = false;
      this.boundaryExtrudeApplied = true;
      this.boundaryOffsetApplied = true;
      this.boundaryFinalSmoothPreviewBaseSnapshot = null;
      this.boundaryDirectionalExtrudePreviewBaseSnapshot = baseSnapshot;
      this.restoreViewState(viewState);
      this.updateHoleHoverOverlay();
      this.emitBoundaryWorkflow();
    } catch (error) {
      console.error(error);
      return {
        success: false,
        message:
          error instanceof Error ? error.message : 'Directional boundary extrusion preview failed on the current mesh.',
        stats: null,
      };
    }

    return {
      success: true,
      message: `Previewing the final wall extrusion with X ${rotateXDegrees.toFixed(1)}° and Y ${rotateYDegrees.toFixed(1)}° tilt.`,
      stats: {
        vertexCount: this.editableMesh?.vertexCount ?? 0,
        triangleCount: this.editableMesh?.triangleCount ?? 0,
        boundsRadius: this.editableMesh?.boundsRadius ?? 0,
      },
    };
  }

  commitBoundaryDirectionalExtrude(rotateXDegrees: number, rotateYDegrees: number): ViewportActionResult {
    const previewResult = this.previewBoundaryDirectionalExtrude(rotateXDegrees, rotateYDegrees);
    if (!previewResult.success) {
      return previewResult;
    }

    const beforeSnapshot = this.boundaryDirectionalExtrudePreviewBaseSnapshot;
    const afterSnapshot = this.captureSessionSnapshot();
    if (beforeSnapshot?.positions && !floatArraysEqual(beforeSnapshot.positions, afterSnapshot.positions)) {
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.emitHistory();
    }

    this.boundaryDirectionalExtrudePreviewBaseSnapshot = this.captureSessionSnapshot();
    this.emitBoundaryWorkflow();

    return {
      success: true,
      message: `Final wall extrusion applied with X ${rotateXDegrees.toFixed(1)}° and Y ${rotateYDegrees.toFixed(1)}° tilt.`,
      stats: {
        vertexCount: this.editableMesh?.vertexCount ?? 0,
        triangleCount: this.editableMesh?.triangleCount ?? 0,
        boundsRadius: this.editableMesh?.boundsRadius ?? 0,
      },
    };
  }

  resetView(): void {
    if (!this.editableMesh) {
      this.controls.target.set(0, 0, 0);
      this.camera.position.set(2.8, 1.8, 3.4);
      this.camera.near = 0.01;
      this.camera.far = 1000;
      this.camera.updateProjectionMatrix();
      this.controls.update();
      return;
    }

    const radius = Math.max(this.editableMesh.boundsRadius, 0.5);
    const distance = radius / Math.tan((this.camera.fov * Math.PI) / 360) * 1.25;
    this.camera.position.set(radius * 1.35, radius * 0.9, distance);
    this.camera.near = Math.max(radius / 500, 0.001);
    this.camera.far = Math.max(radius * 25, 10);
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setSession(meshData: EditableMeshData, sculptEngine: SculptEngine): void {
    this.installSession(meshData, sculptEngine, {
      sessionId: this.allocateSessionId(),
      resetActionHistory: true,
      resetView: true,
    });
  }

  private installSession(
    meshData: EditableMeshData,
    sculptEngine: SculptEngine,
    options: SessionInstallOptions = {},
  ): void {
    this.finishStroke();
    this.finishSelectionGesture();
    this.clearSceneMesh();

    if (options.resetActionHistory) {
      this.historyUndoStack = [];
      this.historyRedoStack = [];
    }

    this.editableMesh = meshData;
    this.sculptEngine = sculptEngine;
    this.currentSessionId = options.sessionId ?? this.allocateSessionId();
    this.activeBoundaryLoopIndex = -1;
    this.activeBoundaryVertexIds = null;
    this.boundaryGuide = null;
    this.boundaryPreviewBaseSnapshot = null;
    this.boundaryThickenPreviewBaseSnapshot = null;
    this.boundaryExtrudePreviewBaseSnapshot = null;
    this.boundaryFinalSmoothPreviewBaseSnapshot = null;
    this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
    this.boundarySmoothCommitted = false;
    this.boundaryRemeshApplied = false;
    this.boundaryThickenApplied = false;
    this.boundaryExtrudeApplied = false;
    this.boundaryOffsetApplied = false;
    this.selectedTriangleMask =
      options.selectedTriangleMask && options.selectedTriangleMask.length === meshData.triangleCount
        ? options.selectedTriangleMask.slice()
        : new Uint8Array(meshData.triangleCount);
    this.selectedTriangleCount = countSelectedTriangles(this.selectedTriangleMask);
    this.selectionDirty = true;

    this.meshMaterial = new MeshMatcapMaterial({
      color: '#e8ebef',
      matcap: this.sculptMatcapTexture,
      side: DoubleSide,
      wireframe: this.wireframeEnabled,
      vertexColors: true,
    });

    this.mesh = new Mesh(meshData.geometry, this.meshMaterial);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);

    const cursorGeometry = new SphereGeometry(1, 20, 16);
    const cursorMaterial = new MeshBasicMaterial({
      color: '#8ed8ff',
      transparent: true,
      opacity: 0.24,
      side: DoubleSide,
      depthTest: false,
      depthWrite: false,
    });
    this.cursor = new Mesh(cursorGeometry, cursorMaterial);
    this.cursor.visible = false;
    this.cursor.renderOrder = 5;
    this.mesh.add(this.cursor);

    this.selectionOverlayGeometry = new BufferGeometry();
    this.selectionOverlayGeometry.setAttribute('position', meshData.positionAttribute);
    this.selectionOverlayGeometry.setAttribute('normal', meshData.normalAttribute);
    this.selectionOverlayGeometry.setIndex(new BufferAttribute(new Uint32Array(0), 1));

    const selectionMaterial = new MeshBasicMaterial({
      color: '#5b1fa5',
      transparent: true,
      opacity: 0.8,
      side: DoubleSide,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
    this.selectionOverlay = new Mesh(this.selectionOverlayGeometry, selectionMaterial);
    this.selectionOverlay.visible = false;
    this.selectionOverlay.renderOrder = 4;
    this.mesh.add(this.selectionOverlay);

    if (this.holeFillMode) {
      this.rebuildHoleLoopOverlays();
      this.updateHoleLoopOverlayVisibility();
    }

    if (options.resetView !== false) {
      this.resetView();
    }
    this.rebuildSelectionOverlay();
    this.emitHistory();
    this.emitSelection();
    this.emitBoundaryWorkflow();
    this.emitMeshStats();
    this.updateCursorVisuals();
  }

  dispose(): void {
    this.finishStroke();
    this.finishSelectionGesture();
    this.resizeObserver.disconnect();
    const dom = this.renderer.domElement;
    dom.removeEventListener('mousedown', this.handleMouseDown);
    dom.removeEventListener('auxclick', this.handleAuxClick);
    dom.removeEventListener('wheel', this.handleWheel);
    dom.removeEventListener('pointerenter', this.handlePointerEnter);
    dom.removeEventListener('pointerleave', this.handlePointerLeave);
    dom.removeEventListener('pointermove', this.handlePointerMove);
    dom.removeEventListener('pointerdown', this.handlePointerDown);
    dom.removeEventListener('pointerup', this.handlePointerUp);
    dom.removeEventListener('pointercancel', this.handlePointerUp);
    this.controls.dispose();
    this.renderer.setAnimationLoop(null);
    this.sculptMatcapTexture.dispose();
    this.renderer.dispose();
    this.clearSceneMesh();
  }

  undo(): void {
    if (this.activeStroke || this.selectionGestureActive) {
      return;
    }

    const action = this.historyUndoStack.pop();
    if (!action) {
      return;
    }

    if (action.kind === 'stroke') {
      if (!this.sculptEngine || action.sessionId !== this.currentSessionId || !this.sculptEngine.undo()) {
        this.historyUndoStack.push(action);
        return;
      }

      this.historyRedoStack.push(action);
      this.emitHistory();
      return;
    }

    const viewState = this.captureViewState();
    this.historyRedoStack.push(action);
    this.applySessionSnapshot(action.before, viewState);
  }

  redo(): void {
    if (this.activeStroke || this.selectionGestureActive) {
      return;
    }

    const action = this.historyRedoStack.pop();
    if (!action) {
      return;
    }

    if (action.kind === 'stroke') {
      if (!this.sculptEngine || action.sessionId !== this.currentSessionId || !this.sculptEngine.redo()) {
        this.historyRedoStack.push(action);
        return;
      }

      this.historyUndoStack.push(action);
      this.emitHistory();
      return;
    }

    const viewState = this.captureViewState();
    this.historyUndoStack.push(action);
    this.applySessionSnapshot(action.after, viewState);
  }

  clearSelection(): boolean {
    if (!this.selectedTriangleMask || this.selectedTriangleCount === 0) {
      return false;
    }

    const viewState = this.captureViewState();
    this.selectedTriangleMask.fill(0);
    this.selectedTriangleCount = 0;
    this.selectionDirty = true;
    this.boundaryOffsetApplied = false;
    this.emitSelection();
    this.restoreViewState(viewState);
    return true;
  }

  deleteSelection(): MeshStats | null {
    if (!this.editableMesh || !this.selectedTriangleMask || this.selectedTriangleCount === 0) {
      return null;
    }

    this.sculptEngine?.discardRedoHistory();
    const beforeSnapshot = this.captureSessionSnapshot();
    const viewState = this.captureViewState();
    const nextMesh = createGeometryWithoutSelectedTriangles(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.referencePositions,
      this.selectedTriangleMask,
    );

    if (!nextMesh.geometry || !nextMesh.referencePositions) {
      const afterSnapshot = createEmptySessionSnapshot(this.allocateSessionId());
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.clearCurrentSession(afterSnapshot.sessionId);
      this.restoreViewState(viewState);
      this.emitHistory();
      this.emitSelection();
      this.emitMeshStats();
      return {
        vertexCount: 0,
        triangleCount: 0,
        boundsRadius: 0,
      };
    }

    nextMesh.geometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });

    const editable = createEditableMeshData(nextMesh.geometry, {
      referencePositions: nextMesh.referencePositions,
    });
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    const nextSessionId = this.allocateSessionId();
    const afterSnapshot = {
      sessionId: nextSessionId,
      positions: editable.positions.slice(),
      indices: editable.indices.slice(),
      referencePositions: editable.referencePositions.slice(),
      history: engine.exportHistorySnapshot(),
      selectedTriangleMask: new Uint8Array(editable.triangleCount),
      selectedTriangleCount: 0,
    } satisfies SessionSnapshot;
    this.pushHistoryAction({
      kind: 'session',
      before: beforeSnapshot,
      after: afterSnapshot,
    });
    this.installSession(editable, engine, {
      sessionId: nextSessionId,
      resetActionHistory: false,
      resetView: false,
    });
    this.restoreViewState(viewState);

    return {
      vertexCount: editable.vertexCount,
      triangleCount: editable.triangleCount,
      boundsRadius: editable.boundsRadius,
    };
  }

  refineSelection(): ViewportActionResult {
    if (!this.editableMesh || !this.selectedTriangleMask || this.selectedTriangleCount === 0) {
      return {
        success: false,
        message: 'Select some faces before using Refine.',
        stats: null,
      };
    }

    this.sculptEngine?.discardRedoHistory();
    const beforeSnapshot = this.captureSessionSnapshot();
    const viewState = this.captureViewState();
    const refined = refineSelectedTriangles(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.referencePositions,
      this.selectedTriangleMask,
    );
    if (!refined) {
      return {
        success: false,
        message: 'Refine could not create any new triangles from the current selection.',
        stats: null,
      };
    }

    const geometry = createGeometryFromMeshArrays(refined.positions, refined.indices);
    geometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });

    const editable = createEditableMeshData(geometry, {
      referencePositions: refined.referencePositions,
    });
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    const nextSessionId = this.allocateSessionId();
    const selectedTriangleCount = countSelectedTriangles(refined.selectedTriangleMask);
    const afterSnapshot = {
      sessionId: nextSessionId,
      positions: editable.positions.slice(),
      indices: editable.indices.slice(),
      referencePositions: editable.referencePositions.slice(),
      history: engine.exportHistorySnapshot(),
      selectedTriangleMask: refined.selectedTriangleMask.slice(),
      selectedTriangleCount,
    } satisfies SessionSnapshot;
    this.pushHistoryAction({
      kind: 'session',
      before: beforeSnapshot,
      after: afterSnapshot,
    });
    this.installSession(editable, engine, {
      sessionId: nextSessionId,
      resetActionHistory: false,
      resetView: false,
      selectedTriangleMask: refined.selectedTriangleMask,
      selectedTriangleCount,
    });
    this.restoreViewState(viewState);

    return {
      success: true,
      message: `Refined ${selectedTriangleCount.toLocaleString()} selected triangles into a denser local patch.`,
      stats: {
        vertexCount: editable.vertexCount,
        triangleCount: editable.triangleCount,
        boundsRadius: editable.boundsRadius,
      },
    };
  }

  remeshSelection(targetEdgeSize: number): ViewportActionResult {
    if (!this.editableMesh || !this.selectedTriangleMask || this.selectedTriangleCount === 0) {
      return {
        success: false,
        message: 'Select some faces before using Remesh Selected.',
        stats: null,
      };
    }

    this.sculptEngine?.discardRedoHistory();
    const beforeSnapshot = this.captureSessionSnapshot();
    const viewState = this.captureViewState();
    const boundaryState =
      this.interactionMode === 'boundary' || this.interactionMode === 'positive'
        ? this.captureBoundarySessionState()
        : null;
    try {
      const remeshed = remeshSelectedTriangles(
        this.editableMesh.positions,
        this.editableMesh.indices,
        this.editableMesh.referencePositions,
        this.selectedTriangleMask,
        targetEdgeSize,
      );
      if (!remeshed) {
        return {
          success: false,
          message: 'Selected remesh could not build a stable fixed-boundary patch from the current selection.',
          stats: null,
        };
      }

      const geometry = createGeometryFromMeshArrays(remeshed.positions, remeshed.indices);
      orientGeometryOutward(geometry);
      geometry.computeBoundsTree({
        maxLeafSize: 20,
        setBoundingBox: false,
        indirect: true,
      });

      const editable = createEditableMeshData(geometry, {
        referencePositions: remeshed.referencePositions,
      });
      const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
      const nextSessionId = this.allocateSessionId();
      const selectedTriangleCount = countSelectedTriangles(remeshed.selectedTriangleMask);
      const afterSnapshot = {
        sessionId: nextSessionId,
        positions: editable.positions.slice(),
        indices: editable.indices.slice(),
        referencePositions: editable.referencePositions.slice(),
        history: engine.exportHistorySnapshot(),
        selectedTriangleMask: remeshed.selectedTriangleMask.slice(),
        selectedTriangleCount,
      } satisfies SessionSnapshot;
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.installSession(editable, engine, {
        sessionId: nextSessionId,
        resetActionHistory: false,
        resetView: false,
        selectedTriangleMask: remeshed.selectedTriangleMask,
        selectedTriangleCount,
      });
      if (boundaryState?.offsetApplied) {
        this.restoreBoundarySessionState(boundaryState);
      }
      this.restoreViewState(viewState);

      return {
        success: true,
        message: remeshed.clamped
          ? `Selected remesh applied at ${remeshed.effectiveEdgeSize.toFixed(3)} mm after clamping the target size (fixed boundary).`
          : `Selected remesh applied at ${remeshed.effectiveEdgeSize.toFixed(3)} mm with ${remeshed.iterations} passes (fixed boundary).`,
        stats: {
          vertexCount: editable.vertexCount,
          triangleCount: editable.triangleCount,
          boundsRadius: editable.boundsRadius,
        },
      };
    } catch (error) {
      console.error(error);
      return {
        success: false,
        message:
          error instanceof Error
            ? error.message
            : 'Selected remesh failed on the current selection.',
        stats: null,
      };
    }
  }

  smoothSelection(intensity: number, iterations: number): ViewportActionResult {
    if (!this.editableMesh || !this.selectedTriangleMask || this.selectedTriangleCount === 0) {
      return {
        success: false,
        message: 'Select some faces before using Smooth.',
        stats: null,
      };
    }

    this.sculptEngine?.discardRedoHistory();
    const beforeSnapshot = this.captureSessionSnapshot();
    const viewState = this.captureViewState();
    const smoothed = laplacianSmoothSelected(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.referencePositions,
      this.selectedTriangleMask,
      intensity,
      iterations,
    );
    if (!smoothed) {
      return {
        success: false,
        message: 'Smooth could not find any selected vertices to relax.',
        stats: null,
      };
    }

    this.editableMesh.positions.set(smoothed.positions);
    recomputeAllNormals(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.faceNormals,
      this.editableMesh.normals,
      this.editableMesh.vertexFaceOffsets,
      this.editableMesh.vertexFaces,
    );
    recomputeDisplacementColorsRange(
      this.editableMesh.positions,
      this.editableMesh.referencePositions,
      this.editableMesh.normals,
      this.editableMesh.colors,
      0,
      this.editableMesh.vertexCount,
    );
    this.editableMesh.positionAttribute.needsUpdate = true;
    this.editableMesh.normalAttribute.needsUpdate = true;
    this.editableMesh.colorAttribute.needsUpdate = true;
    this.editableMesh.geometry.computeBoundingBox();
    this.editableMesh.geometry.computeBoundingSphere();
    this.editableMesh.boundsRadius = this.editableMesh.geometry.boundingSphere?.radius ?? this.editableMesh.boundsRadius;
    (
      this.editableMesh.geometry as BufferGeometry & {
        boundsTree?: { refit?: () => void };
      }
    ).boundsTree?.refit?.();
    this.selectedTriangleMask = smoothed.selectedTriangleMask.slice();
    this.selectedTriangleCount = countSelectedTriangles(this.selectedTriangleMask);
    this.selectionDirty = true;
    const afterSnapshot = this.captureSessionSnapshot();
    this.pushHistoryAction({
      kind: 'session',
      before: beforeSnapshot,
      after: afterSnapshot,
    });
    this.restoreViewState(viewState);
    this.emitHistory();
    this.emitSelection();
    this.emitMeshStats();

    return {
      success: true,
      message: `Smoothed the selected region with intensity ${intensity.toFixed(2)} for ${Math.max(1, Math.round(iterations))} iterations.`,
      stats: {
        vertexCount: this.editableMesh.vertexCount,
        triangleCount: this.editableMesh.triangleCount,
        boundsRadius: this.editableMesh.boundsRadius,
      },
    };
  }

  smoothSelectionBoundary(intensity: number, iterations: number): ViewportActionResult {
    if (!this.editableMesh || !this.selectedTriangleMask || this.selectedTriangleCount === 0) {
      return {
        success: false,
        message: 'Select some faces before using Smooth Boundary.',
        stats: null,
      };
    }

    this.sculptEngine?.discardRedoHistory();
    const beforeSnapshot = this.captureSessionSnapshot();
    const viewState = this.captureViewState();
    const smoothed = laplacianSmoothSelectionBoundary(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.referencePositions,
      this.selectedTriangleMask,
      intensity,
      iterations,
    );
    if (!smoothed) {
      return {
        success: false,
        message: 'Smooth Boundary could not find a clean selected border to relax.',
        stats: null,
      };
    }

    this.editableMesh.positions.set(smoothed.positions);
    recomputeAllNormals(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.faceNormals,
      this.editableMesh.normals,
      this.editableMesh.vertexFaceOffsets,
      this.editableMesh.vertexFaces,
    );
    recomputeDisplacementColorsRange(
      this.editableMesh.positions,
      this.editableMesh.referencePositions,
      this.editableMesh.normals,
      this.editableMesh.colors,
      0,
      this.editableMesh.vertexCount,
    );
    this.editableMesh.positionAttribute.needsUpdate = true;
    this.editableMesh.normalAttribute.needsUpdate = true;
    this.editableMesh.colorAttribute.needsUpdate = true;
    this.editableMesh.geometry.computeBoundingBox();
    this.editableMesh.geometry.computeBoundingSphere();
    this.editableMesh.boundsRadius = this.editableMesh.geometry.boundingSphere?.radius ?? this.editableMesh.boundsRadius;
    (
      this.editableMesh.geometry as BufferGeometry & {
        boundsTree?: { refit?: () => void };
      }
    ).boundsTree?.refit?.();
    this.selectedTriangleMask = smoothed.selectedTriangleMask.slice();
    this.selectedTriangleCount = countSelectedTriangles(this.selectedTriangleMask);
    this.selectionDirty = true;
    const afterSnapshot = this.captureSessionSnapshot();
    this.pushHistoryAction({
      kind: 'session',
      before: beforeSnapshot,
      after: afterSnapshot,
    });
    this.restoreViewState(viewState);
    this.emitHistory();
    this.emitSelection();
    this.emitMeshStats();

    return {
      success: true,
      message: `Smoothed the selected boundary with intensity ${intensity.toFixed(2)} for ${Math.max(1, Math.round(iterations))} iterations.`,
      stats: {
        vertexCount: this.editableMesh.vertexCount,
        triangleCount: this.editableMesh.triangleCount,
        boundsRadius: this.editableMesh.boundsRadius,
      },
    };
  }

  applySurfaceRemesh(targetEdgeSize: number, boundaryMode: RemeshBoundaryMode = 'refined'): ViewportActionResult {
    if (!this.editableMesh) {
      return {
        success: false,
        message: 'Load a mesh before using Remesh.',
        stats: null,
      };
    }

    this.sculptEngine?.discardRedoHistory();
    const beforeSnapshot = this.captureSessionSnapshot();
    const viewState = this.captureViewState();

    try {
      const remesh = surfaceRemeshMesh(this.editableMesh, targetEdgeSize, { boundaryMode });
      orientGeometryOutward(remesh.geometry);
      remesh.geometry.computeBoundsTree({
        maxLeafSize: 20,
        setBoundingBox: false,
        indirect: true,
      });

      const editable = createEditableMeshData(remesh.geometry);
      const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
      const nextSessionId = this.allocateSessionId();
      const afterSnapshot = this.createSessionSnapshotFromEditable(editable, engine, nextSessionId);
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.installSession(editable, engine, {
        sessionId: nextSessionId,
        resetActionHistory: false,
        resetView: false,
      });
      this.restoreViewState(viewState);

      const stats = {
        vertexCount: editable.vertexCount,
        triangleCount: editable.triangleCount,
        boundsRadius: editable.boundsRadius,
      };

      return {
        success: true,
        message: remesh.clamped
          ? `Remesh applied at ${remesh.effectiveEdgeSize.toFixed(3)} mm after clamping the target size (${boundaryMode} boundary mode).`
          : `Remesh applied at ${remesh.effectiveEdgeSize.toFixed(3)} mm with ${remesh.iterations} passes (${boundaryMode} boundary mode).`,
        stats,
      };
    } catch (error) {
      console.error(error);
      return {
        success: false,
        message:
          error instanceof Error
            ? error.message
            : 'Remesh failed on the current mesh.',
        stats: null,
      };
    }
  }

  applyThicken(thickness: number): ViewportActionResult {
    if (!this.editableMesh) {
      return {
        success: false,
        message: 'Load a mesh before using Thicken.',
        stats: null,
      };
    }

    this.sculptEngine?.discardRedoHistory();
    const beforeSnapshot = this.captureSessionSnapshot();
    const viewState = this.captureViewState();

    try {
      const thickened = thickenMesh(this.editableMesh, thickness);
      thickened.geometry.computeBoundsTree({
        maxLeafSize: 20,
        setBoundingBox: false,
        indirect: true,
      });

      const editable = createEditableMeshData(thickened.geometry);
      const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
      const nextSessionId = this.allocateSessionId();
      const afterSnapshot = this.createSessionSnapshotFromEditable(editable, engine, nextSessionId);
      this.pushHistoryAction({
        kind: 'session',
        before: beforeSnapshot,
        after: afterSnapshot,
      });
      this.installSession(editable, engine, {
        sessionId: nextSessionId,
        resetActionHistory: false,
        resetView: false,
      });
      this.restoreViewState(viewState);

      const stats = {
        vertexCount: editable.vertexCount,
        triangleCount: editable.triangleCount,
        boundsRadius: editable.boundsRadius,
      };

      return {
        success: true,
        message: `Thicken applied with ${thickness.toFixed(3)} mm of shell offset.`,
        stats,
      };
    } catch (error) {
      console.error(error);
      return {
        success: false,
        message:
          error instanceof Error ? error.message : 'Thicken failed on the current mesh.',
        stats: null,
      };
    }
  }

  private attachEvents(): void {
    const dom = this.renderer.domElement;
    dom.addEventListener('mousedown', this.handleMouseDown);
    dom.addEventListener('auxclick', this.handleAuxClick);
    dom.addEventListener('wheel', this.handleWheel, { passive: false });
    dom.addEventListener('pointerenter', this.handlePointerEnter);
    dom.addEventListener('pointerleave', this.handlePointerLeave);
    dom.addEventListener('pointermove', this.handlePointerMove);
    dom.addEventListener('pointerdown', this.handlePointerDown);
    dom.addEventListener('pointerup', this.handlePointerUp);
    dom.addEventListener('pointercancel', this.handlePointerUp);
  }

  private readonly handlePointerEnter = (event: PointerEvent): void => {
    this.pointerInside = true;
    this.updatePointerFromEvent(event);
  };

  private readonly handleMouseDown = (event: MouseEvent): void => {
    if (event.button === 1 || event.button === 2) {
      event.preventDefault();
    }
  };

  private readonly handleAuxClick = (event: MouseEvent): void => {
    if (event.button === 1) {
      event.preventDefault();
    }
  };

  private readonly handleWheel = (event: WheelEvent): void => {
    event.preventDefault();
  };

  private readonly handlePointerLeave = (): void => {
    this.pointerInside = false;
    if (!this.activeStroke && !this.selectionGestureActive && this.cursor) {
      this.cursor.visible = false;
    }
  };

  private readonly handlePointerMove = (event: PointerEvent): void => {
    this.updatePointerFromEvent(event);

    if (!this.selectionGestureActive) {
      return;
    }

    if (this.selectionTool === 'box') {
      this.selectionCurrent.copy(this.pointerClient);
    } else if (this.selectionTool === 'snip') {
      const lastPoint = this.selectionPath[this.selectionPath.length - 1];
      if (!lastPoint || lastPoint.distanceToSquared(this.pointerClient) > 4) {
        this.selectionPath.push(this.pointerClient.clone());
      } else {
        lastPoint.copy(this.pointerClient);
      }

      this.selectionCurrent.copy(this.pointerClient);
    }
  };

  private readonly handlePointerDown = (event: PointerEvent): void => {
    if (event.button !== 0) {
      return;
    }

    this.pointerDown = true;
    this.updatePointerFromEvent(event);

    if (this.holeFillMode) {
      this.debugHoleFill('pointerdown', {
        pointerInside: this.pointerInside,
        hoveredHoleLoopIndex: this.hoveredHoleLoopIndex,
        loopCount: this.holeLoops.length,
        pointerX: this.pointerClient.x,
        pointerY: this.pointerClient.y,
      });
      this.updateHoleLoopHover();
      if (this.interactionMode === 'fill') {
        this.fillHoveredHoleLoop();
      } else if (this.interactionMode === 'boundary' || this.interactionMode === 'positive') {
        const result = this.selectHoveredBoundaryLoop();
        this.callbacks.onBoundaryAction?.({
          success: result.success,
          message: result.message,
        });
      }
      event.preventDefault();
      return;
    }

    if (this.interactionMode !== 'sculpt' && this.interactionMode !== 'select') {
      return;
    }

    if (event.altKey || !this.editableMesh || !this.sculptEngine) {
      return;
    }

    if (this.interactionMode === 'sculpt') {
      this.refreshHoverHit();
      if (!this.hoverHit) {
        return;
      }

      this.activeStroke = true;
      this.controls.enabled = false;
      this.renderer.domElement.setPointerCapture(event.pointerId);
      this.sculptEngine.beginStroke();
      this.applyImmediateStamp(this.hoverHit);
      this.lastStampPoint.copy(this.hoverHit.pointLocal);
      this.lastStampNormal.copy(this.hoverHit.normalLocal);
      event.preventDefault();
      return;
    }

    this.selectionOperation = resolveSelectionOperation(event, this.selectionTool);
    this.selectionGestureActive = true;
    this.controls.enabled = false;
    this.renderer.domElement.setPointerCapture(event.pointerId);

    if (this.selectionTool === 'sphere') {
      this.refreshHoverHit();
      if (!this.hoverHit) {
        this.finishSelectionGesture();
        if (this.renderer.domElement.hasPointerCapture(event.pointerId)) {
          this.renderer.domElement.releasePointerCapture(event.pointerId);
        }

        return;
      }

      this.applySphereSelectionStamp(this.hoverHit.pointLocal, this.hoverHit.faceIndex);
      this.lastStampPoint.copy(this.hoverHit.pointLocal);
      this.lastStampNormal.copy(this.hoverHit.normalLocal);
    } else {
      this.selectionStart.copy(this.pointerClient);
      this.selectionCurrent.copy(this.pointerClient);
      this.selectionPath = this.selectionTool === 'snip' ? [this.pointerClient.clone()] : [];
    }

    event.preventDefault();
  };

  private readonly handlePointerUp = (event: PointerEvent): void => {
    this.pointerDown = false;
    if (this.renderer.domElement.hasPointerCapture(event.pointerId)) {
      this.renderer.domElement.releasePointerCapture(event.pointerId);
    }

    this.finishStroke();

    if (!this.selectionGestureActive) {
      this.controls.enabled = true;
      return;
    }

    if (this.selectionTool === 'box') {
      this.selectionCurrent.copy(this.pointerClient);
      this.applyScreenSelection('box');
    } else if (this.selectionTool === 'snip') {
      if (this.selectionPath.length >= 3) {
        this.applyScreenSelection('snip');
      } else if (this.selectionOperation === 'replace') {
        this.clearSelection();
      }
    }

    this.finishSelectionGesture();
  };

  private finishStroke(): void {
    if (!this.activeStroke) {
      return;
    }

    this.activeStroke = false;
    this.controls.enabled = true;
    const record = this.sculptEngine?.endStroke() ?? null;
    if (record) {
      this.pushHistoryAction({
        kind: 'stroke',
        sessionId: this.currentSessionId,
      });
    }
    this.emitHistory();
  }

  private finishSelectionGesture(): void {
    if (!this.selectionGestureActive) {
      this.controls.enabled = true;
      this.clearOverlayCanvas();
      return;
    }

    this.selectionGestureActive = false;
    this.controls.enabled = true;
    this.selectionPath = [];
    this.clearOverlayCanvas();
  }

  private tick(): void {
    this.controls.update();
    if (this.holeFillMode) {
      this.refreshHoverHit();
      this.updateHoleLoopHover();
    } else {
      this.refreshHoverHit();

      if (this.activeStroke && this.hoverHit && this.sculptEngine) {
        this.processContinuousStroke(this.hoverHit);
      } else if (
        this.selectionGestureActive &&
        this.selectionTool === 'sphere' &&
        this.hoverHit &&
        this.sculptEngine
      ) {
        this.processContinuousSphereSelection(this.hoverHit);
      }
    }

    if (this.selectionDirty) {
      this.rebuildSelectionOverlay();
    }

    this.drawSelectionPreview();
    this.renderer.render(this.scene, this.camera);
  }

  private processContinuousStroke(hit: HoverHit): void {
    const radius = this.getBrushRadiusWorld();
    const spacing = Math.max(radius * 0.28, radius * 0.08);
    const distance = this.lastStampPoint.distanceTo(hit.pointLocal);

    if (distance < spacing) {
      return;
    }

    const steps = Math.min(16, Math.max(1, Math.floor(distance / spacing)));
    for (let step = 1; step <= steps; step += 1) {
      const alpha = step / steps;
      this.interpolatedPoint.lerpVectors(this.lastStampPoint, hit.pointLocal, alpha);
      this.interpolatedNormal.lerpVectors(this.lastStampNormal, hit.normalLocal, alpha).normalize();
      this.sculptEngine!.applyStamp({
        pointLocal: this.interpolatedPoint,
        normalLocal: this.interpolatedNormal,
        faceIndex: hit.faceIndex,
        radius,
        strength: this.brushStrength,
        type: this.brushType,
      });
    }

    this.lastStampPoint.copy(hit.pointLocal);
    this.lastStampNormal.copy(hit.normalLocal);
  }

  private processContinuousSphereSelection(hit: HoverHit): void {
    const radius = this.getSelectionRadiusWorld();
    const spacing = Math.max(radius * 0.28, radius * 0.08);
    const distance = this.lastStampPoint.distanceTo(hit.pointLocal);

    if (distance < spacing) {
      return;
    }

    const steps = Math.min(12, Math.max(1, Math.floor(distance / spacing)));
    for (let step = 1; step <= steps; step += 1) {
      const alpha = step / steps;
      this.interpolatedPoint.lerpVectors(this.lastStampPoint, hit.pointLocal, alpha);
      this.applySphereSelectionStamp(this.interpolatedPoint, hit.faceIndex);
    }

    this.lastStampPoint.copy(hit.pointLocal);
  }

  private applyImmediateStamp(hit: HoverHit): void {
    this.sculptEngine?.applyStamp({
      pointLocal: hit.pointLocal,
      normalLocal: hit.normalLocal,
      faceIndex: hit.faceIndex,
      radius: this.getBrushRadiusWorld(),
      strength: this.brushStrength,
      type: this.brushType,
    });
  }

  private applySphereSelectionStamp(pointLocal: Vector3, faceIndex: number): void {
    if (!this.sculptEngine) {
      return;
    }

    const triangleCount = this.sculptEngine.collectTrianglesInSphere(
      faceIndex,
      pointLocal,
      this.getSelectionRadiusWorld(),
    );
    this.applyTriangleSelection(
      this.sculptEngine.data.regionTriangles,
      triangleCount,
      this.selectionOperation,
    );
  }

  private applyScreenSelection(tool: SelectionTool): void {
    if (!this.editableMesh || !this.mesh || (tool !== 'box' && tool !== 'snip')) {
      return;
    }

    const triangleIds = new Uint32Array(this.editableMesh.triangleCount);
    let triangleCount = 0;

    this.camera.getWorldPosition(this.cameraWorldPosition);
    const { indices, positions, faceNormals } = this.editableMesh;
    const width = this.overlayCanvas.clientWidth;
    const height = this.overlayCanvas.clientHeight;

    for (let triangle = 0; triangle < this.editableMesh.triangleCount; triangle += 1) {
      const triOffset = triangle * 3;
      const a = indices[triOffset] * 3;
      const b = indices[triOffset + 1] * 3;
      const c = indices[triOffset + 2] * 3;

      this.triangleCentroid.set(
        (positions[a] + positions[b] + positions[c]) / 3,
        (positions[a + 1] + positions[b + 1] + positions[c + 1]) / 3,
        (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3,
      );

      this.triangleWorldPoint.copy(this.triangleCentroid);
      this.mesh.localToWorld(this.triangleWorldPoint);

      this.triangleNormal.set(
        faceNormals[triOffset],
        faceNormals[triOffset + 1],
        faceNormals[triOffset + 2],
      );
      if (this.triangleNormal.lengthSq() <= 1e-12) {
        continue;
      }

      this.triangleWorldNormal.copy(this.triangleNormal).normalize();
      this.viewDirection.copy(this.cameraWorldPosition).sub(this.triangleWorldPoint);
      if (this.triangleWorldNormal.dot(this.viewDirection) <= 0) {
        continue;
      }

      this.projectedPoint.copy(this.triangleWorldPoint).project(this.camera);
      if (this.projectedPoint.z < -1 || this.projectedPoint.z > 1) {
        continue;
      }

      const screenX = (this.projectedPoint.x * 0.5 + 0.5) * width;
      const screenY = (-this.projectedPoint.y * 0.5 + 0.5) * height;

      this.triangleWorldA.set(positions[a], positions[a + 1], positions[a + 2]);
      this.mesh.localToWorld(this.triangleWorldA);
      this.projectedPointA.copy(this.triangleWorldA).project(this.camera);

      this.triangleWorldB.set(positions[b], positions[b + 1], positions[b + 2]);
      this.mesh.localToWorld(this.triangleWorldB);
      this.projectedPointB.copy(this.triangleWorldB).project(this.camera);

      this.triangleWorldC.set(positions[c], positions[c + 1], positions[c + 2]);
      this.mesh.localToWorld(this.triangleWorldC);
      this.projectedPointC.copy(this.triangleWorldC).project(this.camera);

      const inside =
        containsSelectionSample(
          tool,
          screenX,
          screenY,
          this.selectionStart,
          this.selectionCurrent,
          this.selectionPath,
        ) ||
        containsProjectedSelectionSample(
          tool,
          this.projectedPointA,
          width,
          height,
          this.selectionStart,
          this.selectionCurrent,
          this.selectionPath,
        ) ||
        containsProjectedSelectionSample(
          tool,
          this.projectedPointB,
          width,
          height,
          this.selectionStart,
          this.selectionCurrent,
          this.selectionPath,
        ) ||
        containsProjectedSelectionSample(
          tool,
          this.projectedPointC,
          width,
          height,
          this.selectionStart,
          this.selectionCurrent,
          this.selectionPath,
        );
      if (!inside) {
        continue;
      }

      this.selectionRayNdc.set(this.projectedPoint.x, this.projectedPoint.y);
      this.raycaster.setFromCamera(this.selectionRayNdc, this.camera);
      const hit = this.raycaster.intersectObject(this.mesh, false)[0];
      if (!hit || hit.faceIndex !== triangle) {
        continue;
      }

      triangleIds[triangleCount] = triangle;
      triangleCount += 1;
    }

    this.applyTriangleSelection(triangleIds, triangleCount, this.selectionOperation);
  }

  private applyTriangleSelection(
    triangleIds: Uint32Array,
    triangleCount: number,
    operation: SelectionOperation,
  ): boolean {
    if (!this.selectedTriangleMask) {
      return false;
    }

    let changed = false;
    if (operation === 'replace' && this.selectedTriangleCount > 0) {
      this.selectedTriangleMask.fill(0);
      this.selectedTriangleCount = 0;
      changed = true;
    }

    for (let i = 0; i < triangleCount; i += 1) {
      const triangle = triangleIds[i];
      if (operation === 'subtract') {
        if (this.selectedTriangleMask[triangle] !== 0) {
          this.selectedTriangleMask[triangle] = 0;
          this.selectedTriangleCount -= 1;
          changed = true;
        }
      } else if (this.selectedTriangleMask[triangle] === 0) {
        this.selectedTriangleMask[triangle] = 1;
        this.selectedTriangleCount += 1;
        changed = true;
      }
    }

    if (changed) {
      this.selectionDirty = true;
      this.emitSelection();
    }

    return changed;
  }

  private refreshHoverHit(): void {
    const showCursor =
      this.interactionMode === 'sculpt' ||
      (this.interactionMode === 'select' && this.selectionTool === 'sphere');

    if (this.holeFillMode) {
      this.hoverHit = null;
      if (this.cursor) {
        this.cursor.visible = false;
      }
      return;
    }

    if (!this.mesh || !this.sculptEngine || (!this.pointerInside && !this.pointerDown)) {
      this.hoverHit = null;
      if (this.cursor) {
        this.cursor.visible = false;
      }
      return;
    }

    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    const faceIndex = hit?.faceIndex;
    if (!hit || faceIndex == null) {
      this.hoverHit = null;
      if (this.cursor) {
        this.cursor.visible = false;
      }
      return;
    }

    this.worldHitPoint.copy(hit.point);
    this.localHitPoint.copy(this.worldHitPoint);
    this.mesh.worldToLocal(this.localHitPoint);
    this.sculptEngine.getFaceNormal(faceIndex, this.localHitNormal);

    const hoverHit =
      this.hoverHit ??
      (this.hoverHit = {
        faceIndex,
        pointLocal: new Vector3(),
        normalLocal: new Vector3(),
      });

    hoverHit.faceIndex = faceIndex;
    hoverHit.pointLocal.copy(this.localHitPoint);
    hoverHit.normalLocal.copy(this.localHitNormal);

    if (showCursor) {
      const radius =
        this.interactionMode === 'sculpt'
          ? this.getBrushRadiusWorld()
          : this.getSelectionRadiusWorld();
      this.updateCursor(hoverHit.pointLocal, radius);
    } else if (this.cursor) {
      this.cursor.visible = false;
    }
  }

  private updateCursor(center: Vector3, radius: number): void {
    if (!this.cursor) {
      return;
    }

    this.cursor.position.copy(center);
    this.cursor.scale.setScalar(radius);
    this.updateCursorVisuals();
    this.cursor.visible = true;
  }

  private updatePointerFromEvent(event: PointerEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    this.pointerClient.set(localX, localY);
    this.pointerNdc.set(
      (localX / rect.width) * 2 - 1,
      -(localY / rect.height) * 2 + 1,
    );
  }

  private getBrushRadiusWorld(): number {
    if (!this.editableMesh) {
      return this.brushRadiusMm;
    }

    return Math.max(this.brushRadiusMm, 0.0005);
  }

  private getSelectionRadiusWorld(): number {
    if (!this.editableMesh) {
      return this.selectionRadiusMm;
    }

    return Math.max(this.selectionRadiusMm, 0.0005);
  }

  private applyPositionsInPlace(nextPositions: Float32Array): void {
    if (!this.editableMesh) {
      return;
    }

    this.editableMesh.positions.set(nextPositions);
    recomputeAllNormals(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.faceNormals,
      this.editableMesh.normals,
      this.editableMesh.vertexFaceOffsets,
      this.editableMesh.vertexFaces,
    );
    recomputeDisplacementColorsRange(
      this.editableMesh.positions,
      this.editableMesh.referencePositions,
      this.editableMesh.normals,
      this.editableMesh.colors,
      0,
      this.editableMesh.vertexCount,
    );
    this.editableMesh.positionAttribute.needsUpdate = true;
    this.editableMesh.normalAttribute.needsUpdate = true;
    this.editableMesh.colorAttribute.needsUpdate = true;
    this.editableMesh.geometry.computeBoundingBox();
    this.editableMesh.geometry.computeBoundingSphere();
    this.editableMesh.boundsRadius =
      this.editableMesh.geometry.boundingSphere?.radius ?? this.editableMesh.boundsRadius;
    (
      this.editableMesh.geometry as BufferGeometry & {
        boundsTree?: { refit?: () => void };
      }
    ).boundsTree?.refit?.();
    this.emitMeshStats();
  }

  private resize(): void {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    const pixelRatio = Math.min(window.devicePixelRatio, 2);

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.renderer.setPixelRatio(pixelRatio);

    this.overlayCanvas.width = Math.floor(width * pixelRatio);
    this.overlayCanvas.height = Math.floor(height * pixelRatio);
    this.overlayCanvas.style.width = `${width}px`;
    this.overlayCanvas.style.height = `${height}px`;
    this.overlayContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    this.updateHoleLoopMaterialResolution(width, height);
  }

  private captureViewState(): ViewState {
    return {
      position: this.camera.position.clone(),
      target: this.controls.target.clone(),
      near: this.camera.near,
      far: this.camera.far,
      zoom: this.camera.zoom,
    };
  }

  private resolveBoundaryLoopVertexIds(
    editable: EditableMeshData,
    guide: Float32Array,
  ): Uint32Array | null {
    const loops = buildHoleLoopSet(editable.indices).loops;
    let bestLoop: Uint32Array | null = null;
    let bestScore = Infinity;

    for (let i = 0; i < loops.length; i += 1) {
      const loop = loops[i];
      if (!loop.isBoundaryLoop || !loop.orderedVertexIds) {
        continue;
      }

      const score = this.scoreBoundaryLoopAgainstGuide(
        editable.positions,
        loop.orderedVertexIds,
        guide,
      );
      if (score < bestScore) {
        bestScore = score;
        bestLoop = loop.orderedVertexIds;
      }
    }

    return bestLoop?.slice() ?? null;
  }

  private computeBoundaryDirectionalExtrudeDirection(
    editable: EditableMeshData,
    orderedVertexIds: Uint32Array,
    rotateXDegrees: number,
    rotateYDegrees: number,
  ): { x: number; y: number; z: number } {
    const coherentNormals = computeCoherentBoundaryNormals(editable.normals, orderedVertexIds);
    const baseDirection = new Vector3();
    for (let i = 0; i < coherentNormals.length; i += 3) {
      baseDirection.x += coherentNormals[i];
      baseDirection.y += coherentNormals[i + 1];
      baseDirection.z += coherentNormals[i + 2];
    }

    if (baseDirection.lengthSq() <= 1e-12) {
      baseDirection.set(0, 0, 1);
    } else {
      baseDirection.normalize();
    }

    let centerX = 0;
    let centerY = 0;
    let centerZ = 0;
    for (let i = 0; i < orderedVertexIds.length; i += 1) {
      const offset = orderedVertexIds[i] * 3;
      centerX += editable.positions[offset];
      centerY += editable.positions[offset + 1];
      centerZ += editable.positions[offset + 2];
    }

    const invCount = 1 / orderedVertexIds.length;
    centerX *= invCount;
    centerY *= invCount;
    centerZ *= invCount;

    let positiveCount = 0;
    let negativeCount = 0;
    for (let i = 0; i < orderedVertexIds.length; i += 1) {
      const offset = orderedVertexIds[i] * 3;
      const signedDistance =
        (editable.positions[offset] - centerX) * baseDirection.x +
        (editable.positions[offset + 1] - centerY) * baseDirection.y +
        (editable.positions[offset + 2] - centerZ) * baseDirection.z;
      if (signedDistance > 1e-5) {
        positiveCount += 1;
      } else if (signedDistance < -1e-5) {
        negativeCount += 1;
      }
    }

    if (positiveCount > negativeCount) {
      baseDirection.multiplyScalar(-1);
    }

    const rotateX = Math.max(-45, Math.min(45, rotateXDegrees)) * (Math.PI / 180);
    const rotateY = Math.max(-45, Math.min(45, rotateYDegrees)) * (Math.PI / 180);
    baseDirection.applyAxisAngle(new Vector3(1, 0, 0), rotateX);
    baseDirection.applyAxisAngle(new Vector3(0, 1, 0), rotateY);
    if (baseDirection.lengthSq() <= 1e-12) {
      baseDirection.set(0, 0, 1);
    } else {
      baseDirection.normalize();
    }

    return {
      x: baseDirection.x,
      y: baseDirection.y,
      z: baseDirection.z,
    };
  }

  private computeLargestBoundingBoxDimension(editable: EditableMeshData): number {
    editable.geometry.computeBoundingBox();
    const boundingBox = editable.geometry.boundingBox;
    if (!boundingBox) {
      return Math.max(editable.boundsRadius * 2, 1);
    }

    const sizeX = boundingBox.max.x - boundingBox.min.x;
    const sizeY = boundingBox.max.y - boundingBox.min.y;
    const sizeZ = boundingBox.max.z - boundingBox.min.z;
    return Math.max(sizeX, sizeY, sizeZ, 1);
  }

  private scoreBoundaryLoopAgainstGuide(
    positions: ArrayLike<number>,
    orderedVertexIds: Uint32Array,
    guide: Float32Array,
  ): number {
    if (orderedVertexIds.length === 0 || guide.length < 3) {
      return Infinity;
    }

    let guideCentroidX = 0;
    let guideCentroidY = 0;
    let guideCentroidZ = 0;
    for (let i = 0; i < guide.length; i += 3) {
      guideCentroidX += guide[i];
      guideCentroidY += guide[i + 1];
      guideCentroidZ += guide[i + 2];
    }
    const guideInvCount = 1 / (guide.length / 3);
    guideCentroidX *= guideInvCount;
    guideCentroidY *= guideInvCount;
    guideCentroidZ *= guideInvCount;

    let loopCentroidX = 0;
    let loopCentroidY = 0;
    let loopCentroidZ = 0;
    let score = 0;
    for (let i = 0; i < orderedVertexIds.length; i += 1) {
      const offset = orderedVertexIds[i] * 3;
      const x = positions[offset];
      const y = positions[offset + 1];
      const z = positions[offset + 2];
      loopCentroidX += x;
      loopCentroidY += y;
      loopCentroidZ += z;
      score += this.pointToGuideDistanceSq(x, y, z, guide);
    }

    const loopInvCount = 1 / orderedVertexIds.length;
    loopCentroidX *= loopInvCount;
    loopCentroidY *= loopInvCount;
    loopCentroidZ *= loopInvCount;

    const centroidDx = loopCentroidX - guideCentroidX;
    const centroidDy = loopCentroidY - guideCentroidY;
    const centroidDz = loopCentroidZ - guideCentroidZ;

    return score * loopInvCount + (centroidDx * centroidDx + centroidDy * centroidDy + centroidDz * centroidDz);
  }

  private pointToGuideDistanceSq(x: number, y: number, z: number, guide: Float32Array): number {
    let bestDistanceSq = Infinity;

    for (let i = 0; i < guide.length; i += 3) {
      const next = (i + 3) % guide.length;
      const distanceSq = this.pointToSegmentDistanceSq(
        x,
        y,
        z,
        guide[i],
        guide[i + 1],
        guide[i + 2],
        guide[next],
        guide[next + 1],
        guide[next + 2],
      );
      if (distanceSq < bestDistanceSq) {
        bestDistanceSq = distanceSq;
      }
    }

    return bestDistanceSq;
  }

  private pointToSegmentDistanceSq(
    px: number,
    py: number,
    pz: number,
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
  ): number {
    const abx = bx - ax;
    const aby = by - ay;
    const abz = bz - az;
    const apx = px - ax;
    const apy = py - ay;
    const apz = pz - az;
    const abLengthSq = abx * abx + aby * aby + abz * abz;
    if (abLengthSq <= 1e-12) {
      return apx * apx + apy * apy + apz * apz;
    }

    const t = Math.min(Math.max((apx * abx + apy * aby + apz * abz) / abLengthSq, 0), 1);
    const closestX = ax + abx * t;
    const closestY = ay + aby * t;
    const closestZ = az + abz * t;
    const dx = px - closestX;
    const dy = py - closestY;
    const dz = pz - closestZ;
    return dx * dx + dy * dy + dz * dz;
  }

  private captureBoundarySessionState(): BoundarySessionState {
    return {
      guide: this.boundaryGuide?.slice() ?? null,
      activeBoundaryVertexIds: this.activeBoundaryVertexIds?.slice() ?? null,
      smoothCommitted: this.boundarySmoothCommitted,
      remeshApplied: this.boundaryRemeshApplied,
      thickenApplied: this.boundaryThickenApplied,
      extrudeApplied: this.boundaryExtrudeApplied,
      offsetApplied: this.boundaryOffsetApplied,
    };
  }

  private createEditableFromSnapshot(
    snapshot: SessionSnapshot,
  ): { editable: EditableMeshData; engine: SculptEngine } {
    if (!snapshot.positions || !snapshot.indices || !snapshot.referencePositions) {
      throw new Error('The requested boundary preview snapshot is missing mesh arrays.');
    }

    const geometry = createGeometryFromMeshArrays(snapshot.positions, snapshot.indices);
    geometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });
    const editable = createEditableMeshData(geometry, {
      referencePositions: snapshot.referencePositions,
    });
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    engine.importHistorySnapshot(snapshot.history);
    return { editable, engine };
  }

  private restoreBoundarySessionState(state: BoundarySessionState): void {
    this.boundaryGuide = state.guide?.slice() ?? null;
    this.activeBoundaryVertexIds = state.activeBoundaryVertexIds?.slice() ?? null;
    this.boundarySmoothCommitted = state.smoothCommitted;
    this.boundaryRemeshApplied = state.remeshApplied;
    this.boundaryThickenApplied = state.thickenApplied;
    this.boundaryExtrudeApplied = state.extrudeApplied;
    this.boundaryOffsetApplied = state.offsetApplied;
    this.emitBoundaryWorkflow();
  }

  private restoreViewState(viewState: ViewState): void {
    this.camera.position.copy(viewState.position);
    this.camera.near = viewState.near;
    this.camera.far = viewState.far;
    this.camera.zoom = viewState.zoom;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(viewState.target);
    this.controls.update();
  }

  private allocateSessionId(): number {
    const sessionId = this.nextSessionId;
    this.nextSessionId += 1;
    return sessionId;
  }

  private pushHistoryAction(action: ViewportHistoryAction): void {
    this.historyUndoStack.push(action);
    if (this.historyUndoStack.length > ACTION_HISTORY_LIMIT) {
      this.historyUndoStack.shift();
    }

    this.historyRedoStack = [];
  }

  private captureSessionSnapshot(): SessionSnapshot {
    if (!this.editableMesh || !this.sculptEngine) {
      return createEmptySessionSnapshot(this.currentSessionId);
    }

    return {
      sessionId: this.currentSessionId,
      positions: this.editableMesh.positions.slice(),
      indices: this.editableMesh.indices.slice(),
      referencePositions: this.editableMesh.referencePositions.slice(),
      history: this.sculptEngine.exportHistorySnapshot(),
      selectedTriangleMask: this.selectedTriangleMask?.slice() ?? new Uint8Array(this.editableMesh.triangleCount),
      selectedTriangleCount: this.selectedTriangleCount,
    };
  }

  private createSessionSnapshotFromEditable(
    editable: EditableMeshData,
    engine: SculptEngine,
    sessionId: number,
  ): SessionSnapshot {
    return {
      sessionId,
      positions: editable.positions.slice(),
      indices: editable.indices.slice(),
      referencePositions: editable.referencePositions.slice(),
      history: engine.exportHistorySnapshot(),
      selectedTriangleMask: new Uint8Array(editable.triangleCount),
      selectedTriangleCount: 0,
    };
  }

  private applySessionSnapshot(snapshot: SessionSnapshot, viewState: ViewState): void {
    if (!snapshot.positions || !snapshot.indices || !snapshot.referencePositions) {
      this.clearCurrentSession(snapshot.sessionId);
      this.restoreViewState(viewState);
      this.emitHistory();
      this.emitSelection();
      this.emitMeshStats();
      return;
    }

    const geometry = createGeometryFromMeshArrays(snapshot.positions, snapshot.indices);
    geometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });

    const editable = createEditableMeshData(geometry, {
      referencePositions: snapshot.referencePositions,
    });
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    engine.importHistorySnapshot(snapshot.history);
    this.installSession(editable, engine, {
      sessionId: snapshot.sessionId,
      resetActionHistory: false,
      resetView: false,
      selectedTriangleMask: snapshot.selectedTriangleMask,
      selectedTriangleCount: snapshot.selectedTriangleCount,
    });
    this.restoreViewState(viewState);
  }

  private clearCurrentSession(sessionId: number): void {
    this.finishStroke();
    this.finishSelectionGesture();
    this.clearSceneMesh();
    this.currentSessionId = sessionId;
    this.activeBoundaryLoopIndex = -1;
    this.activeBoundaryVertexIds = null;
    this.boundaryGuide = null;
    this.boundaryPreviewBaseSnapshot = null;
    this.boundaryThickenPreviewBaseSnapshot = null;
    this.boundaryExtrudePreviewBaseSnapshot = null;
    this.boundaryFinalSmoothPreviewBaseSnapshot = null;
    this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
    this.boundarySmoothCommitted = false;
    this.boundaryRemeshApplied = false;
    this.boundaryThickenApplied = false;
    this.boundaryExtrudeApplied = false;
    this.boundaryOffsetApplied = false;
    this.selectedTriangleMask = null;
    this.selectedTriangleCount = 0;
    this.selectionDirty = false;
    this.updateCursorVisuals();
    this.emitBoundaryWorkflow();
  }

  private updateHoleLoopMaterialResolution(width: number, height: number): void {
    if (this.holeLoopOverlay) {
      this.updateSingleLineMaterialResolution(this.holeLoopOverlay.material as LineMaterial, width, height);
    }

    if (this.holeHoverOverlay) {
      this.updateSingleLineMaterialResolution(this.holeHoverOverlay.material as LineMaterial, width, height);
    }
  }

  private updateSingleLineMaterialResolution(
    material: LineMaterial,
    width = this.container.clientWidth,
    height = this.container.clientHeight,
  ): void {
    material.resolution.set(Math.max(width, 1), Math.max(height, 1));
  }

  private emitHistory(): void {
    this.callbacks.onHistoryChange?.({
      canUndo: this.historyUndoStack.length > 0,
      canRedo: this.historyRedoStack.length > 0,
      undoCount: this.historyUndoStack.length,
      redoCount: this.historyRedoStack.length,
    });
  }

  private emitSelection(): void {
    this.callbacks.onSelectionChange?.({
      selectedTriangleCount: this.selectedTriangleCount,
      canDelete: this.selectedTriangleCount > 0,
    });
    this.emitBoundaryWorkflow();
  }

  private emitBoundaryWorkflow(): void {
    this.callbacks.onBoundaryWorkflowChange?.(this.getBoundaryWorkflowState());
  }

  private emitMeshStats(): void {
    this.callbacks.onMeshStatsChange?.({
      vertexCount: this.editableMesh?.vertexCount ?? 0,
      triangleCount: this.editableMesh?.triangleCount ?? 0,
      boundsRadius: this.editableMesh?.boundsRadius ?? 0,
    });
  }

  private rebuildHoleLoopOverlays(): HoleLoopSummary {
    this.clearHoleLoopOverlays();

    if (!this.editableMesh || !this.mesh) {
      this.holeLoops = [];
      return {
        loopCount: 0,
        edgeCount: 0,
      };
    }

    const holeLoopSet = buildHoleLoopSet(this.editableMesh.indices);
    this.holeLoops = holeLoopSet.loops;
    this.hoveredHoleLoopIndex = -1;

    this.holeLoopOverlayGeometry = new LineSegmentsGeometry();
    this.holeLoopOverlayGeometry.setPositions(
      createLoopSegmentPositionArray(this.editableMesh.positions, this.holeLoops),
    );

    const holeLoopMaterial = new LineMaterial({
      color: '#29b8ff',
      linewidth: 4.5,
      transparent: true,
      opacity: 1,
      depthTest: false,
      depthWrite: false,
    });
    this.updateSingleLineMaterialResolution(holeLoopMaterial);
    this.holeLoopOverlay = new LineSegments2(this.holeLoopOverlayGeometry, holeLoopMaterial);
    this.holeLoopOverlay.frustumCulled = false;
    this.holeLoopOverlay.visible = this.holeFillMode && holeLoopSet.edgeCount > 0;
    this.holeLoopOverlay.renderOrder = 6;
    this.mesh.add(this.holeLoopOverlay);

    this.holeHoverOverlayGeometry = new LineSegmentsGeometry();
    this.holeHoverOverlayGeometry.setPositions(new Float32Array(0));

    const holeHoverMaterial = new LineMaterial({
      color: '#5b1fa5',
      linewidth: 6.5,
      transparent: true,
      opacity: 1,
      depthTest: false,
      depthWrite: false,
    });
    this.updateSingleLineMaterialResolution(holeHoverMaterial);
    this.holeHoverOverlay = new LineSegments2(this.holeHoverOverlayGeometry, holeHoverMaterial);
    this.holeHoverOverlay.frustumCulled = false;
    this.holeHoverOverlay.visible = false;
    this.holeHoverOverlay.renderOrder = 7;
    this.mesh.add(this.holeHoverOverlay);
    this.updateHoleLoopBaseOverlay();

    return {
      loopCount: holeLoopSet.loops.length,
      edgeCount: holeLoopSet.edgeCount,
    };
  }

  private updateHoleLoopOverlayVisibility(): void {
    const visible = this.holeFillMode && this.holeLoops.length > 0;
    if (this.holeLoopOverlay) {
      this.holeLoopOverlay.visible = visible;
    }

    if (!visible) {
      this.hoveredHoleLoopIndex = -1;
    }

    this.updateHoleLoopBaseOverlay();
    this.updateHoleHoverOverlay();
  }

  private updateHoleLoopHover(): void {
    if (
      !this.holeFillMode ||
      !this.editableMesh ||
      !this.mesh ||
      !this.pointerInside ||
      this.holeLoops.length === 0
    ) {
      if (this.hoveredHoleLoopIndex !== -1) {
        this.hoveredHoleLoopIndex = -1;
        this.updateHoleHoverOverlay();
      }

      return;
    }

    const width = this.overlayCanvas.clientWidth;
    const height = this.overlayCanvas.clientHeight;
    const thresholdSq = HOLE_LOOP_HOVER_DISTANCE_PX * HOLE_LOOP_HOVER_DISTANCE_PX;
    const positions = this.editableMesh.positions;

    let closestLoop = -1;
    let closestDistanceSq = thresholdSq;

    for (let loopIndex = 0; loopIndex < this.holeLoops.length; loopIndex += 1) {
      const segmentVertexPairs = this.holeLoops[loopIndex].segmentVertexPairs;

      for (let pairIndex = 0; pairIndex < segmentVertexPairs.length; pairIndex += 2) {
        const a = segmentVertexPairs[pairIndex] * 3;
        const b = segmentVertexPairs[pairIndex + 1] * 3;

        this.triangleWorldA.set(positions[a], positions[a + 1], positions[a + 2]);
        this.mesh.localToWorld(this.triangleWorldA);
        this.projectedPointA.copy(this.triangleWorldA).project(this.camera);

        this.triangleWorldB.set(positions[b], positions[b + 1], positions[b + 2]);
        this.mesh.localToWorld(this.triangleWorldB);
        this.projectedPointB.copy(this.triangleWorldB).project(this.camera);

        if (
          this.projectedPointA.z < -1 ||
          this.projectedPointA.z > 1 ||
          this.projectedPointB.z < -1 ||
          this.projectedPointB.z > 1
        ) {
          continue;
        }

        const ax = (this.projectedPointA.x * 0.5 + 0.5) * width;
        const ay = (-this.projectedPointA.y * 0.5 + 0.5) * height;
        const bx = (this.projectedPointB.x * 0.5 + 0.5) * width;
        const by = (-this.projectedPointB.y * 0.5 + 0.5) * height;
        const distanceSq = distanceToSegmentSquared(
          this.pointerClient.x,
          this.pointerClient.y,
          ax,
          ay,
          bx,
          by,
        );

        if (distanceSq >= closestDistanceSq) {
          continue;
        }

        closestDistanceSq = distanceSq;
        closestLoop = loopIndex;
      }
    }

    if (closestLoop !== this.hoveredHoleLoopIndex) {
      this.debugHoleFill('hover-change', {
        previousLoopIndex: this.hoveredHoleLoopIndex,
        nextLoopIndex: closestLoop,
      });
      this.hoveredHoleLoopIndex = closestLoop;
      this.updateHoleLoopBaseOverlay();
      this.updateHoleHoverOverlay();
    }
  }

  private updateHoleLoopBaseOverlay(): void {
    if (!this.holeLoopOverlay || !this.holeLoopOverlayGeometry || !this.editableMesh) {
      return;
    }

    if (!this.holeFillMode || this.holeLoops.length === 0) {
      this.holeLoopOverlayGeometry.setPositions(new Float32Array(0));
      this.holeLoopOverlay.visible = false;
      return;
    }

    this.holeLoopOverlayGeometry.setPositions(
      createLoopSegmentPositionArray(this.editableMesh.positions, this.holeLoops),
    );
    this.holeLoopOverlay.visible = this.holeLoops.length > 0;
  }

  private updateHoleHoverOverlay(): void {
    if (!this.holeHoverOverlay || !this.holeHoverOverlayGeometry || !this.editableMesh) {
      return;
    }

    const highlightIndex =
      this.hoveredHoleLoopIndex >= 0
        ? this.hoveredHoleLoopIndex
        : this.activeBoundaryLoopIndex >= 0
          ? this.activeBoundaryLoopIndex
          : -1;
    if (!this.holeFillMode || highlightIndex < 0 || !this.holeLoops[highlightIndex]) {
      this.holeHoverOverlay.visible = false;
      this.holeHoverOverlayGeometry.setPositions(new Float32Array(0));
      return;
    }

    const loop = this.holeLoops[highlightIndex];
    const nextGeometry = new LineSegmentsGeometry();
    nextGeometry.setPositions(createLoopHighlightPositionArray(this.editableMesh.positions, loop));
    const previousGeometry = this.holeHoverOverlay.geometry as LineSegmentsGeometry;
    this.holeHoverOverlay.geometry = nextGeometry;
    this.holeHoverOverlayGeometry = nextGeometry;
    previousGeometry.dispose();
    this.holeHoverOverlay.visible = true;
  }

  private fillHoveredHoleLoop(): boolean {
    this.debugHoleFill('fill-click', {
      hoveredHoleLoopIndex: this.hoveredHoleLoopIndex,
      loopCount: this.holeLoops.length,
      hasEditableMesh: Boolean(this.editableMesh),
    });
    if (!this.editableMesh || this.hoveredHoleLoopIndex < 0) {
      this.debugHoleFill('fill-click-no-hovered-loop', {
        hoveredHoleLoopIndex: this.hoveredHoleLoopIndex,
        hasEditableMesh: Boolean(this.editableMesh),
      });
      this.callbacks.onHoleFill?.({
        success: false,
        message: 'Hover a clean boundary loop before using Fill Hole.',
      });
      return false;
    }

    const loop = this.holeLoops[this.hoveredHoleLoopIndex];
    this.debugHoleFill('fill-click-loop', {
      loopIndex: this.hoveredHoleLoopIndex,
      edgeCount: loop?.edgeCount,
      boundaryEdgeCount: loop?.boundaryEdgeCount,
      isBoundaryLoop: loop?.isBoundaryLoop,
      orderedVertexCount: loop?.orderedVertexIds?.length ?? 0,
    });
    if (!loop || !loop.isBoundaryLoop || !loop.orderedVertexIds) {
      this.callbacks.onHoleFill?.({
        success: false,
        message: 'That highlighted contour is not a simple open boundary loop, so it was not filled.',
      });
      return false;
    }

    const fillMesh = createHoleFillMesh(this.editableMesh.positions, this.editableMesh.indices);
    this.debugHoleFill('fill-kernel-start', {
      orderedVertexCount: loop.orderedVertexIds.length,
    });
    const result = executeHoleFill(fillMesh, Array.from(loop.orderedVertexIds), {
      ignoreSharpFeatureValidation: true,
      debugLog: (stage, details) => this.debugHoleFill(stage, details),
    });
    this.debugHoleFill('fill-kernel-result', {
      success: result.success,
      reason: result.reason,
      message: result.message,
      timings: result.timings,
    });
    if (!result.success) {
      this.callbacks.onHoleFill?.({
        success: false,
        message: result.message,
      });
      return false;
    }

    this.sculptEngine?.discardRedoHistory();
    const beforeSnapshot = this.captureSessionSnapshot();
    const viewState = this.captureViewState();
    const referencePositions = createHoleFillReferencePositions(
      this.editableMesh.referencePositions,
      fillMesh.positions,
      this.editableMesh.vertexCount,
    );
    const geometry = createGeometryFromMeshArrays(fillMesh.positions, fillMesh.indices);
    geometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });
    this.debugHoleFill('fill-geometry-built', {
      faceCount: fillMesh.indices.length / 3,
      vertexCount: fillMesh.positions.length / 3,
    });

    const editable = createEditableMeshData(geometry, { referencePositions });
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    const nextSessionId = this.allocateSessionId();
    const afterSnapshot = {
      sessionId: nextSessionId,
      positions: editable.positions.slice(),
      indices: editable.indices.slice(),
      referencePositions: editable.referencePositions.slice(),
      history: engine.exportHistorySnapshot(),
      selectedTriangleMask: new Uint8Array(editable.triangleCount),
      selectedTriangleCount: 0,
    } satisfies SessionSnapshot;
    this.pushHistoryAction({
      kind: 'session',
      before: beforeSnapshot,
      after: afterSnapshot,
    });
    this.installSession(editable, engine, {
      sessionId: nextSessionId,
      resetActionHistory: false,
      resetView: false,
    });
    this.restoreViewState(viewState);
    this.debugHoleFill('fill-session-updated', {
      restoredCamera: true,
    });

    this.callbacks.onHoleFill?.({
      success: true,
      message: result.message,
    });
    return true;
  }

  private debugHoleFill(stage: string, details?: unknown): void {
    if (!HOLE_FILL_DEBUG) {
      return;
    }

    if (details === undefined) {
      console.log(`[hole-fill] ${stage}`);
      return;
    }

    console.log(`[hole-fill] ${stage}`, details);
  }

  private updateCursorVisuals(): void {
    if (!this.cursor) {
      return;
    }

    if (this.holeFillMode) {
      this.cursor.visible = false;
      return;
    }

    const material = this.cursor.material as MeshBasicMaterial;
    if (this.interactionMode === 'select' && this.selectionTool === 'sphere') {
      material.color.set('#7e22ce');
      material.opacity = 0.18;
    } else {
      material.color.set('#8ed8ff');
      material.opacity = 0.24;
    }

    const shouldShow =
      this.hoverHit &&
      (this.interactionMode === 'sculpt' ||
        (this.interactionMode === 'select' && this.selectionTool === 'sphere'));
    this.cursor.visible = Boolean(shouldShow);
  }

  private rebuildSelectionOverlay(): void {
    if (
      !this.selectionOverlay ||
      !this.selectionOverlayGeometry ||
      !this.editableMesh ||
      !this.selectedTriangleMask
    ) {
      this.selectionDirty = false;
      return;
    }

    const selectedTriangleCount = countSelectedTriangles(this.selectedTriangleMask);
    if (selectedTriangleCount !== this.selectedTriangleCount) {
      this.selectedTriangleCount = selectedTriangleCount;
      this.emitSelection();
    }
    const indexArray = new Uint32Array(selectedTriangleCount * 3);
    let cursor = 0;
    for (let triangle = 0; triangle < this.editableMesh.triangleCount; triangle += 1) {
      if (this.selectedTriangleMask[triangle] === 0) {
        continue;
      }

      const src = triangle * 3;
      indexArray[cursor] = this.editableMesh.indices[src];
      indexArray[cursor + 1] = this.editableMesh.indices[src + 1];
      indexArray[cursor + 2] = this.editableMesh.indices[src + 2];
      cursor += 3;
    }

    this.selectionOverlayGeometry.setIndex(new BufferAttribute(indexArray, 1));
    this.selectionOverlay.visible = selectedTriangleCount > 0;
    this.selectionDirty = false;
  }

  private drawSelectionPreview(): void {
    if (
      !this.selectionGestureActive ||
      this.selectionTool === 'sphere' ||
      this.interactionMode !== 'select'
    ) {
      this.clearOverlayCanvas();
      return;
    }

    const ctx = this.overlayContext;
    this.clearOverlayCanvas();

    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 6]);
    ctx.strokeStyle = '#7e22ce';
    ctx.fillStyle = 'rgba(126, 34, 206, 0.14)';

    if (this.selectionTool === 'box') {
      const left = Math.min(this.selectionStart.x, this.selectionCurrent.x);
      const top = Math.min(this.selectionStart.y, this.selectionCurrent.y);
      const width = Math.abs(this.selectionCurrent.x - this.selectionStart.x);
      const height = Math.abs(this.selectionCurrent.y - this.selectionStart.y);
      ctx.fillRect(left, top, width, height);
      ctx.strokeRect(left, top, width, height);
    } else if (this.selectionPath.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(this.selectionPath[0].x, this.selectionPath[0].y);
      for (let i = 1; i < this.selectionPath.length; i += 1) {
        ctx.lineTo(this.selectionPath[i].x, this.selectionPath[i].y);
      }

      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    ctx.restore();
  }

  private clearOverlayCanvas(): void {
    this.overlayContext.clearRect(
      0,
      0,
      this.overlayCanvas.clientWidth,
      this.overlayCanvas.clientHeight,
    );
  }

  private clearHoleLoopOverlays(): void {
    if (this.holeLoopOverlay) {
      this.holeLoopOverlay.removeFromParent();
      this.holeLoopOverlayGeometry?.dispose();
      (this.holeLoopOverlay.material as LineMaterial).dispose();
      this.holeLoopOverlay = null;
      this.holeLoopOverlayGeometry = null;
    }

    if (this.holeHoverOverlay) {
      this.holeHoverOverlay.removeFromParent();
      this.holeHoverOverlayGeometry?.dispose();
      (this.holeHoverOverlay.material as LineMaterial).dispose();
      this.holeHoverOverlay = null;
      this.holeHoverOverlayGeometry = null;
    }

    this.holeLoops = [];
    this.hoveredHoleLoopIndex = -1;
    this.activeBoundaryLoopIndex = -1;
  }

  private clearSceneMesh(): void {
    if (this.cursor) {
      this.cursor.removeFromParent();
      this.cursor.geometry.dispose();
      (this.cursor.material as MeshBasicMaterial).dispose();
      this.cursor = null;
    }

    if (this.selectionOverlay) {
      this.selectionOverlay.removeFromParent();
      this.selectionOverlayGeometry?.dispose();
      (this.selectionOverlay.material as MeshBasicMaterial).dispose();
      this.selectionOverlay = null;
      this.selectionOverlayGeometry = null;
    }

    this.clearHoleLoopOverlays();

    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.mesh.removeFromParent();
      this.meshMaterial?.dispose();
      this.mesh = null;
      this.meshMaterial = null;
    }

    this.editableMesh = null;
    this.sculptEngine = null;
    this.hoverHit = null;
    this.selectedTriangleMask = null;
    this.selectedTriangleCount = 0;
    this.selectionDirty = false;
  }
}

function floatArraysEqual(a: Float32Array | null, b: Float32Array | null): boolean {
  if (!a || !b || a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }

  return true;
}

function resolveSelectionOperation(
  event: PointerEvent,
  tool: SelectionTool,
): SelectionOperation {
  if (event.ctrlKey) {
    return 'subtract';
  }

  if (event.shiftKey) {
    return 'add';
  }

  return tool === 'sphere' ? 'add' : 'replace';
}

function pointInRect(x: number, y: number, a: Vector2, b: Vector2): boolean {
  const minX = Math.min(a.x, b.x);
  const maxX = Math.max(a.x, b.x);
  const minY = Math.min(a.y, b.y);
  const maxY = Math.max(a.y, b.y);
  return x >= minX && x <= maxX && y >= minY && y <= maxY;
}

function pointInPolygon(x: number, y: number, polygon: Vector2[]): boolean {
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const crossesY = (yi > y) !== (yj > y);
    if (!crossesY) {
      continue;
    }

    const intersects = x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;

    if (intersects) {
      inside = !inside;
    }
  }

  return inside;
}

function containsProjectedSelectionSample(
  tool: SelectionTool,
  projectedPoint: Vector3,
  width: number,
  height: number,
  selectionStart: Vector2,
  selectionCurrent: Vector2,
  selectionPath: Vector2[],
): boolean {
  if (projectedPoint.z < -1 || projectedPoint.z > 1) {
    return false;
  }

  const screenX = (projectedPoint.x * 0.5 + 0.5) * width;
  const screenY = (-projectedPoint.y * 0.5 + 0.5) * height;
  return containsSelectionSample(
    tool,
    screenX,
    screenY,
    selectionStart,
    selectionCurrent,
    selectionPath,
  );
}

function containsSelectionSample(
  tool: SelectionTool,
  screenX: number,
  screenY: number,
  selectionStart: Vector2,
  selectionCurrent: Vector2,
  selectionPath: Vector2[],
): boolean {
  return tool === 'box'
    ? pointInRect(screenX, screenY, selectionStart, selectionCurrent)
    : pointInPolygon(screenX, screenY, selectionPath);
}

function createLoopSegmentPositionArray(
  positions: Float32Array,
  loops: HoleLoop[],
): Float32Array {
  let totalSegments = 0;
  for (let i = 0; i < loops.length; i += 1) {
    totalSegments += loops[i].edgeCount;
  }

  const segmentPositions = new Float32Array(totalSegments * 6);
  let cursor = 0;

  for (let loopIndex = 0; loopIndex < loops.length; loopIndex += 1) {
    const segmentVertexPairs = loops[loopIndex].segmentVertexPairs;
    for (let pairIndex = 0; pairIndex < segmentVertexPairs.length; pairIndex += 2) {
      const a = segmentVertexPairs[pairIndex] * 3;
      const b = segmentVertexPairs[pairIndex + 1] * 3;

      segmentPositions[cursor] = positions[a];
      segmentPositions[cursor + 1] = positions[a + 1];
      segmentPositions[cursor + 2] = positions[a + 2];
      segmentPositions[cursor + 3] = positions[b];
      segmentPositions[cursor + 4] = positions[b + 1];
      segmentPositions[cursor + 5] = positions[b + 2];
      cursor += 6;
    }
  }

  return segmentPositions;
}

function createLoopHighlightPositionArray(
  positions: Float32Array,
  loop: HoleLoop,
): Float32Array {
  const orderedVertexIds = loop.orderedVertexIds;
  if (!orderedVertexIds || orderedVertexIds.length < 2) {
    return createLoopSegmentPositionArray(positions, [loop]);
  }

  const segmentPositions = new Float32Array(orderedVertexIds.length * 6);
  let cursor = 0;

  for (let i = 0; i < orderedVertexIds.length; i += 1) {
    const a = orderedVertexIds[i] * 3;
    const b = orderedVertexIds[(i + 1) % orderedVertexIds.length] * 3;

    segmentPositions[cursor] = positions[a];
    segmentPositions[cursor + 1] = positions[a + 1];
    segmentPositions[cursor + 2] = positions[a + 2];
    segmentPositions[cursor + 3] = positions[b];
    segmentPositions[cursor + 4] = positions[b + 1];
    segmentPositions[cursor + 5] = positions[b + 2];
    cursor += 6;
  }

  return segmentPositions;
}

function distanceToSegmentSquared(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const lengthSq = abx * abx + aby * aby;
  if (lengthSq <= 1e-9) {
    const dx = px - ax;
    const dy = py - ay;
    return dx * dx + dy * dy;
  }

  const t = Math.min(Math.max(((px - ax) * abx + (py - ay) * aby) / lengthSq, 0), 1);
  const closestX = ax + abx * t;
  const closestY = ay + aby * t;
  const dx = px - closestX;
  const dy = py - closestY;
  return dx * dx + dy * dy;
}

function createGeometryWithoutSelectedTriangles(
  positions: Float32Array,
  indices: Uint32Array,
  referencePositions: Float32Array,
  selectedMask: Uint8Array,
): { geometry: BufferGeometry | null; referencePositions: Float32Array | null } {
  const vertexMap = new Int32Array(positions.length / 3);
  vertexMap.fill(-1);

  const nextPositions: number[] = [];
  const nextReferencePositions: number[] = [];
  const nextIndices: number[] = [];

  for (let triangle = 0; triangle < indices.length / 3; triangle += 1) {
    if (selectedMask[triangle] !== 0) {
      continue;
    }

    const triOffset = triangle * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const sourceVertex = indices[triOffset + corner];
      let targetVertex = vertexMap[sourceVertex];
      if (targetVertex === -1) {
        targetVertex = nextPositions.length / 3;
        vertexMap[sourceVertex] = targetVertex;
        const positionOffset = sourceVertex * 3;
        nextPositions.push(
          positions[positionOffset],
          positions[positionOffset + 1],
          positions[positionOffset + 2],
        );
        nextReferencePositions.push(
          referencePositions[positionOffset],
          referencePositions[positionOffset + 1],
          referencePositions[positionOffset + 2],
        );
      }

      nextIndices.push(targetVertex);
    }
  }

  if (nextIndices.length === 0) {
    return {
      geometry: null,
      referencePositions: null,
    };
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(nextPositions), 3));
  geometry.setIndex(new BufferAttribute(new Uint32Array(nextIndices), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return {
    geometry,
    referencePositions: new Float32Array(nextReferencePositions),
  };
}

function createGeometryFromMeshArrays(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  geometry.setIndex(new BufferAttribute(new Uint32Array(indices), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function orientGeometryOutward(geometry: BufferGeometry): void {
  const positionAttribute = geometry.getAttribute('position');
  const indexAttribute = geometry.getIndex();
  if (!positionAttribute || !indexAttribute) {
    return;
  }

  const positions = positionAttribute.array as ArrayLike<number>;
  const indexArray = indexAttribute.array as ArrayLike<number>;
  if (positionAttribute.count === 0 || indexArray.length < 3) {
    return;
  }

  let centroidX = 0;
  let centroidY = 0;
  let centroidZ = 0;
  for (let vertex = 0; vertex < positionAttribute.count; vertex += 1) {
    const offset = vertex * 3;
    centroidX += positions[offset];
    centroidY += positions[offset + 1];
    centroidZ += positions[offset + 2];
  }

  const invVertexCount = 1 / positionAttribute.count;
  centroidX *= invVertexCount;
  centroidY *= invVertexCount;
  centroidZ *= invVertexCount;

  let orientationScore = 0;
  for (let triangle = 0; triangle < indexArray.length; triangle += 3) {
    const aOffset = indexArray[triangle] * 3;
    const bOffset = indexArray[triangle + 1] * 3;
    const cOffset = indexArray[triangle + 2] * 3;

    const ax = positions[aOffset];
    const ay = positions[aOffset + 1];
    const az = positions[aOffset + 2];
    const bx = positions[bOffset];
    const by = positions[bOffset + 1];
    const bz = positions[bOffset + 2];
    const cx = positions[cOffset];
    const cy = positions[cOffset + 1];
    const cz = positions[cOffset + 2];

    const abx = bx - ax;
    const aby = by - ay;
    const abz = bz - az;
    const acx = cx - ax;
    const acy = cy - ay;
    const acz = cz - az;
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;

    const triCentroidX = (ax + bx + cx) / 3;
    const triCentroidY = (ay + by + cy) / 3;
    const triCentroidZ = (az + bz + cz) / 3;
    orientationScore +=
      nx * (triCentroidX - centroidX) +
      ny * (triCentroidY - centroidY) +
      nz * (triCentroidZ - centroidZ);
  }

  if (orientationScore >= 0) {
    return;
  }

  const mutableIndexArray = indexAttribute.array as Uint16Array | Uint32Array;
  for (let triangle = 0; triangle < mutableIndexArray.length; triangle += 3) {
    const swap = mutableIndexArray[triangle + 1];
    mutableIndexArray[triangle + 1] = mutableIndexArray[triangle + 2];
    mutableIndexArray[triangle + 2] = swap;
  }

  indexAttribute.needsUpdate = true;
}

function createHoleFillReferencePositions(
  previousReferencePositions: Float32Array,
  nextPositions: ArrayLike<number>,
  preservedVertexCount: number,
): Float32Array {
  const referencePositions = new Float32Array(nextPositions.length);
  const preservedLength = Math.min(previousReferencePositions.length, preservedVertexCount * 3);
  referencePositions.set(previousReferencePositions.subarray(0, preservedLength), 0);

  for (let i = preservedLength; i < nextPositions.length; i += 1) {
    referencePositions[i] = nextPositions[i];
  }

  return referencePositions;
}

function createEmptySessionSnapshot(sessionId: number): SessionSnapshot {
  return {
    sessionId,
    positions: null,
    indices: null,
    referencePositions: null,
    history: null,
    selectedTriangleMask: null,
    selectedTriangleCount: 0,
  };
}

function countSelectedTriangles(selectedTriangleMask: Uint8Array): number {
  let count = 0;
  for (let i = 0; i < selectedTriangleMask.length; i += 1) {
    count += selectedTriangleMask[i] !== 0 ? 1 : 0;
  }

  return count;
}

function createStudioClayMatcapTexture(): CanvasTexture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Failed to create the sculpt matcap texture.');
  }

  const image = context.createImageData(size, size);
  const pixels = image.data;

  for (let y = 0; y < size; y += 1) {
    const v = (y / (size - 1)) * 2 - 1;
    for (let x = 0; x < size; x += 1) {
      const u = (x / (size - 1)) * 2 - 1;
      const radial = u * u + v * v;
      const pixel = (y * size + x) * 4;

      if (radial > 1) {
        pixels[pixel] = 0;
        pixels[pixel + 1] = 0;
        pixels[pixel + 2] = 0;
        pixels[pixel + 3] = 0;
        continue;
      }

      const nx = u;
      const ny = -v;
      const nz = Math.sqrt(1 - radial);

      const diffuseA = Math.max(0, nx * -0.44 + ny * 0.48 + nz * 0.76);
      const diffuseB = Math.max(0, nx * 0.62 + ny * -0.22 + nz * 0.66);
      const shadow = Math.max(0, nx * 0.33 + ny * -0.36 + nz * 0.05);
      const edgeShadow = Math.pow(1 - Math.max(0, nz), 1.08);
      const rim = Math.pow(1 - Math.max(0, nz), 2.45);
      const specular = Math.pow(Math.max(0, nx * -0.25 + ny * 0.24 + nz * 0.945), 34);
      const broadHighlight = Math.pow(Math.max(0, nx * -0.09 + ny * 0.13 + nz * 0.988), 3.2);

      let intensity =
        0.2 +
        diffuseA * 0.45 +
        diffuseB * 0.07 +
        broadHighlight * 0.16 +
        specular * 0.14 -
        shadow * 0.36 -
        edgeShadow * 0.25 -
        rim * 0.11;
      intensity = Math.min(Math.max(intensity, 0), 1);
      intensity =
        intensity < 0.5
          ? 0.5 * Math.pow(intensity * 2, 1.7)
          : 1 - 0.5 * Math.pow((1 - intensity) * 2, 2.05);

      const coolShift = shadow * 0.19 + edgeShadow * 0.16 + rim * 0.06;
      const warmShift = broadHighlight * 0.05 + specular * 0.045;
      pixels[pixel] = Math.round(128 + intensity * 92 + warmShift * 10);
      pixels[pixel + 1] = Math.round(133 + intensity * 94 + warmShift * 8);
      pixels[pixel + 2] = Math.round(144 + intensity * 99 - coolShift * 22);
      pixels[pixel + 3] = 255;
    }
  }

  context.putImageData(image, 0, 0);

  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}
