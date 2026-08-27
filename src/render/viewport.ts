import {
  ArrowHelper,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  GridHelper,
  Group,
  LinearFilter,
  LinearMipmapLinearFilter,
  MOUSE,
  type Material,
  Mesh,
  MeshBasicMaterial,
  MeshMatcapMaterial,
  NoToneMapping,
  PerspectiveCamera,
  PlaneGeometry,
  Raycaster,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  TorusGeometry,
  Triangle,
  type Texture,
  Vector2,
  Vector3,
  WebGLRenderer,
} from 'three';
import type { HitPointInfo, MeshBVH } from 'three-mesh-bvh';
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
  extrudeBoundaryLoopToZPlane,
} from '../ops/boundary-extrude';
import { surfaceRemeshMesh, weldGeometryByDistance } from '../ops/surface-remesh';
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
import { buildHoleLoopSet, buildOpenBoundaryLoopCandidates } from '../sculpt/hole-loops';
import { SculptEngine } from '../sculpt/sculpt-engine';
import type {
  BrushType,
  BoundaryWorkflowState,
  HistoryState,
  HoleLoopSummary,
  InteractionMode,
  MeasurementState,
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
  onBoundaryAction?: (result: { success: boolean; message: string; complete?: boolean }) => void;
  onPositiveLimbProgress?: (state: { visible: boolean; message: string }) => void;
  onMeshStatsChange?: (stats: MeshStats) => void;
  onHoleFill?: (result: { success: boolean; message: string }) => void;
  onViewCubeTransform?: (transform: string) => void;
  onMeasurementChange?: (state: MeasurementState) => void;
  onMeasurementCaptured?: (heightMm: number) => void;
  onMeasurementPickStateChange?: (active: boolean) => void;
  onPointToPointMeasurementCaptured?: (distanceMm: number) => void;
  onPointToPointPickStateChange?: (active: boolean) => void;
  onMeasurementStartCaptured?: (heightMm: number) => void;
  onMeasurementStartPickStateChange?: (active: boolean) => void;
  onMeasurementHoverChange?: (rowIndex: number | null) => void;
  onMeasurementVisibilityChange?: (visible: boolean) => void;
  onRotationDraftChange?: (angles: Record<MeshRotationAxis, number>) => void;
  onMeshViewModeChange?: (mode: MeshViewMode) => void;
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
  uvs: Float32Array | null;
  colors?: Float32Array | null;
  bakedVertexColorsActive?: boolean;
  history: SculptHistorySnapshot | null;
  selectedTriangleMask: Uint8Array | null;
  selectedTriangleCount: number;
  faceMaterialIndices: Uint8Array | null;
  meshViewMode?: MeshViewMode;
  rotationSessionAngles?: Record<MeshRotationAxis, number>;
}

interface SessionInstallOptions {
  sessionId?: number;
  resetActionHistory?: boolean;
  resetView?: boolean;
  selectedTriangleMask?: Uint8Array | null;
  selectedTriangleCount?: number;
  faceMaterialIndices?: Uint8Array | null;
  texture?: Texture | null;
  bakedVertexColorsActive?: boolean;
}

interface BakedColorSource {
  geometry: BufferGeometry;
  boundsTree: MeshBVH;
  positions: Float32Array;
  indices: Uint32Array;
  uvs: Float32Array | null;
  colors: Float32Array | null;
  faceMaterialIndices: Uint8Array | null;
  textureSampler: TextureColorSampler | null;
}

interface TextureColorSampler {
  width: number;
  height: number;
  flipY: boolean;
  data: Uint8ClampedArray;
}

interface MeasurementSection {
  distanceFromDistalMm: number;
  circumferenceMm: number;
  zMm: number;
  positions: Float32Array;
}

interface ViewTransition {
  startTime: number;
  duration: number;
  fromPosition: Vector3;
  toPosition: Vector3;
  fromUp: Vector3;
  toUp: Vector3;
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
  complete?: boolean;
}

export interface ExportedMeshFile {
  filename: string;
  blob: Blob;
}

export interface ObjExportResult {
  files: ExportedMeshFile[];
}

export type MeshViewMode = 'colored' | 'shaded' | 'wireframe';
export type MeshExportUnit = 'mm' | 'cm' | 'm' | 'in';
export type MeshRotationAxis = 'x' | 'y' | 'z';
export type ViewportUiTheme = 'light' | 'dark';

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
type OrientationView = 'front' | 'left' | 'right' | 'back' | 'proximal' | 'distal';

const DISABLED_MOUSE_ACTION = -1 as OrbitMouseAction;
const HOLE_LOOP_HOVER_DISTANCE_PX = 16;
const HOLE_FILL_DEBUG = true;
const HOLE_DIAGNOSTIC_NEAR_WELD_MM = 0.01;
const ACTION_HISTORY_LIMIT = 12;
const SCULPT_HISTORY_LIMIT = 12;
const MEASUREMENT_SPACING_MM = 25;
const POSITIVE_AUTO_FULL_REMESH_MM = 3;
const POSITIVE_AUTO_BOUNDARY_SMOOTH = 0.4;
const POSITIVE_AUTO_BOUNDARY_SMOOTH_ITERATIONS = 10;
const POSITIVE_AUTO_FIXED_REMESH_MM = 3.2;
const POSITIVE_AUTO_NORMAL_EXTRUDE_MM = 3;
const POSITIVE_AUTO_Z_PLANE_OFFSET_MM = 20;
const POSITIVE_FINAL_WELD_TOLERANCE_MM = 0.01;
const POSITIVE_FINAL_REPAIR_PASSES = 3;
const OBJ_TEXTURE_ATLAS_MAX_SIZE = 4096;
const OBJ_TEXTURE_ATLAS_PREFERRED_TILE_SIZE = 8;
const OBJ_TEXTURE_ATLAS_MIN_TILE_SIZE = 4;
const ORBIT_POLE_EPSILON = 0.0015;

function lerp(start: number, end: number, alpha: number): number {
  return start + (end - start) * alpha;
}

function createStableZUpOrbitOffset(direction: Vector3, distance: number): Vector3 {
  const normalized = direction.clone().normalize();
  if (Math.abs(normalized.z) > 1 - ORBIT_POLE_EPSILON) {
    const polar = normalized.z > 0 ? ORBIT_POLE_EPSILON : Math.PI - ORBIT_POLE_EPSILON;
    const sinPolar = Math.sin(polar);
    return new Vector3(0, -distance * sinPolar, distance * Math.cos(polar));
  }

  return normalized.multiplyScalar(distance);
}

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
  private readonly projectedPoint = new Vector3();
  private readonly projectedPointA = new Vector3();
  private readonly projectedPointB = new Vector3();
  private readonly projectedPointC = new Vector3();
  private readonly resizeObserver: ResizeObserver;

  private editableMesh: EditableMeshData | null = null;
  private sculptEngine: SculptEngine | null = null;
  private mesh: Mesh | null = null;
  private meshMaterial: Material | Material[] | null = null;
  private meshTexture: Texture | null = null;
  private meshViewMode: MeshViewMode = 'colored';
  private bakedVertexColorsActive = false;
  private faceMaterialIndices: Uint8Array | null = null;
  private cursor: Mesh | null = null;
  private selectionOverlay: Mesh | null = null;
  private selectionOverlayGeometry: BufferGeometry | null = null;
  private holeLoopOverlay: LineSegments2 | null = null;
  private holeLoopOverlayGeometry: LineSegmentsGeometry | null = null;
  private holeHoverOverlay: LineSegments2 | null = null;
  private holeHoverOverlayGeometry: LineSegmentsGeometry | null = null;
  private measurementOverlay: LineSegments2 | null = null;
  private measurementOverlayGeometry: LineSegmentsGeometry | null = null;
  private measurementHoverOverlay: LineSegments2 | null = null;
  private measurementHoverOverlayGeometry: LineSegmentsGeometry | null = null;
  private measurementHeightOverlay: LineSegments2 | null = null;
  private measurementHeightOverlayGeometry: LineSegmentsGeometry | null = null;
  private measurementPointOverlay: LineSegments2 | null = null;
  private measurementPointOverlayGeometry: LineSegmentsGeometry | null = null;
  private measurementGridOverlay: LineSegments2 | null = null;
  private measurementGridOverlayGeometry: LineSegmentsGeometry | null = null;
  private measurementAxisOverlay: LineSegments2 | null = null;
  private measurementAxisOverlayGeometry: LineSegmentsGeometry | null = null;
  private measurementHeightPointMarker: Mesh | null = null;
  private measurementPointStartMarker: Mesh | null = null;
  private measurementPointEndMarker: Mesh | null = null;
  private measurementHoverLabel: HTMLDivElement;
  private measurementHeightLabel: HTMLDivElement;
  private measurementPointLabel: HTMLDivElement;
  private rotationOverlay: Group | null = null;
  private rotationRings: Mesh[] = [];
  private rotationPickRings: Mesh[] = [];
  private rotationHoveredRing: Mesh | null = null;
  private positiveDirectionGuide: Group | null = null;
  private positiveLimbAutomationActive = false;
  private rotationOverlayVisible = false;
  private rotationDragAxis: MeshRotationAxis | null = null;
  private rotationDragPointerId = -1;
  private rotationDragStartVector: Vector3 | null = null;
  private rotationDragStartAngles: Record<MeshRotationAxis, number> = { x: 0, y: 0, z: 0 };
  private rotationDraftAngles: Record<MeshRotationAxis, number> = { x: 0, y: 0, z: 0 };
  private rotationDraftBeforeSnapshot: SessionSnapshot | null = null;
  private rotationDraftBasePositions: Float32Array | null = null;
  private rotationDraftBaseReferencePositions: Float32Array | null = null;
  private rotationDraftCenter: Vector3 | null = null;
  private rotationDraftRadius = 1;
  private rotationSessionAngles: Record<MeshRotationAxis, number> = { x: 0, y: 0, z: 0 };
  private rotationSessionCenter: Vector3 | null = null;
  private rotationSessionRadius = 1;
  private holeLoops: HoleLoop[] = [];
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
  private selectOnlyVisible = true;
  private selectionOperation: SelectionOperation = 'replace';
  private selectionPath: Vector2[] = [];
  private selectedTriangleMask: Uint8Array | null = null;
  private selectedTriangleCount = 0;
  private selectionDirty = false;
  private measurementState: MeasurementState = createEmptyMeasurementState();
  private measurementSections: MeasurementSection[] = [];
  private hoveredMeasurementIndex: number | null = null;
  private measurementDistalZ = 0;
  private measurementStartZ: number | null = null;
  private measurementHeightPoint: Vector3 | null = null;
  private measurementPointPickActive = false;
  private measurementPointStart: Vector3 | null = null;
  private measurementPointEnd: Vector3 | null = null;
  private measurementPointPreview: Vector3 | null = null;
  private measurementCircumferenceVisible = false;
  private measurementPickActive = false;
  private measurementStartPickActive = false;
  private brushType: BrushType = 'smooth';
  private brushRadiusMm = 5;
  private brushStrength = 0.35;
  private smoothOnlyTrimline = false;
  private selectionRadiusMm = 6;
  private nextSessionId = 1;
  private currentSessionId = 0;
  private historyUndoStack: ViewportHistoryAction[] = [];
  private historyRedoStack: ViewportHistoryAction[] = [];
  private lastViewCubeTransform = '';
  private lastViewCubeAzimuthDeg: number | null = null;
  private viewTransition: ViewTransition | null = null;
  private uiTheme: ViewportUiTheme = 'light';

  constructor(container: HTMLElement, callbacks: ViewportCallbacks = {}) {
    this.container = container;
    this.callbacks = callbacks;

    this.scene = new Scene();
    this.scene.background = new Color('#e9eef2');

    this.camera = new PerspectiveCamera(50, 1, 0.01, 1000);
    this.camera.position.set(2.8, 1.8, 3.4);
    this.camera.up.set(0, 0, 1);
    this.scene.add(this.camera);

    this.renderer = new WebGLRenderer({ antialias: true, alpha: true });
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
    this.measurementHoverLabel = document.createElement('div');
    this.measurementHoverLabel.className = 'measurement-hover-label';
    this.measurementHoverLabel.hidden = true;
    this.container.append(this.measurementHoverLabel);
    this.measurementHeightLabel = document.createElement('div');
    this.measurementHeightLabel.className = 'measurement-height-label';
    this.measurementHeightLabel.hidden = true;
    this.container.append(this.measurementHeightLabel);
    this.measurementPointLabel = document.createElement('div');
    this.measurementPointLabel.className = 'measurement-height-label measurement-point-label';
    this.measurementPointLabel.hidden = true;
    this.container.append(this.measurementPointLabel);
    this.sculptMatcapTexture = createStudioClayMatcapTexture();

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);
    this.controls.mouseButtons.LEFT = DISABLED_MOUSE_ACTION;
    this.controls.mouseButtons.MIDDLE = MOUSE.PAN;
    this.controls.mouseButtons.RIGHT = MOUSE.ROTATE;

    this.raycaster.firstHitOnly = true;

    this.attachEvents();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.applyUiTheme();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  setUiTheme(theme: ViewportUiTheme): void {
    if (this.uiTheme === theme) {
      return;
    }

    this.uiTheme = theme;
    this.applyUiTheme();
    this.refreshMeshMaterial();
  }

  private applyUiTheme(): void {
    const dark = this.uiTheme === 'dark';
    if (dark) {
      this.scene.background = null;
      this.renderer.setClearColor(new Color('#000312'), 0);
    } else {
      const background = new Color('#e9eef2');
      this.scene.background = background;
      this.renderer.setClearColor(background, 1);
    }

    this.setMeshBasicColor(this.cursor, dark ? '#00d1c8' : '#8ed8ff');
    this.setMeshBasicColor(this.selectionOverlay, dark ? '#8b5cf6' : '#5b1fa5');
    this.setLineColor(this.measurementOverlay, dark ? '#00d1c8' : '#0694a2');
    this.setLineColor(this.measurementHoverOverlay, dark ? '#7dd3fc' : '#f59e0b');
    this.setLineColor(this.measurementHeightOverlay, dark ? '#c4b5fd' : '#111827');
    this.setLineColor(this.measurementPointOverlay, '#86efac');
    this.setLineColor(this.measurementGridOverlay, dark ? '#5b7c91' : '#6b7280');
    this.setLineColor(this.measurementAxisOverlay, dark ? '#a7f3f0' : '#111827');
    this.setMeshBasicColor(this.measurementHeightPointMarker, dark ? '#c4b5fd' : '#111827');
    this.setMeshBasicColor(this.measurementPointStartMarker, '#86efac');
    this.setMeshBasicColor(this.measurementPointEndMarker, '#86efac');
    this.setLineColor(this.holeLoopOverlay, dark ? '#00d1c8' : '#29b8ff');
    this.setLineColor(this.holeHoverOverlay, dark ? '#9b5cff' : '#5b1fa5');
    this.applyRotationOverlayTheme();
  }

  private setLineColor(line: LineSegments2 | null, color: string): void {
    if (!line || Array.isArray(line.material)) {
      return;
    }

    (line.material as LineMaterial).color.set(color);
  }

  private setMeshBasicColor(mesh: Mesh | null, color: string): void {
    if (!mesh || Array.isArray(mesh.material) || !(mesh.material instanceof MeshBasicMaterial)) {
      return;
    }

    mesh.material.color.set(color);
  }

  private applyRotationOverlayTheme(): void {
    if (!this.rotationOverlay) {
      return;
    }

    const dark = this.uiTheme === 'dark';
    const plane = this.rotationOverlay.children[0] as Mesh | undefined;
    if (plane && !Array.isArray(plane.material) && plane.material instanceof MeshBasicMaterial) {
      plane.material.color.set(dark ? '#00d1c8' : '#8aa0b5');
      plane.material.opacity = dark ? 0.09 : 0.12;
    }

    const ringColors: Record<MeshRotationAxis, string> = dark
      ? { x: '#ff5b6d', y: '#00d1c8', z: '#67a8ff' }
      : { x: '#d14646', y: '#2f9b62', z: '#2f6eea' };
    for (const ring of this.rotationRings) {
      const axis = ring.userData.axis as MeshRotationAxis | undefined;
      if (!axis) {
        continue;
      }

      ring.userData.baseColor = ringColors[axis];
      this.setRotationRingHoverState(ring, this.rotationHoveredRing === ring);
    }
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

  setSmoothOnlyTrimline(enabled: boolean): void {
    this.smoothOnlyTrimline = enabled;
  }

  setInteractionMode(mode: InteractionMode): void {
    this.interactionMode = mode;
    if (mode !== 'sculpt') {
      this.finishStroke();
    }

    if (mode !== 'select') {
      this.finishSelectionGesture();
    }

    this.updateMeasurementOverlayVisibility();
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

    if (mode === 'positive') {
      this.showPositiveDirectionGuide();
    } else {
      this.hidePositiveDirectionGuide();
    }

    this.updateCursorVisuals();
    this.emitBoundaryWorkflow();
  }

  getMeasurementState(): MeasurementState {
    return {
      rows: this.measurementState.rows.map((row) => ({ ...row })),
      totalHeightMm: this.measurementState.totalHeightMm,
      clickedHeightMm: this.measurementState.clickedHeightMm,
      pointToPointDistanceMm: this.measurementState.pointToPointDistanceMm,
    };
  }

  setHoveredMeasurementIndex(rowIndex: number | null): void {
    if (
      rowIndex !== null &&
      (rowIndex < 0 || rowIndex >= this.measurementSections.length || !this.measurementCircumferenceVisible)
    ) {
      rowIndex = null;
    }

    if (this.hoveredMeasurementIndex === rowIndex) {
      this.updateMeasurementHoverLabel();
      return;
    }

    this.hoveredMeasurementIndex = rowIndex;
    this.rebuildMeasurementHoverOverlay();
    this.updateMeasurementHoverLabel();
    this.callbacks.onMeasurementHoverChange?.(rowIndex);
  }

  setMeasurementCircumferenceVisible(visible: boolean): void {
    this.measurementCircumferenceVisible = visible;
    if (!visible) {
      this.clearMeasurementHeight();
      this.cancelMeasurementPick();
      this.cancelMeasurementStartPick();
      this.cancelPointToPointMeasurementPick();
      this.clearPointToPointMeasurement();
      this.setHoveredMeasurementIndex(null);
    }
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
    this.updatePositiveDirectionGuideForCurrentState();
    this.callbacks.onMeasurementVisibilityChange?.(visible);
  }

  clearMeasurementHeight(): void {
    if (this.measurementState.clickedHeightMm === null && !this.measurementHeightPoint) {
      return;
    }

    this.measurementState = {
      ...this.measurementState,
      clickedHeightMm: null,
    };
    this.measurementHeightPoint = null;
    this.rebuildMeasurementHeightOverlay();
    this.updateMeasurementHeightLabel();
    this.emitMeasurements();
  }

  clearPointToPointMeasurement(): void {
    if (
      this.measurementState.pointToPointDistanceMm === null &&
      !this.measurementPointStart &&
      !this.measurementPointEnd &&
      !this.measurementPointPreview
    ) {
      return;
    }

    this.measurementState = {
      ...this.measurementState,
      pointToPointDistanceMm: null,
    };
    this.measurementPointStart = null;
    this.measurementPointEnd = null;
    this.measurementPointPreview = null;
    this.rebuildMeasurementHeightOverlay();
    this.updateMeasurementPointMarkers();
    this.updateMeasurementPointLabel();
    this.emitMeasurements();
  }

  beginMeasurementPick(): boolean {
    if (!this.mesh || !this.editableMesh) {
      return false;
    }

    this.measurementPickActive = true;
    this.callbacks.onMeasurementPickStateChange?.(true);
    this.previewMeasurementHeightAtPointer();
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
    this.updatePositiveDirectionGuideForCurrentState();
    return true;
  }

  cancelMeasurementPick(): void {
    if (!this.measurementPickActive) {
      return;
    }

    this.measurementPickActive = false;
    this.callbacks.onMeasurementPickStateChange?.(false);
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
    this.updatePositiveDirectionGuideForCurrentState();
  }

  beginPointToPointMeasurementPick(): boolean {
    if (!this.mesh || !this.editableMesh) {
      return false;
    }

    this.measurementPointPickActive = true;
    this.measurementPointStart = null;
    this.measurementPointPreview = null;
    this.measurementPointEnd = null;
    this.measurementState = {
      ...this.measurementState,
      pointToPointDistanceMm: null,
    };
    this.callbacks.onPointToPointPickStateChange?.(true);
    this.previewPointToPointMeasurementAtPointer();
    this.rebuildMeasurementHeightOverlay();
    this.updateMeasurementPointMarkers();
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
    this.updatePositiveDirectionGuideForCurrentState();
    return true;
  }

  cancelPointToPointMeasurementPick(): void {
    if (!this.measurementPointPickActive) {
      return;
    }

    this.measurementPointPickActive = false;
    this.measurementPointStart = null;
    this.measurementPointPreview = null;
    this.callbacks.onPointToPointPickStateChange?.(false);
    this.rebuildMeasurementHeightOverlay();
    this.updateMeasurementPointMarkers();
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
    this.updatePositiveDirectionGuideForCurrentState();
  }

  beginMeasurementStartPick(): boolean {
    if (!this.mesh || !this.editableMesh) {
      return false;
    }

    this.measurementStartPickActive = true;
    this.measurementCircumferenceVisible = true;
    this.clearMeasurementHeight();
    this.callbacks.onMeasurementStartPickStateChange?.(true);
    this.callbacks.onMeasurementVisibilityChange?.(true);
    this.previewMeasurementStartAtPointer();
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
    this.updatePositiveDirectionGuideForCurrentState();
    return true;
  }

  cancelMeasurementStartPick(): void {
    if (!this.measurementStartPickActive) {
      return;
    }

    this.measurementStartPickActive = false;
    this.callbacks.onMeasurementStartPickStateChange?.(false);
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
    this.updatePositiveDirectionGuideForCurrentState();
  }

  setSelectionTool(tool: SelectionTool): void {
    this.selectionTool = tool;
    this.finishSelectionGesture();
    this.updateCursorVisuals();
  }

  setMeshViewMode(mode: MeshViewMode): void {
    if (this.meshViewMode === mode) {
      return;
    }

    this.meshViewMode = mode;
    if (!this.mesh) {
      this.callbacks.onMeshViewModeChange?.(mode);
      return;
    }

    const nextMaterial = createMeshMaterials(
      this.meshTexture,
      this.sculptMatcapTexture,
      this.faceMaterialIndices,
      this.meshViewMode,
      this.bakedVertexColorsActive,
      this.uiTheme,
    );
    disposeMaterial(this.meshMaterial);
    this.meshMaterial = nextMaterial;
    this.mesh.material = nextMaterial;
    this.callbacks.onMeshViewModeChange?.(mode);
  }

  setSelectOnlyVisible(enabled: boolean): void {
    this.selectOnlyVisible = enabled;
  }

  setSelectionRadiusMm(radiusMm: number): void {
    this.selectionRadiusMm = radiusMm;
  }

  setRotationOverlayVisible(visible: boolean): void {
    this.rotationOverlayVisible = visible;
    this.ensureRotationOverlay();
    if (visible) {
      this.hidePositiveDirectionGuide();
      this.beginRotationDraft();
    }
    if (this.rotationOverlay) {
      this.rotationOverlay.visible = visible && Boolean(this.editableMesh);
      this.updateRotationOverlayScale();
    }

    if (!visible) {
      this.finishRotationDraft(true);
      this.updatePositiveDirectionGuideForCurrentState();
    }
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
  }

  commitRotationDraft(): void {
    this.finishRotationDraft(true);
  }

  setMeshRotationDraft(
    angles: Partial<Record<MeshRotationAxis, number>>,
    emitChange = true,
  ): ViewportActionResult {
    if (!this.editableMesh || !this.sculptEngine) {
      return {
        success: false,
        message: 'Load a mesh before rotating.',
        stats: null,
      };
    }

    if (!this.rotationDraftBasePositions || !this.rotationDraftBaseReferencePositions) {
      this.beginRotationDraft();
    }

    const displayedAngles = this.getDisplayedRotationAngles();
    const nextDisplayedAngles = {
      x: angles.x ?? displayedAngles.x,
      y: angles.y ?? displayedAngles.y,
      z: angles.z ?? displayedAngles.z,
    };
    if (
      !Number.isFinite(nextDisplayedAngles.x) ||
      !Number.isFinite(nextDisplayedAngles.y) ||
      !Number.isFinite(nextDisplayedAngles.z)
    ) {
      return {
        success: false,
        message: 'Enter valid rotation angles.',
        stats: null,
      };
    }

    this.finishStroke();
    this.finishSelectionGesture();
    this.rotationDraftAngles = {
      x: nextDisplayedAngles.x - this.rotationSessionAngles.x,
      y: nextDisplayedAngles.y - this.rotationSessionAngles.y,
      z: nextDisplayedAngles.z - this.rotationSessionAngles.z,
    };
    this.applyRotationDraft();
    if (emitChange) {
      this.callbacks.onRotationDraftChange?.(this.getDisplayedRotationAngles());
    }

    return {
      success: true,
      message: `Rotation set to X ${nextDisplayedAngles.x.toFixed(3)}, Y ${nextDisplayedAngles.y.toFixed(3)}, Z ${nextDisplayedAngles.z.toFixed(3)} deg.`,
      stats: {
        vertexCount: this.editableMesh.vertexCount,
        triangleCount: this.editableMesh.triangleCount,
        boundsRadius: this.editableMesh.boundsRadius,
      },
    };
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

  setOrientationView(view: OrientationView): void {
    const direction = new Vector3();

    switch (view) {
      case 'left':
        direction.set(1, 0, 0);
        break;
      case 'right':
        direction.set(-1, 0, 0);
        break;
      case 'back':
        direction.set(0, 1, 0);
        break;
      case 'proximal':
        direction.set(0, 0, 1);
        break;
      case 'distal':
        direction.set(0, 0, -1);
        break;
      case 'front':
      default:
        direction.set(0, -1, 0);
        break;
    }

    this.setOrientationDirection(direction);
  }

  setOrientationDirection(direction: Vector3): void {
    if (direction.lengthSq() <= 1e-8) {
      return;
    }

    const currentOffset = this.camera.position.clone().sub(this.controls.target);
    const currentDistance = Math.max(currentOffset.length(), 0.5);
    const targetOffset = createStableZUpOrbitOffset(direction, currentDistance);

    this.viewTransition = {
      startTime: performance.now(),
      duration: 320,
      fromPosition: this.camera.position.clone(),
      toPosition: this.controls.target.clone().add(targetOffset),
      fromUp: this.camera.up.clone(),
      toUp: new Vector3(0, 0, 1),
    };
    const boundsRadius = Math.max(this.editableMesh?.boundsRadius ?? 1, 0.5);
    this.camera.near = Math.max(boundsRadius / 500, 0.001);
    this.camera.far = Math.max(boundsRadius * 25, 10);
    this.camera.updateProjectionMatrix();
  }

  setOrientationVector(x: number, y: number, z: number): void {
    this.setOrientationDirection(new Vector3(x, y, z));
  }

  orbitFromViewCube(deltaX: number, deltaY: number): void {
    this.viewTransition = null;
    const offset = this.camera.position.clone().sub(this.controls.target);
    const distance = offset.length();
    if (distance <= 1e-8) {
      return;
    }

    const theta = Math.atan2(offset.x, -offset.y) - deltaX * 0.012;
    const currentPolar = Math.acos(Math.min(Math.max(offset.z / distance, -1), 1));
    const polar = Math.min(
      Math.max(currentPolar - deltaY * 0.012, ORBIT_POLE_EPSILON),
      Math.PI - ORBIT_POLE_EPSILON,
    );
    const sinPolar = Math.sin(polar);
    offset.set(
      distance * sinPolar * Math.sin(theta),
      -distance * sinPolar * Math.cos(theta),
      distance * Math.cos(polar),
    );
    this.camera.position.copy(this.controls.target).add(offset);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(this.controls.target);
    this.controls.update();
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

  resetPositiveLimbPrompt(): HoleLoopSummary | null {
    if (this.interactionMode !== 'positive') {
      return null;
    }

    this.positiveLimbAutomationActive = false;
    this.hidePositiveDirectionGuide();
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
    this.holeFillMode = true;
    if (this.editableMesh && this.mesh) {
      this.rebuildHoleLoopOverlays();
    }
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.emitBoundaryWorkflow();
    return this.getHoleLoopSummary();
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
    const resolvedLoop = resolveUsableBoundaryLoop(loop, this.editableMesh.referencePositions);
    if (!resolvedLoop) {
      return {
        success: false,
        message: diagnoseUnfillableHoleLoop(loop, this.editableMesh.referencePositions),
        stats: null,
      };
    }

    this.activeBoundaryLoopIndex = this.hoveredHoleLoopIndex;
    this.activeBoundaryVertexIds = resolvedLoop.orderedVertexIds.slice();
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

    if (this.interactionMode === 'positive') {
      const result = this.startPositiveLimbAutomation();
      if (result.success && resolvedLoop.autoMessage) {
        return {
          ...result,
          message: `${resolvedLoop.autoMessage} ${result.message}`,
        };
      }
      return result;
    }

    return {
      success: true,
      message: resolvedLoop.autoMessage
        ? `${resolvedLoop.autoMessage} Targeted a boundary loop with ${resolvedLoop.orderedVertexIds.length} edges.`
        : `Targeted a boundary loop with ${resolvedLoop.orderedVertexIds.length} edges. Adjust the boundary smooth sliders, then press Done Boundary Smooth.`,
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
        message: 'Run the fixed-boundary remesh before previewing a positive limb extrusion.',
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
      message: `Previewing a positive limb extrusion at ${distance.toFixed(3)} mm.`,
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

  private startPositiveLimbAutomation(): ViewportActionResult {
    if (this.positiveLimbAutomationActive) {
      return {
        success: false,
        message: 'Positive Limb is already processing.',
        stats: null,
      };
    }

    this.positiveLimbAutomationActive = true;
    this.reportPositiveLimbProgress('Selected open edge. Preparing Positive Limb.');
    window.setTimeout(() => {
      void this.runPositiveLimbAutomation();
    }, 0);

    return {
      success: true,
      message: 'Positive Limb processing started.',
      stats: this.editableMesh
        ? {
            vertexCount: this.editableMesh.vertexCount,
            triangleCount: this.editableMesh.triangleCount,
            boundsRadius: this.editableMesh.boundsRadius,
          }
        : null,
    };
  }

  private async runPositiveLimbAutomation(): Promise<void> {
    if (!this.editableMesh || !this.sculptEngine || !this.boundaryGuide || !this.activeBoundaryVertexIds) {
      this.positiveLimbAutomationActive = false;
      this.reportPositiveLimbProgress('', false);
      this.callbacks.onBoundaryAction?.({
        success: false,
        message: 'Target a boundary loop before running Positive Limb.',
        complete: true,
      });
      return;
    }

    this.sculptEngine.discardRedoHistory();
    const beforeSnapshot = this.boundaryPreviewBaseSnapshot ?? this.captureSessionSnapshot();
    const viewState = this.captureViewState();
    const initialGuide = this.boundaryGuide.slice();
    const initialBoundary = this.activeBoundaryVertexIds.slice();
    const expectsColorTransfer = Boolean(this.meshTexture || this.bakedVertexColorsActive);
    const colorSource = this.capturePositiveLimbColorSource();
    let colorSourceDisposed = false;
    const meshProximalBounds = computeAxisBoundsZ(this.editableMesh.positions);
    const capPlaneZ =
      Math.max(computeHighestZ(this.editableMesh.positions, initialBoundary), meshProximalBounds.maxZ) +
      POSITIVE_AUTO_Z_PLANE_OFFSET_MM;
    this.showPositiveDirectionGuide(capPlaneZ);

    try {
      if (expectsColorTransfer && !colorSource) {
        throw new Error('The loaded scan color could not be sampled for Positive Limb.');
      }
      await this.reportPositiveLimbProgress('Full mesh remesh at 3.0 mm.');
      let editable = this.installAutomatedGeometry(
        surfaceRemeshMesh(this.editableMesh, POSITIVE_AUTO_FULL_REMESH_MM, { boundaryMode: 'refined' }).geometry,
        viewState,
        { weld: false },
      );
      this.showPositiveDirectionGuide(capPlaneZ);
      this.boundaryGuide = initialGuide;
      this.activeBoundaryVertexIds = this.resolveBoundaryLoopVertexIds(editable, initialGuide);
      if (!this.activeBoundaryVertexIds) {
        throw new Error('The selected boundary could not be found after the first 3.000 mm remesh.');
      }

      await this.reportPositiveLimbProgress('Smoothing selected open edge: 5 free passes, then 5 tangent passes.');
      const freeSmoothIterations = Math.min(5, POSITIVE_AUTO_BOUNDARY_SMOOTH_ITERATIONS);
      const tangentSmoothIterations = Math.max(0, POSITIVE_AUTO_BOUNDARY_SMOOTH_ITERATIONS - freeSmoothIterations);
      let smoothedPositions = smoothBoundaryLoopVertices(
        editable.positions,
        editable.normals,
        this.activeBoundaryVertexIds,
        POSITIVE_AUTO_BOUNDARY_SMOOTH,
        freeSmoothIterations,
        { constrainToTangent: false },
      );
      if (!smoothedPositions) {
        throw new Error('The selected boundary could not be smoothed.');
      }
      if (tangentSmoothIterations > 0) {
        const smoothNormals = editable.normals.slice();
        const smoothFaceNormals = editable.faceNormals.slice();
        recomputeAllNormals(
          smoothedPositions,
          editable.indices,
          smoothFaceNormals,
          smoothNormals,
          editable.vertexFaceOffsets,
          editable.vertexFaces,
        );
        const tangentSmoothedPositions = smoothBoundaryLoopVertices(
          smoothedPositions,
          smoothNormals,
          this.activeBoundaryVertexIds,
          POSITIVE_AUTO_BOUNDARY_SMOOTH,
          tangentSmoothIterations,
          { constrainToTangent: true },
        );
        if (!tangentSmoothedPositions) {
          throw new Error('The selected boundary could not complete its tangent smooth pass.');
        }
        smoothedPositions = tangentSmoothedPositions;
      }
      this.applyPositionsInPlace(smoothedPositions);
      if (!this.editableMesh) {
        throw new Error('Positive Limb lost the active mesh after boundary smoothing.');
      }
      editable = this.editableMesh;
      const smoothedGuide = captureBoundaryGuide(editable.positions, this.activeBoundaryVertexIds);
      if (!smoothedGuide) {
        throw new Error('The selected boundary guide could not be rebuilt after smoothing.');
      }
      this.boundaryGuide = smoothedGuide.slice();

      await this.reportPositiveLimbProgress('Voxel remesh at 3.2 mm.');
      editable = this.installAutomatedGeometry(
        surfaceRemeshMesh(editable, POSITIVE_AUTO_FIXED_REMESH_MM, { boundaryMode: 'fixed' }).geometry,
        viewState,
        { weld: false },
      );
      this.showPositiveDirectionGuide(capPlaneZ);
      this.boundaryGuide = smoothedGuide;
      this.activeBoundaryVertexIds = this.resolveBoundaryLoopVertexIds(editable, smoothedGuide);
      if (!this.activeBoundaryVertexIds) {
        throw new Error('The selected boundary could not be found after the 3.200 mm fixed remesh.');
      }

      await this.reportPositiveLimbProgress('Extruding outward along surface normals.');
      const scanTriangleCount = editable.triangleCount;
      const normalExtrude = extrudeBoundaryLoop(
        editable,
        this.activeBoundaryVertexIds,
        POSITIVE_AUTO_NORMAL_EXTRUDE_MM,
      );
      const normalExtrudePositions = normalExtrude.geometry.getAttribute('position')?.array as
        | ArrayLike<number>
        | undefined;
      const normalExtrudeGuide = normalExtrudePositions
        ? captureBoundaryGuide(normalExtrudePositions, normalExtrude.outerVertexIds)
        : null;
      if (!normalExtrudeGuide) {
        throw new Error('The normal extrusion boundary could not be tracked for the proximal extension.');
      }
      editable = this.installAutomatedGeometry(normalExtrude.geometry, viewState);
      this.showPositiveDirectionGuide(capPlaneZ);
      this.boundaryGuide = normalExtrudeGuide;
      this.activeBoundaryVertexIds = this.resolveBoundaryLoopVertexIds(editable, normalExtrudeGuide);
      if (!this.activeBoundaryVertexIds) {
        throw new Error('The normal extrusion boundary could not be found after weld before the proximal extension.');
      }
      this.boundaryGuide = captureBoundaryGuide(editable.positions, this.activeBoundaryVertexIds);
      if (!this.boundaryGuide) {
        throw new Error('The welded normal extrusion boundary could not be rebuilt before the proximal extension.');
      }

      await this.reportPositiveLimbProgress('Extending toward proximal and capping the top.');
      const zExtrude = extrudeBoundaryLoopToZPlane(editable, this.activeBoundaryVertexIds, capPlaneZ);
      const sourceTriangleCount = Math.floor((zExtrude.geometry.getIndex()?.count ?? 0) / 3);
      const sourceFaceMaterialIndices = new Uint8Array(sourceTriangleCount);
      sourceFaceMaterialIndices.fill(1, Math.min(scanTriangleCount, sourceTriangleCount));
      await this.reportPositiveLimbProgress('Repairing seams, manifold topology, holes, and winding.');
      const finalized = finalizePositiveLimbGeometry(zExtrude.geometry, sourceFaceMaterialIndices);
      editable = this.installAutomatedGeometry(finalized.geometry, viewState, {
        weld: false,
        bakedVertexColorsActive: false,
        faceMaterialIndices: finalized.faceMaterialIndices,
      });
      if (colorSource) {
        if (!this.bakePositiveLimbVertexColors(editable, colorSource)) {
          throw new Error('Positive Limb finished its geometry, but the scan color could not be transferred.');
        }
        this.bakedVertexColorsActive = true;
        if (this.sculptEngine) {
          this.sculptEngine.preserveVertexColors = true;
        }
        this.refreshMeshMaterial();
      }
      if (colorSource) {
        colorSource.geometry.dispose();
        colorSourceDisposed = true;
      }

      this.activeBoundaryLoopIndex = -1;
      this.activeBoundaryVertexIds = null;
      this.boundaryGuide = null;
      this.boundaryPreviewBaseSnapshot = null;
      this.boundaryThickenPreviewBaseSnapshot = null;
      this.boundaryExtrudePreviewBaseSnapshot = null;
      this.boundaryFinalSmoothPreviewBaseSnapshot = null;
      this.boundaryDirectionalExtrudePreviewBaseSnapshot = null;
      this.boundarySmoothCommitted = true;
      this.boundaryRemeshApplied = true;
      this.boundaryThickenApplied = false;
      this.boundaryExtrudeApplied = true;
      this.boundaryOffsetApplied = true;
      this.holeFillMode = false;
      this.hoveredHoleLoopIndex = -1;
      this.clearHoleLoopOverlays();
      this.emitBoundaryWorkflow();

      const afterSnapshot = this.captureSessionSnapshot();
      if (beforeSnapshot.positions && !floatArraysEqual(beforeSnapshot.positions, afterSnapshot.positions)) {
        this.pushHistoryAction({
          kind: 'session',
          before: beforeSnapshot,
          after: afterSnapshot,
        });
        this.emitHistory();
      }

      this.positiveLimbAutomationActive = false;
      this.hidePositiveDirectionGuide();
      this.reportPositiveLimbProgress('', false);
      const topologyMessage =
        finalized.topology.boundaryEdges === 0 &&
        finalized.topology.nonManifoldEdges === 0 &&
        finalized.topology.inconsistentWindingEdges === 0
          ? ' Closed manifold topology and outward normals validated.'
          :
            ` Final cleanup kept the completed mesh with ${finalized.topology.boundaryEdges} boundary edge${
              finalized.topology.boundaryEdges === 1 ? '' : 's'
            }, ${finalized.topology.nonManifoldEdges} non-manifold edge${
              finalized.topology.nonManifoldEdges === 1 ? '' : 's'
            }, and ${finalized.topology.inconsistentWindingEdges} winding conflict${
              finalized.topology.inconsistentWindingEdges === 1 ? '' : 's'
            } instead of rejecting it.`;
      this.callbacks.onBoundaryAction?.({
        success: true,
        message:
          `Positive Limb complete: 3.000 mm remesh, 0.200 boundary smooth, 3.200 mm fixed remesh, 3.000 mm normal extrusion, and capped proximal +Z plane at ${capPlaneZ.toFixed(1)} mm.` +
          topologyMessage +
          (this.bakedVertexColorsActive ? ' Scan color transferred to the repaired mesh.' : ''),
        complete: true,
      });
      this.emitMeshStats();
    } catch (error) {
      if (colorSource && !colorSourceDisposed) {
        colorSource.geometry.dispose();
      }
      console.error(error);
      if (beforeSnapshot.positions) {
        this.applySessionSnapshot(beforeSnapshot, viewState);
      }
      this.positiveLimbAutomationActive = false;
      this.hidePositiveDirectionGuide();
      this.reportPositiveLimbProgress('', false);
      this.callbacks.onBoundaryAction?.({
        success: false,
        message: error instanceof Error ? error.message : 'Positive Limb automation failed.',
        complete: true,
      });
    }
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
        message: 'Finish the positive limb smoothing stages before previewing the final wall extrusion.',
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
      this.camera.up.set(0, 0, 1);
      this.camera.position.set(2.8, -3.4, 1.8);
      this.camera.near = 0.01;
      this.camera.far = 1000;
      this.camera.updateProjectionMatrix();
      this.controls.update();
      return;
    }

    const radius = Math.max(this.editableMesh.boundsRadius, 0.5);
    const distance = radius / Math.tan((this.camera.fov * Math.PI) / 360) * 1.25;
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(radius * 1.35, -distance, radius * 0.9);
    this.camera.near = Math.max(radius / 500, 0.001);
    this.camera.far = Math.max(radius * 25, 10);
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setSession(meshData: EditableMeshData, sculptEngine: SculptEngine, texture: Texture | null = null): void {
    this.installSession(meshData, sculptEngine, {
      sessionId: this.allocateSessionId(),
      resetActionHistory: true,
      resetView: true,
      texture,
      bakedVertexColorsActive: false,
    });
  }

  exportStl(baseName: string, unit: MeshExportUnit = 'mm'): ExportedMeshFile | null {
    if (!this.editableMesh || this.editableMesh.triangleCount === 0) {
      return null;
    }

    const stl = serializeAsciiStl(
      sanitizeExportName(baseName),
      this.editableMesh.positions,
      this.editableMesh.indices,
      getMillimeterExportScale(unit),
    );
    return {
      filename: `${sanitizeExportName(baseName)}.stl`,
      blob: new Blob([stl], { type: 'model/stl' }),
    };
  }

  exportObj(baseName: string, unit: MeshExportUnit = 'mm'): ObjExportResult | null {
    if (!this.editableMesh || this.editableMesh.triangleCount === 0) {
      return null;
    }

    const exportName = sanitizeExportName(baseName);
    const sourceUvs = copyGeometryUvs(this.editableMesh.geometry);
    const bakedColors =
      this.bakedVertexColorsActive && this.editableMesh.colors.length >= this.editableMesh.vertexCount * 3
        ? this.editableMesh.colors
        : null;
    const textureExport =
      !bakedColors && sourceUvs && this.meshTexture
        ? createObjSourceTextureExport(this.meshTexture, sourceUvs)
        : createObjTextureAtlas({
            vertexCount: this.editableMesh.vertexCount,
            indices: this.editableMesh.indices,
            sourceUvs,
            sourceColors: bakedColors,
            textureSampler: sourceUvs && this.meshTexture ? createTextureColorSampler(this.meshTexture) : null,
            faceMaterialIndices: this.faceMaterialIndices,
          });
    const textureFilename = `${exportName}_texture.png`;
    const obj = serializeObj({
      objectName: exportName,
      materialFilename: `${exportName}.mtl`,
      unit,
      coordinateScale: getMillimeterExportScale(unit),
      positions: this.editableMesh.positions,
      normals: this.editableMesh.normals,
      indices: this.editableMesh.indices,
      uvs: textureExport.uvs,
      triangleUvIndices: textureExport.triangleUvIndices,
      faceMaterialIndices: this.faceMaterialIndices,
      scanMaterialName: 'scan_texture',
      fillMaterialName: 'fill_light_gray',
    });
    const mtl = serializeMtl({
      scanMaterialName: 'scan_texture',
      fillMaterialName: 'fill_light_gray',
      textureFilename,
    });

    return {
      files: [
        { filename: `${exportName}.obj`, blob: new Blob([obj], { type: 'model/obj' }) },
        { filename: `${exportName}.mtl`, blob: new Blob([mtl], { type: 'text/plain' }) },
        { filename: textureFilename, blob: textureExport.blob },
      ],
    };
  }

  private installSession(
    meshData: EditableMeshData,
    sculptEngine: SculptEngine,
    options: SessionInstallOptions = {},
  ): void {
    this.finishStroke();
    this.finishSelectionGesture();
    const nextTexture = options.texture === undefined ? this.meshTexture : options.texture;
    const nextBakedVertexColorsActive = options.bakedVertexColorsActive ?? this.bakedVertexColorsActive;
    this.clearSceneMesh(nextTexture !== this.meshTexture);
    if (nextTexture) {
      this.configureDisplayTexture(nextTexture);
    }

    if (options.resetActionHistory) {
      this.historyUndoStack = [];
      this.historyRedoStack = [];
      this.resetRotationSession();
    }

    this.editableMesh = meshData;
    this.sculptEngine = sculptEngine;
    this.currentSessionId = options.sessionId ?? this.allocateSessionId();
    this.bakedVertexColorsActive = nextBakedVertexColorsActive;
    this.sculptEngine.preserveVertexColors = this.bakedVertexColorsActive;
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
    this.faceMaterialIndices =
      options.faceMaterialIndices && options.faceMaterialIndices.length === meshData.triangleCount
        ? options.faceMaterialIndices.slice()
        : null;
    this.selectionDirty = true;
    if (this.bakedVertexColorsActive) {
      meshData.geometry.clearGroups();
    } else {
      applyFaceMaterialGroups(meshData.geometry, this.faceMaterialIndices);
    }

    this.meshMaterial = createMeshMaterials(
      nextTexture,
      this.sculptMatcapTexture,
      this.faceMaterialIndices,
      this.meshViewMode,
      this.bakedVertexColorsActive,
      this.uiTheme,
    );
    this.meshTexture = nextTexture ?? null;

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

    this.measurementOverlayGeometry = new LineSegmentsGeometry();
    const measurementMaterial = new LineMaterial({
      color: '#0694a2',
      linewidth: 2.4,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
      depthWrite: false,
    });
    this.updateSingleLineMaterialResolution(measurementMaterial);
    this.measurementOverlay = new LineSegments2(this.measurementOverlayGeometry, measurementMaterial);
    this.measurementOverlay.frustumCulled = false;
    this.measurementOverlay.visible = false;
    this.measurementOverlay.renderOrder = 8;
    this.mesh.add(this.measurementOverlay);

    this.measurementHoverOverlayGeometry = new LineSegmentsGeometry();
    const measurementHoverMaterial = new LineMaterial({
      color: '#f59e0b',
      linewidth: 5,
      transparent: true,
      opacity: 1,
      depthTest: false,
      depthWrite: false,
    });
    this.updateSingleLineMaterialResolution(measurementHoverMaterial);
    this.measurementHoverOverlay = new LineSegments2(this.measurementHoverOverlayGeometry, measurementHoverMaterial);
    this.measurementHoverOverlay.frustumCulled = false;
    this.measurementHoverOverlay.visible = false;
    this.measurementHoverOverlay.renderOrder = 10;
    this.mesh.add(this.measurementHoverOverlay);

    this.measurementHeightOverlayGeometry = new LineSegmentsGeometry();
    const heightMaterial = new LineMaterial({
      color: '#111827',
      linewidth: 3,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
      depthWrite: false,
    });
    this.updateSingleLineMaterialResolution(heightMaterial);
    this.measurementHeightOverlay = new LineSegments2(this.measurementHeightOverlayGeometry, heightMaterial);
    this.measurementHeightOverlay.frustumCulled = false;
    this.measurementHeightOverlay.visible = false;
    this.measurementHeightOverlay.renderOrder = 9;
    this.mesh.add(this.measurementHeightOverlay);

    this.measurementPointOverlayGeometry = new LineSegmentsGeometry();
    const pointMeasureMaterial = new LineMaterial({
      color: '#86efac',
      linewidth: 3,
      transparent: true,
      opacity: 0.96,
      depthTest: false,
      depthWrite: false,
    });
    this.updateSingleLineMaterialResolution(pointMeasureMaterial);
    this.measurementPointOverlay = new LineSegments2(this.measurementPointOverlayGeometry, pointMeasureMaterial);
    this.measurementPointOverlay.frustumCulled = false;
    this.measurementPointOverlay.visible = false;
    this.measurementPointOverlay.renderOrder = 9;
    this.mesh.add(this.measurementPointOverlay);

    this.measurementGridOverlayGeometry = new LineSegmentsGeometry();
    const gridMaterial = new LineMaterial({
      color: '#6b7280',
      linewidth: 1.2,
      transparent: true,
      opacity: 0.42,
      depthTest: false,
      depthWrite: false,
    });
    this.updateSingleLineMaterialResolution(gridMaterial);
    this.measurementGridOverlay = new LineSegments2(this.measurementGridOverlayGeometry, gridMaterial);
    this.measurementGridOverlay.frustumCulled = false;
    this.measurementGridOverlay.visible = false;
    this.measurementGridOverlay.renderOrder = 6;
    this.mesh.add(this.measurementGridOverlay);

    this.measurementAxisOverlayGeometry = new LineSegmentsGeometry();
    const axisMaterial = new LineMaterial({
      color: '#111827',
      linewidth: 2.4,
      transparent: true,
      opacity: 0.7,
      depthTest: false,
      depthWrite: false,
    });
    this.updateSingleLineMaterialResolution(axisMaterial);
    this.measurementAxisOverlay = new LineSegments2(this.measurementAxisOverlayGeometry, axisMaterial);
    this.measurementAxisOverlay.frustumCulled = false;
    this.measurementAxisOverlay.visible = false;
    this.measurementAxisOverlay.renderOrder = 7;
    this.mesh.add(this.measurementAxisOverlay);

    this.measurementHeightPointMarker = new Mesh(
      new SphereGeometry(1, 16, 12),
      new MeshBasicMaterial({
        color: '#111827',
        transparent: true,
        opacity: 0.95,
        depthTest: false,
        depthWrite: false,
      }),
    );
    this.measurementHeightPointMarker.visible = false;
    this.measurementHeightPointMarker.renderOrder = 11;
    this.mesh.add(this.measurementHeightPointMarker);
    const pointMarkerMaterial = new MeshBasicMaterial({
      color: '#86efac',
      transparent: true,
      opacity: 0.96,
      depthTest: false,
      depthWrite: false,
    });
    this.measurementPointStartMarker = new Mesh(new SphereGeometry(1, 16, 12), pointMarkerMaterial.clone());
    this.measurementPointStartMarker.visible = false;
    this.measurementPointStartMarker.renderOrder = 12;
    this.mesh.add(this.measurementPointStartMarker);
    this.measurementPointEndMarker = new Mesh(new SphereGeometry(1, 16, 12), pointMarkerMaterial);
    this.measurementPointEndMarker.visible = false;
    this.measurementPointEndMarker.renderOrder = 12;
    this.mesh.add(this.measurementPointEndMarker);
    this.rebuildMeasurements();

    if (this.holeFillMode) {
      this.rebuildHoleLoopOverlays();
      this.updateHoleLoopOverlayVisibility();
    }

    this.applyUiTheme();
    if (options.resetView !== false) {
      this.resetView();
    }
    if (this.rotationOverlay) {
      this.rotationOverlay.visible = this.rotationOverlayVisible;
      this.updateRotationOverlayScale();
    }
    this.rebuildSelectionOverlay();
    this.emitHistory();
    this.emitSelection();
    this.emitBoundaryWorkflow();
    this.emitMeshStats();
    this.updateCursorVisuals();
  }

  private configureDisplayTexture(texture: Texture): void {
    texture.colorSpace = SRGBColorSpace;
    texture.magFilter = LinearFilter;
    texture.minFilter = LinearMipmapLinearFilter;
    texture.generateMipmaps = true;
    texture.anisotropy = Math.max(texture.anisotropy, this.renderer.capabilities.getMaxAnisotropy());
    texture.needsUpdate = true;
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
    this.disposeRotationOverlay();
    this.renderer.dispose();
    this.clearSceneMesh();
    this.measurementHoverLabel.remove();
    this.measurementHeightLabel.remove();
    this.measurementPointLabel.remove();
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
      copyGeometryUvs(this.editableMesh.geometry),
      this.faceMaterialIndices,
      this.bakedVertexColorsActive ? this.editableMesh.colors : null,
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
    if (this.bakedVertexColorsActive && nextMesh.colors) {
      applyVertexColors(editable, nextMesh.colors);
    }
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    const nextSessionId = this.allocateSessionId();
    const afterSnapshot = {
      sessionId: nextSessionId,
      positions: editable.positions.slice(),
      indices: editable.indices.slice(),
      referencePositions: editable.referencePositions.slice(),
      uvs: copyGeometryUvs(editable.geometry),
      colors: this.bakedVertexColorsActive ? editable.colors.slice() : null,
      bakedVertexColorsActive: this.bakedVertexColorsActive,
      history: engine.exportHistorySnapshot(),
      selectedTriangleMask: new Uint8Array(editable.triangleCount),
      selectedTriangleCount: 0,
      faceMaterialIndices: nextMesh.faceMaterialIndices,
      meshViewMode: this.meshViewMode,
      rotationSessionAngles: { ...this.rotationSessionAngles },
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
      faceMaterialIndices: nextMesh.faceMaterialIndices,
      bakedVertexColorsActive: this.bakedVertexColorsActive,
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
      uvs: copyGeometryUvs(editable.geometry),
      history: engine.exportHistorySnapshot(),
      selectedTriangleMask: refined.selectedTriangleMask.slice(),
      selectedTriangleCount,
      faceMaterialIndices: null,
      meshViewMode: this.meshViewMode,
      rotationSessionAngles: { ...this.rotationSessionAngles },
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
        uvs: copyGeometryUvs(editable.geometry),
      history: engine.exportHistorySnapshot(),
      selectedTriangleMask: remeshed.selectedTriangleMask.slice(),
      selectedTriangleCount,
      faceMaterialIndices: null,
      meshViewMode: this.meshViewMode,
      rotationSessionAngles: { ...this.rotationSessionAngles },
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
    if (!this.bakedVertexColorsActive) {
      recomputeDisplacementColorsRange(
        this.editableMesh.positions,
        this.editableMesh.referencePositions,
        this.editableMesh.normals,
        this.editableMesh.colors,
        0,
        this.editableMesh.vertexCount,
      );
    }
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
    if (!this.bakedVertexColorsActive) {
      recomputeDisplacementColorsRange(
        this.editableMesh.positions,
        this.editableMesh.referencePositions,
        this.editableMesh.normals,
        this.editableMesh.colors,
        0,
        this.editableMesh.vertexCount,
      );
    }
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
    this.setHoveredRotationRing(null);
    this.setHoveredMeasurementIndex(null);
    if (!this.activeStroke && !this.selectionGestureActive && this.cursor) {
      this.cursor.visible = false;
    }
  };

  private readonly handlePointerMove = (event: PointerEvent): void => {
    this.updatePointerFromEvent(event);

    if (this.measurementStartPickActive) {
      this.previewMeasurementStartAtPointer();
      event.preventDefault();
      return;
    }

    if (this.measurementPickActive) {
      this.previewMeasurementHeightAtPointer();
    }

    if (this.measurementPointPickActive) {
      this.previewPointToPointMeasurementAtPointer();
    }

    if (this.rotationDragAxis && event.pointerId === this.rotationDragPointerId) {
      this.updateRotationDrag(event);
      event.preventDefault();
      return;
    }

    this.updateRotationHover();
    this.updateMeasurementHoverFromPointer();

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

    if (this.tryBeginRotationDrag(event)) {
      event.preventDefault();
      return;
    }

    if (this.rotationOverlayVisible) {
      event.preventDefault();
      return;
    }

    if (this.measurementStartPickActive) {
      const measured = this.previewMeasurementStartAtPointer();
      if (measured) {
        this.measurementStartPickActive = false;
        this.callbacks.onMeasurementStartPickStateChange?.(false);
        this.callbacks.onMeasurementStartCaptured?.(Math.max(this.localHitPoint.z - this.measurementDistalZ, 0));
        this.updateMeasurementOverlayVisibility();
      }
      event.preventDefault();
      return;
    }

    if (this.measurementPickActive) {
      const measured = this.measureHeightAtPointer();
      if (measured) {
        this.measurementPickActive = false;
        this.callbacks.onMeasurementPickStateChange?.(false);
        this.updateMeasurementOverlayVisibility();
        this.updateMeasurementHeightLabel();
        this.updateCursorVisuals();
        this.updateHoleLoopOverlayVisibility();
        this.updateHoleHoverOverlay();
        this.rebuildSelectionOverlay();
        this.updatePositiveDirectionGuideForCurrentState();
      }
      event.preventDefault();
      return;
    }

    if (this.measurementPointPickActive) {
      this.capturePointToPointMeasurementAtPointer();
      event.preventDefault();
      return;
    }

    this.resetMeasurementForModelAction();

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
          complete: result.complete,
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
    if (this.rotationDragAxis && event.pointerId === this.rotationDragPointerId) {
      this.finishRotationDrag();
      event.preventDefault();
      return;
    }

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
      this.rebuildMeasurements();
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
    this.updateViewTransition();
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

    this.updateMeasurementHoverLabel();
    this.updateMeasurementHeightLabel();
    this.updateMeasurementPointLabel();
    this.drawSelectionPreview();
    this.emitViewCubeTransform();
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
        smoothOnlyTrimline: this.brushType === 'smooth' && this.smoothOnlyTrimline,
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
      smoothOnlyTrimline: this.brushType === 'smooth' && this.smoothOnlyTrimline,
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
    const visibleTriangleCount = this.filterVisibleTrianglesByCentroid(
      this.sculptEngine.data.regionTriangles,
      triangleCount,
    );
    this.applyTriangleSelection(
      this.sculptEngine.data.regionTriangles,
      visibleTriangleCount,
      this.selectionOperation,
    );
  }

  private applyScreenSelection(tool: SelectionTool): void {
    if (!this.editableMesh || !this.mesh || (tool !== 'box' && tool !== 'snip')) {
      return;
    }

    const triangleIds = new Uint32Array(this.editableMesh.triangleCount);
    let triangleCount = 0;

    const { indices, positions } = this.editableMesh;
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

      if (this.selectOnlyVisible) {
        this.selectionRayNdc.set(this.projectedPoint.x, this.projectedPoint.y);
        this.raycaster.setFromCamera(this.selectionRayNdc, this.camera);
        const hit = this.raycaster.intersectObject(this.mesh, false)[0];
        if (!hit || hit.faceIndex !== triangle) {
          continue;
        }
      }

      triangleIds[triangleCount] = triangle;
      triangleCount += 1;
    }

    this.applyTriangleSelection(triangleIds, triangleCount, this.selectionOperation);
  }

  private filterVisibleTrianglesByCentroid(triangleIds: Uint32Array, triangleCount: number): number {
    if (!this.selectOnlyVisible || !this.editableMesh || !this.mesh) {
      return triangleCount;
    }

    const { indices, positions } = this.editableMesh;
    let visibleTriangleCount = 0;
    for (let i = 0; i < triangleCount; i += 1) {
      const triangle = triangleIds[i];
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
      this.projectedPoint.copy(this.triangleWorldPoint).project(this.camera);
      if (this.projectedPoint.z < -1 || this.projectedPoint.z > 1) {
        continue;
      }

      this.selectionRayNdc.set(this.projectedPoint.x, this.projectedPoint.y);
      this.raycaster.setFromCamera(this.selectionRayNdc, this.camera);
      const hit = this.raycaster.intersectObject(this.mesh, false)[0];
      if (!hit || hit.faceIndex !== triangle) {
        continue;
      }

      triangleIds[visibleTriangleCount] = triangle;
      visibleTriangleCount += 1;
    }

    return visibleTriangleCount;
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
      !this.rotationOverlayVisible &&
      !this.isMeasurementToolActive() &&
      (this.interactionMode === 'sculpt' ||
        (this.interactionMode === 'select' && this.selectionTool === 'sphere'));

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

    if (this.rotationOverlayVisible || this.isMeasurementToolActive()) {
      this.cursor.visible = false;
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

  private updateViewTransition(): void {
    if (!this.viewTransition) {
      return;
    }

    const elapsed = performance.now() - this.viewTransition.startTime;
    const progress = Math.min(1, elapsed / this.viewTransition.duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    this.camera.position.lerpVectors(this.viewTransition.fromPosition, this.viewTransition.toPosition, eased);
    this.camera.up.lerpVectors(this.viewTransition.fromUp, this.viewTransition.toUp, eased).normalize();
    this.camera.lookAt(this.controls.target);

    if (progress >= 1) {
      this.camera.position.copy(this.viewTransition.toPosition);
      this.camera.up.copy(this.viewTransition.toUp);
      this.camera.lookAt(this.controls.target);
      this.viewTransition = null;
    }
  }

  private shortestTheta(from: number, to: number): number {
    let delta = to - from;
    while (delta > Math.PI) {
      delta -= Math.PI * 2;
    }
    while (delta < -Math.PI) {
      delta += Math.PI * 2;
    }

    return from + delta;
  }

  private emitViewCubeTransform(): void {
    if (!this.callbacks.onViewCubeTransform) {
      return;
    }

    const direction = this.camera.position.clone().sub(this.controls.target);
    const horizontal = Math.hypot(direction.x, direction.y);
    const rawAzimuth = Math.atan2(direction.x, -direction.y) * (180 / Math.PI);
    const azimuth = this.unwrapViewCubeAzimuth(rawAzimuth);
    const elevation = Math.atan2(direction.z, horizontal) * (180 / Math.PI);
    const transform = `rotateX(${(-elevation).toFixed(2)}deg) rotateY(${(-azimuth).toFixed(2)}deg)`;
    if (transform === this.lastViewCubeTransform) {
      return;
    }

    this.lastViewCubeTransform = transform;
    this.callbacks.onViewCubeTransform(transform);
  }

  private unwrapViewCubeAzimuth(azimuthDeg: number): number {
    if (this.lastViewCubeAzimuthDeg === null) {
      this.lastViewCubeAzimuthDeg = azimuthDeg;
      return azimuthDeg;
    }

    let delta = azimuthDeg - this.lastViewCubeAzimuthDeg;
    while (delta > 180) {
      delta -= 360;
    }
    while (delta < -180) {
      delta += 360;
    }

    this.lastViewCubeAzimuthDeg += delta;
    return this.lastViewCubeAzimuthDeg;
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
    if (!this.bakedVertexColorsActive) {
      recomputeDisplacementColorsRange(
        this.editableMesh.positions,
        this.editableMesh.referencePositions,
        this.editableMesh.normals,
        this.editableMesh.colors,
        0,
        this.editableMesh.vertexCount,
      );
    }
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
    this.rebuildMeasurements();
    this.emitMeshStats();
  }

  private tryBeginRotationDrag(event: PointerEvent): boolean {
    if (
      !this.rotationOverlayVisible ||
      !this.rotationOverlay ||
      !this.editableMesh ||
      !this.sculptEngine ||
      this.rotationRings.length === 0
    ) {
      return false;
    }

    this.beginRotationDraft();
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hits = this.raycaster.intersectObjects(this.rotationPickRings, false);
    const pickRing = hits[0]?.object;
    const axis = pickRing?.userData.axis as MeshRotationAxis | undefined;
    if (axis !== 'x' && axis !== 'y' && axis !== 'z') {
      return false;
    }

    const startVector = this.getRotationPlaneVectorFromPointer(axis);
    if (!startVector) {
      return false;
    }

    this.finishStroke();
    this.finishSelectionGesture();
    this.rotationDragAxis = axis;
    this.rotationDragPointerId = event.pointerId;
    this.rotationDragStartVector = startVector;
    this.rotationDragStartAngles = this.getDisplayedRotationAngles();
    this.setHoveredRotationRing(pickRing?.userData.visualRing instanceof Mesh ? pickRing.userData.visualRing : null);
    this.controls.enabled = false;
    this.renderer.domElement.setPointerCapture(event.pointerId);
    return true;
  }

  private updateRotationDrag(event: PointerEvent): void {
    if (!this.rotationDragAxis || !this.rotationDragStartVector) {
      return;
    }

    this.updatePointerFromEvent(event);
    const currentVector = this.getRotationPlaneVectorFromPointer(this.rotationDragAxis);
    if (!currentVector) {
      return;
    }

    const axisVector = getRotationAxisVector(this.rotationDragAxis);
    const angleDelta = signedAngleDegrees(this.rotationDragStartVector, currentVector, axisVector);
    this.setMeshRotationDraft(
      {
        ...this.rotationDragStartAngles,
        [this.rotationDragAxis]: this.rotationDragStartAngles[this.rotationDragAxis] + angleDelta,
      },
      true,
    );
  }

  private finishRotationDrag(): void {
    if (!this.rotationDragAxis) {
      return;
    }

    const shouldRebaseAfterRelease =
      this.rotationOverlayVisible &&
      Boolean(this.rotationDraftBasePositions) &&
      (
        Math.abs(this.rotationDraftAngles.x) > 0.0005 ||
        Math.abs(this.rotationDraftAngles.y) > 0.0005 ||
        Math.abs(this.rotationDraftAngles.z) > 0.0005
      );
    const pointerId = this.rotationDragPointerId;
    if (this.renderer.domElement.hasPointerCapture(pointerId)) {
      this.renderer.domElement.releasePointerCapture(pointerId);
    }

    this.controls.enabled = true;
    this.rotationDragAxis = null;
    this.rotationDragPointerId = -1;
    this.rotationDragStartVector = null;
    this.rotationDragStartAngles = this.getDisplayedRotationAngles();

    if (shouldRebaseAfterRelease) {
      this.commitActiveRotationDraft();
      this.beginRotationDraft();
    }
    this.updateRotationHover();
  }

  private updateRotationHover(): void {
    if (!this.rotationOverlayVisible || !this.rotationOverlay || this.rotationDragAxis) {
      if (!this.rotationDragAxis) {
        this.setHoveredRotationRing(null);
      }
      return;
    }

    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hits = this.raycaster.intersectObjects(this.rotationPickRings, false);
    const pickRing = hits[0]?.object;
    this.setHoveredRotationRing(pickRing?.userData.visualRing instanceof Mesh ? pickRing.userData.visualRing : null);
  }

  private setHoveredRotationRing(ring: Mesh | null): void {
    if (this.rotationHoveredRing === ring) {
      return;
    }

    if (this.rotationHoveredRing) {
      this.setRotationRingHoverState(this.rotationHoveredRing, false);
    }
    this.rotationHoveredRing = ring;
    if (this.rotationHoveredRing) {
      this.setRotationRingHoverState(this.rotationHoveredRing, true);
    }
  }

  private setRotationRingHoverState(ring: Mesh, hovered: boolean): void {
    const material = ring.material as MeshBasicMaterial;
    const baseColor = ring.userData.baseColor as string | undefined;
    if (!baseColor) {
      return;
    }

    material.color.set(baseColor);
    if (hovered) {
      material.color.lerp(new Color('#ffffff'), 0.34);
      material.opacity = 1;
      ring.scale.setScalar(1.022);
    } else {
      material.opacity = 0.86;
      ring.scale.setScalar(1);
    }
  }

  private beginRotationDraft(): void {
    if (!this.editableMesh || !this.sculptEngine || this.rotationDraftBasePositions) {
      return;
    }

    if (!this.rotationSessionCenter) {
      const boundsCenter = this.editableMesh.geometry.boundingSphere?.center ?? new Vector3();
      const boundsRadius = this.editableMesh.geometry.boundingSphere?.radius ?? this.editableMesh.boundsRadius;
      this.rotationSessionCenter = boundsCenter.clone();
      this.rotationSessionRadius = Math.max(boundsRadius * 1.18, 1);
    }

    this.rotationDraftAngles = { x: 0, y: 0, z: 0 };
    this.rotationDraftBeforeSnapshot = this.captureSessionSnapshot();
    this.rotationDraftBasePositions = this.editableMesh.positions.slice();
    this.rotationDraftBaseReferencePositions = this.editableMesh.referencePositions.slice();
    this.rotationDraftCenter = this.rotationSessionCenter.clone();
    this.rotationDraftRadius = this.rotationSessionRadius;
    this.callbacks.onRotationDraftChange?.(this.getDisplayedRotationAngles());
    this.updateRotationOverlayScale();
  }

  private finishRotationDraft(commit: boolean): void {
    this.finishRotationDrag();
    if (!this.rotationDraftBasePositions) {
      return;
    }

    const didCommit = commit ? this.commitActiveRotationDraft() : false;
    if (didCommit) {
      return;
    }

    const beforeSnapshot = this.rotationDraftBeforeSnapshot;
    this.rotationDraftBeforeSnapshot = null;
    this.rotationDraftBasePositions = null;
    this.rotationDraftBaseReferencePositions = null;
    this.rotationDraftCenter = null;
    this.rotationDraftRadius = 1;
    this.rotationDraftAngles = { x: 0, y: 0, z: 0 };
    this.callbacks.onRotationDraftChange?.(this.getDisplayedRotationAngles());

    if (!commit && beforeSnapshot) {
      this.applySessionSnapshot(beforeSnapshot, this.captureViewState());
    }
  }

  private commitActiveRotationDraft(): boolean {
    if (!this.rotationDraftBasePositions) {
      return false;
    }

    const beforeSnapshot = this.rotationDraftBeforeSnapshot;
    const committedAngles = { ...this.rotationDraftAngles };
    const didRotate =
      Math.abs(this.rotationDraftAngles.x) > 0.0005 ||
      Math.abs(this.rotationDraftAngles.y) > 0.0005 ||
      Math.abs(this.rotationDraftAngles.z) > 0.0005;
    this.rotationDraftBeforeSnapshot = null;
    this.rotationDraftBasePositions = null;
    this.rotationDraftBaseReferencePositions = null;
    this.rotationDraftCenter = null;
    this.rotationDraftRadius = 1;
    this.rotationDraftAngles = { x: 0, y: 0, z: 0 };

    if (!didRotate || !beforeSnapshot) {
      this.callbacks.onRotationDraftChange?.(this.getDisplayedRotationAngles());
      return false;
    }

    this.rotationSessionAngles = {
      x: this.rotationSessionAngles.x + committedAngles.x,
      y: this.rotationSessionAngles.y + committedAngles.y,
      z: this.rotationSessionAngles.z + committedAngles.z,
    };
    const afterSnapshot = this.captureSessionSnapshot();
    this.pushHistoryAction({
      kind: 'session',
      before: beforeSnapshot,
      after: afterSnapshot,
    });
    this.emitHistory();
    this.callbacks.onRotationDraftChange?.(this.getDisplayedRotationAngles());
    return true;
  }

  private getDisplayedRotationAngles(): Record<MeshRotationAxis, number> {
    return {
      x: this.rotationSessionAngles.x + this.rotationDraftAngles.x,
      y: this.rotationSessionAngles.y + this.rotationDraftAngles.y,
      z: this.rotationSessionAngles.z + this.rotationDraftAngles.z,
    };
  }

  private resetRotationSession(): void {
    this.rotationSessionAngles = { x: 0, y: 0, z: 0 };
    this.rotationSessionCenter = null;
    this.rotationSessionRadius = 1;
    this.callbacks.onRotationDraftChange?.(this.getDisplayedRotationAngles());
  }

  private applyRotationDraft(): void {
    if (
      !this.editableMesh ||
      !this.rotationDraftBasePositions ||
      !this.rotationDraftBaseReferencePositions ||
      !this.rotationDraftCenter
    ) {
      return;
    }

    rotatePositionsEulerInto(
      this.rotationDraftBasePositions,
      this.editableMesh.positions,
      this.rotationDraftAngles,
      this.rotationDraftCenter,
    );
    rotatePositionsEulerInto(
      this.rotationDraftBaseReferencePositions,
      this.editableMesh.referencePositions,
      this.rotationDraftAngles,
      this.rotationDraftCenter,
    );
    this.afterRigidGeometryTransform();
  }

  private getRotationPlaneVectorFromPointer(axis: MeshRotationAxis): Vector3 | null {
    const center = this.rotationDraftCenter ?? new Vector3();
    const normal = getRotationAxisVector(axis);
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const denominator = normal.dot(this.raycaster.ray.direction);
    if (Math.abs(denominator) <= 1e-5) {
      return null;
    }

    const distance = normal.dot(center.clone().sub(this.raycaster.ray.origin)) / denominator;
    if (!Number.isFinite(distance)) {
      return null;
    }

    const point = this.raycaster.ray.origin
      .clone()
      .addScaledVector(this.raycaster.ray.direction, distance);
    const vector = point.sub(center);
    vector.addScaledVector(normal, -vector.dot(normal));
    if (vector.lengthSq() <= 1e-8) {
      return null;
    }

    return vector.normalize();
  }

  private afterRigidGeometryTransform(): void {
    if (!this.editableMesh) {
      return;
    }

    recomputeAllNormals(
      this.editableMesh.positions,
      this.editableMesh.indices,
      this.editableMesh.faceNormals,
      this.editableMesh.normals,
      this.editableMesh.vertexFaceOffsets,
      this.editableMesh.vertexFaces,
    );
    if (!this.bakedVertexColorsActive) {
      recomputeDisplacementColorsRange(
        this.editableMesh.positions,
        this.editableMesh.referencePositions,
        this.editableMesh.normals,
        this.editableMesh.colors,
        0,
        this.editableMesh.vertexCount,
      );
    }
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

    if (this.activeBoundaryVertexIds) {
      this.boundaryGuide = captureBoundaryGuide(this.editableMesh.positions, this.activeBoundaryVertexIds);
    }

    this.updateRotationOverlayScale();
    this.rebuildMeasurements();
    this.emitMeshStats();
  }

  private ensureRotationOverlay(): void {
    if (this.rotationOverlay) {
      return;
    }

    const overlay = new Group();
    overlay.visible = false;
    overlay.renderOrder = 10;

    const plane = new Mesh(
      new PlaneGeometry(2, 2),
      new MeshBasicMaterial({
        color: '#8aa0b5',
        transparent: true,
        opacity: 0.12,
        side: DoubleSide,
        depthWrite: false,
      }),
    );
    plane.renderOrder = 1;
    overlay.add(plane);

    const grid = new GridHelper(2, 12, '#728196', '#b4bec8');
    grid.rotation.x = Math.PI / 2;
    grid.renderOrder = 2;
    const gridMaterial = Array.isArray(grid.material) ? grid.material : [grid.material];
    for (const material of gridMaterial) {
      material.transparent = true;
      material.opacity = 0.38;
      material.depthWrite = false;
    }
    overlay.add(grid);

    const xRing = this.createRotationRing('#d14646', 'x');
    xRing.rotation.y = Math.PI / 2;
    const yRing = this.createRotationRing('#2f9b62', 'y');
    yRing.rotation.x = Math.PI / 2;
    const zRing = this.createRotationRing('#2f6eea', 'z');
    const xPickRing = this.createRotationPickRing('x', xRing);
    xPickRing.rotation.copy(xRing.rotation);
    const yPickRing = this.createRotationPickRing('y', yRing);
    yPickRing.rotation.copy(yRing.rotation);
    const zPickRing = this.createRotationPickRing('z', zRing);
    this.rotationRings = [xRing, yRing, zRing];
    this.rotationPickRings = [xPickRing, yPickRing, zPickRing];
    overlay.add(xPickRing, yPickRing, zPickRing, xRing, yRing, zRing);

    this.rotationOverlay = overlay;
    this.scene.add(overlay);
    this.applyUiTheme();
  }

  private createRotationRing(
    color: string,
    axis: MeshRotationAxis,
  ): Mesh {
    const ring = new Mesh(
      new TorusGeometry(1, 0.012, 10, 128),
      new MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.86,
        depthTest: false,
        depthWrite: false,
      }),
    );
    ring.userData.axis = axis;
    ring.userData.baseColor = color;
    ring.renderOrder = 11;
    return ring;
  }

  private createRotationPickRing(
    axis: MeshRotationAxis,
    visualRing: Mesh,
  ): Mesh {
    const ring = new Mesh(
      new TorusGeometry(1.011, 0.055, 8, 96),
      new MeshBasicMaterial({
        color: '#ffffff',
        transparent: true,
        opacity: 0,
        depthTest: false,
        depthWrite: false,
      }),
    );
    ring.userData.axis = axis;
    ring.userData.visualRing = visualRing;
    ring.renderOrder = 10;
    return ring;
  }

  private updateRotationOverlayScale(): void {
    if (!this.rotationOverlay || !this.editableMesh) {
      return;
    }

    const radius =
      this.rotationDraftBasePositions && this.rotationDraftCenter
        ? this.rotationDraftRadius
        : Math.max(this.editableMesh.boundsRadius * 1.18, 1);
    const center =
      this.rotationDraftBasePositions && this.rotationDraftCenter
        ? this.rotationDraftCenter
        : (this.editableMesh.geometry.boundingSphere?.center ?? new Vector3());
    this.rotationOverlay.position.copy(center);
    this.rotationOverlay.scale.setScalar(radius);

    const bounds = computeAxisBoundsXYZ(this.editableMesh.positions);
    const gridLocalZ = (bounds.minZ - center.z) / radius;
    const plane = this.rotationOverlay.children[0];
    const grid = this.rotationOverlay.children[1];
    if (plane) {
      plane.position.z = gridLocalZ;
    }
    if (grid) {
      grid.position.z = gridLocalZ;
    }
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
    this.updateMeasurementMaterialResolution(width, height);
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
    const loops = buildOpenBoundaryLoopCandidates(editable.indices, editable.referencePositions);
    let bestLoop: Uint32Array | null = null;
    let bestScore = Infinity;

    for (let i = 0; i < loops.length; i += 1) {
      const loop = loops[i];
      const orderedVertexIds =
        loop.orderedVertexIds ?? createGuideOrderedBoundaryComponent(editable.positions, loop.segmentVertexPairs, guide);
      if (!orderedVertexIds || orderedVertexIds.length < 3) {
        continue;
      }

      const centerScore = this.scoreBoundaryLoopCenterAgainstGuide(
        editable.positions,
        orderedVertexIds,
        guide,
      );
      const pathScore = this.scoreBoundaryLoopAgainstGuide(
        editable.positions,
        orderedVertexIds,
        guide,
      );
      const score = centerScore + pathScore * 0.001;
      if (score < bestScore) {
        bestScore = score;
        bestLoop = orderedVertexIds;
      }
    }

    if (bestLoop) {
      return bestLoop.slice();
    }

    const visibleLoops = buildHoleLoopSet(editable.indices, editable.referencePositions).loops;
    for (let i = 0; i < visibleLoops.length; i += 1) {
      const loop = visibleLoops[i];
      if (!loop.isBoundaryLoop || !loop.orderedVertexIds) {
        continue;
      }

      const score = this.scoreBoundaryLoopCenterAgainstGuide(
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

  private scoreBoundaryLoopCenterAgainstGuide(
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
    for (let i = 0; i < orderedVertexIds.length; i += 1) {
      const offset = orderedVertexIds[i] * 3;
      loopCentroidX += positions[offset];
      loopCentroidY += positions[offset + 1];
      loopCentroidZ += positions[offset + 2];
    }
    const loopInvCount = 1 / orderedVertexIds.length;
    loopCentroidX *= loopInvCount;
    loopCentroidY *= loopInvCount;
    loopCentroidZ *= loopInvCount;

    const dx = loopCentroidX - guideCentroidX;
    const dy = loopCentroidY - guideCentroidY;
    const dz = loopCentroidZ - guideCentroidZ;
    return dx * dx + dy * dy + dz * dz;
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

    const geometry = createGeometryFromMeshArrays(snapshot.positions, snapshot.indices, snapshot.uvs);
    geometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });
    const editable = createEditableMeshData(geometry, {
      referencePositions: snapshot.referencePositions,
    });
    if (snapshot.bakedVertexColorsActive && snapshot.colors) {
      applyVertexColors(editable, snapshot.colors);
    }
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    engine.importHistorySnapshot(snapshot.history);
    return { editable, engine };
  }

  private capturePositiveLimbColorSource(): BakedColorSource | null {
    if (!this.editableMesh) {
      return null;
    }

    const uvs = copyGeometryUvs(this.editableMesh.geometry);
    const geometry = createGeometryFromMeshArrays(
      this.editableMesh.positions,
      this.editableMesh.indices,
      uvs,
    ) as BufferGeometry & { boundsTree?: MeshBVH };
    geometry.computeBoundsTree?.({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });
    if (!geometry.boundsTree) {
      geometry.dispose();
      return null;
    }

    const textureSampler = uvs && this.meshTexture ? createTextureColorSampler(this.meshTexture) : null;
    const sourceColors =
      this.bakedVertexColorsActive && this.editableMesh.colors.length >= this.editableMesh.vertexCount * 3
        ? this.editableMesh.colors.slice()
        : null;
    if (!textureSampler && !sourceColors) {
      geometry.dispose();
      return null;
    }

    return {
      geometry,
      boundsTree: geometry.boundsTree,
      positions: geometry.getAttribute('position')?.array as Float32Array,
      indices: geometry.getIndex()?.array as Uint32Array,
      uvs,
      colors: sourceColors,
      faceMaterialIndices: this.faceMaterialIndices?.slice() ?? null,
      textureSampler,
    };
  }

  private bakePositiveLimbVertexColors(editable: EditableMeshData, source: BakedColorSource): boolean {
    if (editable.vertexCount === 0 || source.indices.length < 3) {
      return false;
    }

    const bakedColors = new Float32Array(editable.vertexCount * 3);
    const targetPoint = new Vector3();
    const closestInfo: HitPointInfo = {
      point: new Vector3(),
      distance: 0,
      faceIndex: -1,
    };
    const touchesScan = new Uint8Array(editable.vertexCount);
    const touchesFill = new Uint8Array(editable.vertexCount);
    for (let triangle = 0; triangle < editable.triangleCount; triangle += 1) {
      const target = this.faceMaterialIndices?.[triangle] === 1 ? touchesFill : touchesScan;
      const triangleOffset = triangle * 3;
      target[editable.indices[triangleOffset]] = 1;
      target[editable.indices[triangleOffset + 1]] = 1;
      target[editable.indices[triangleOffset + 2]] = 1;
    }
    const color = new Vector3();
    let sampledCount = 0;

    for (let vertex = 0; vertex < editable.vertexCount; vertex += 1) {
      const offset = vertex * 3;
      if (touchesFill[vertex] !== 0 && touchesScan[vertex] === 0) {
        color.set(0.85, 0.87, 0.9);
      } else {
        targetPoint.set(editable.positions[offset], editable.positions[offset + 1], editable.positions[offset + 2]);
        const hit = source.boundsTree.closestPointToPoint(targetPoint, closestInfo);
        if (hit && hit.faceIndex >= 0) {
          sampleBakedColorFromSource(source, hit.faceIndex, hit.point, color);
          sampledCount += 1;
        } else {
          color.set(0.85, 0.87, 0.9);
        }
      }

      bakedColors[offset] = color.x;
      bakedColors[offset + 1] = color.y;
      bakedColors[offset + 2] = color.z;
    }

    if (sampledCount === 0) {
      return false;
    }

    applyVertexColors(editable, bakedColors);
    return true;
  }

  private refreshMeshMaterial(): void {
    if (!this.mesh) {
      return;
    }

    const nextMaterial = createMeshMaterials(
      this.meshTexture,
      this.sculptMatcapTexture,
      this.faceMaterialIndices,
      this.meshViewMode,
      this.bakedVertexColorsActive,
      this.uiTheme,
    );
    disposeMaterial(this.meshMaterial);
    this.meshMaterial = nextMaterial;
    this.mesh.material = nextMaterial;
  }

  private installAutomatedGeometry(
    geometry: BufferGeometry,
    viewState: ViewState,
    options: {
      weld?: boolean;
      bakedVertexColorsActive?: boolean;
      faceMaterialIndices?: Uint8Array | null;
      texture?: Texture | null;
    } = {},
  ): EditableMeshData {
    const installedGeometry = (options.weld === false ? geometry : weldGeometryByDistance(geometry)) as BufferGeometry & {
      computeBoundsTree?: (options?: { maxLeafSize?: number; setBoundingBox?: boolean; indirect?: boolean }) => unknown;
    };
    installedGeometry.computeBoundsTree?.({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });
    const editable = createEditableMeshData(installedGeometry);
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    this.installSession(editable, engine, {
      sessionId: this.allocateSessionId(),
      resetActionHistory: false,
      resetView: false,
      bakedVertexColorsActive: options.bakedVertexColorsActive,
      faceMaterialIndices: options.faceMaterialIndices,
      texture: options.texture,
    });
    this.restoreViewState(viewState);
    return editable;
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
    this.camera.up.set(0, 0, 1);
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
      uvs: copyGeometryUvs(this.editableMesh.geometry),
      colors: this.bakedVertexColorsActive ? this.editableMesh.colors.slice() : null,
      bakedVertexColorsActive: this.bakedVertexColorsActive,
      history: this.sculptEngine.exportHistorySnapshot(),
      selectedTriangleMask: this.selectedTriangleMask?.slice() ?? new Uint8Array(this.editableMesh.triangleCount),
      selectedTriangleCount: this.selectedTriangleCount,
      faceMaterialIndices: this.faceMaterialIndices?.slice() ?? null,
      meshViewMode: this.meshViewMode,
      rotationSessionAngles: { ...this.rotationSessionAngles },
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
      uvs: copyGeometryUvs(editable.geometry),
      colors: this.bakedVertexColorsActive ? editable.colors.slice() : null,
      bakedVertexColorsActive: this.bakedVertexColorsActive,
      history: engine.exportHistorySnapshot(),
      selectedTriangleMask: new Uint8Array(editable.triangleCount),
      selectedTriangleCount: 0,
      faceMaterialIndices: null,
      meshViewMode: this.meshViewMode,
      rotationSessionAngles: { ...this.rotationSessionAngles },
    };
  }

  private applySessionSnapshot(snapshot: SessionSnapshot, viewState: ViewState): void {
    if (!snapshot.positions || !snapshot.indices || !snapshot.referencePositions) {
      this.rotationSessionAngles = snapshot.rotationSessionAngles
        ? { ...snapshot.rotationSessionAngles }
        : { x: 0, y: 0, z: 0 };
      this.rotationSessionCenter = null;
      this.rotationSessionRadius = 1;
      this.clearCurrentSession(snapshot.sessionId);
      this.restoreViewState(viewState);
      this.callbacks.onRotationDraftChange?.(this.getDisplayedRotationAngles());
      this.emitHistory();
      this.emitSelection();
      this.emitMeshStats();
      return;
    }

    const geometry = createGeometryFromMeshArrays(snapshot.positions, snapshot.indices, snapshot.uvs);
    geometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });

    const editable = createEditableMeshData(geometry, {
      referencePositions: snapshot.referencePositions,
    });
    if (snapshot.bakedVertexColorsActive && snapshot.colors) {
      applyVertexColors(editable, snapshot.colors);
    }
    const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
    engine.importHistorySnapshot(snapshot.history);
    const snapshotViewMode = snapshot.meshViewMode ?? this.meshViewMode;
    const didChangeViewMode = this.meshViewMode !== snapshotViewMode;
    this.meshViewMode = snapshotViewMode;
    this.rotationSessionAngles = snapshot.rotationSessionAngles
      ? { ...snapshot.rotationSessionAngles }
      : { ...this.rotationSessionAngles };
    this.rotationSessionCenter = null;
    this.rotationSessionRadius = 1;
    this.installSession(editable, engine, {
      sessionId: snapshot.sessionId,
      resetActionHistory: false,
      resetView: false,
      selectedTriangleMask: snapshot.selectedTriangleMask,
      selectedTriangleCount: snapshot.selectedTriangleCount,
      faceMaterialIndices: snapshot.faceMaterialIndices,
      bakedVertexColorsActive: snapshot.bakedVertexColorsActive ?? false,
    });
    if (didChangeViewMode) {
      this.callbacks.onMeshViewModeChange?.(this.meshViewMode);
    }
    this.callbacks.onRotationDraftChange?.(this.getDisplayedRotationAngles());
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

  private updateMeasurementMaterialResolution(width: number, height: number): void {
    if (this.measurementOverlay) {
      this.updateSingleLineMaterialResolution(this.measurementOverlay.material as LineMaterial, width, height);
    }

    if (this.measurementHoverOverlay) {
      this.updateSingleLineMaterialResolution(this.measurementHoverOverlay.material as LineMaterial, width, height);
    }

    if (this.measurementHeightOverlay) {
      this.updateSingleLineMaterialResolution(
        this.measurementHeightOverlay.material as LineMaterial,
        width,
        height,
      );
    }

    if (this.measurementPointOverlay) {
      this.updateSingleLineMaterialResolution(
        this.measurementPointOverlay.material as LineMaterial,
        width,
        height,
      );
    }

    if (this.measurementGridOverlay) {
      this.updateSingleLineMaterialResolution(this.measurementGridOverlay.material as LineMaterial, width, height);
    }

    if (this.measurementAxisOverlay) {
      this.updateSingleLineMaterialResolution(this.measurementAxisOverlay.material as LineMaterial, width, height);
    }
  }

  private rebuildMeasurements(): void {
    if (!this.editableMesh) {
      this.measurementState = createEmptyMeasurementState();
      this.updateMeasurementOverlayVisibility();
      this.emitMeasurements();
      return;
    }

    const bounds = computeAxisBoundsZ(this.editableMesh.positions);
    this.measurementDistalZ = bounds.minZ;
    const totalHeightMm = Math.max(bounds.maxZ - bounds.minZ, 0);
    const startZ = Math.min(Math.max(this.measurementStartZ ?? bounds.minZ, bounds.minZ), bounds.maxZ);
    const rows: MeasurementState['rows'] = [];
    const sections: MeasurementSection[] = [];
    const overlayPositions: number[] = [];

    for (let z = startZ; z < bounds.maxZ - 1e-4; z += MEASUREMENT_SPACING_MM) {
      const distance = Math.max(z - bounds.minZ, 0);
      const section = computeSectionSegmentsAtZ(this.editableMesh.positions, this.editableMesh.indices, z);
      if (section.circumferenceMm <= 1e-3) {
        continue;
      }

      rows.push({
        distanceFromDistalMm: distance,
        circumferenceMm: section.circumferenceMm,
        zMm: z,
      });
      sections.push({
        distanceFromDistalMm: distance,
        circumferenceMm: section.circumferenceMm,
        zMm: z,
        positions: new Float32Array(section.positions),
      });
      overlayPositions.push(...section.positions);
    }
    this.measurementSections = sections;
    if (
      this.hoveredMeasurementIndex !== null &&
      (this.hoveredMeasurementIndex < 0 || this.hoveredMeasurementIndex >= this.measurementSections.length)
    ) {
      this.hoveredMeasurementIndex = null;
      this.callbacks.onMeasurementHoverChange?.(null);
    }

    this.measurementState = {
      rows,
      totalHeightMm,
      clickedHeightMm:
        this.measurementState.clickedHeightMm === null
          ? null
          : Math.min(Math.max(this.measurementState.clickedHeightMm, 0), totalHeightMm),
      pointToPointDistanceMm: this.measurementState.pointToPointDistanceMm,
    };

    this.replaceMeasurementLineGeometry(
      'measurement',
      overlayPositions.length > 0 ? new Float32Array(overlayPositions) : new Float32Array(0),
    );
    this.rebuildMeasurementHoverOverlay();
    this.rebuildMeasurementGridOverlay();

    this.rebuildMeasurementHeightOverlay();
    this.updateMeasurementOverlayVisibility();
    this.emitMeasurements();
  }

  private measureHeightAtPointer(): boolean {
    return this.updateMeasurementHeightAtPointer(true);
  }

  private previewMeasurementHeightAtPointer(): boolean {
    return this.updateMeasurementHeightAtPointer(false);
  }

  private previewPointToPointMeasurementAtPointer(): boolean {
    if (!this.measurementPointPickActive) {
      return false;
    }

    const point = this.getMeasurementHitPointAtPointer();
    if (!point) {
      return false;
    }

    this.measurementPointPreview = point;
    this.rebuildMeasurementHeightOverlay();
    this.updateMeasurementPointMarkers();
    this.updateMeasurementPointLabel();
    return true;
  }

  private capturePointToPointMeasurementAtPointer(): boolean {
    const point = this.getMeasurementHitPointAtPointer();
    if (!point) {
      return false;
    }

    if (!this.measurementPointStart) {
      this.measurementPointStart = point;
      this.measurementPointPreview = point.clone();
      this.rebuildMeasurementHeightOverlay();
      this.updateMeasurementPointMarkers();
      this.updateMeasurementPointLabel();
      return true;
    }

    this.measurementPointEnd = point;
    this.measurementPointPreview = null;
    const distanceMm = this.measurementPointStart.distanceTo(this.measurementPointEnd);
    this.measurementState = {
      ...this.measurementState,
      pointToPointDistanceMm: distanceMm,
    };
    this.measurementPointPickActive = false;
    this.callbacks.onPointToPointPickStateChange?.(false);
    this.rebuildMeasurementHeightOverlay();
    this.updateMeasurementPointMarkers();
    this.updateMeasurementPointLabel();
    this.emitMeasurements();
    this.callbacks.onPointToPointMeasurementCaptured?.(distanceMm);
    this.updateMeasurementOverlayVisibility();
    this.updateCursorVisuals();
    this.updateHoleLoopOverlayVisibility();
    this.updateHoleHoverOverlay();
    this.rebuildSelectionOverlay();
    this.updatePositiveDirectionGuideForCurrentState();
    return true;
  }

  private getMeasurementHitPointAtPointer(): Vector3 | null {
    if (!this.mesh || !this.editableMesh) {
      return null;
    }

    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit) {
      return null;
    }

    this.localHitPoint.copy(hit.point);
    this.mesh.worldToLocal(this.localHitPoint);
    return this.localHitPoint.clone();
  }

  private updateMeasurementHeightAtPointer(shouldCapture: boolean): boolean {
    const hitPoint = this.getMeasurementHitPointAtPointer();
    if (!hitPoint) {
      return false;
    }

    const clickedHeightMm = Math.min(
      Math.max(hitPoint.z - this.measurementDistalZ, 0),
      this.measurementState.totalHeightMm,
    );
    this.measurementState = {
      ...this.measurementState,
      clickedHeightMm,
    };
    this.measurementHeightPoint = hitPoint;
    this.rebuildMeasurementHeightOverlay();
    this.emitMeasurements();
    if (shouldCapture) {
      this.callbacks.onMeasurementCaptured?.(clickedHeightMm);
    }
    return true;
  }

  private previewMeasurementStartAtPointer(): boolean {
    if (!this.mesh || !this.editableMesh) {
      return false;
    }

    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit) {
      return false;
    }

    this.localHitPoint.copy(hit.point);
    this.mesh.worldToLocal(this.localHitPoint);
    this.measurementStartZ = this.localHitPoint.z;
    this.measurementCircumferenceVisible = true;
    this.rebuildMeasurements();
    this.callbacks.onMeasurementVisibilityChange?.(true);
    return true;
  }

  private rebuildMeasurementHeightOverlay(): void {
    if (!this.measurementHeightOverlayGeometry || !this.editableMesh) {
      return;
    }

    const heightPositions: number[] = [];
    if (this.measurementHeightPoint && this.measurementState.clickedHeightMm !== null) {
      const hitPoint = this.measurementHeightPoint;
      heightPositions.push(
        hitPoint.x,
        hitPoint.y,
        this.measurementDistalZ,
        hitPoint.x,
        hitPoint.y,
        hitPoint.z,
      );
    }

    this.replaceMeasurementLineGeometry('height', new Float32Array(heightPositions));
    this.rebuildMeasurementPointOverlay();

    if (this.measurementHeightPointMarker) {
      if (this.measurementHeightPoint && this.measurementState.clickedHeightMm !== null) {
        this.measurementHeightPointMarker.position.copy(this.measurementHeightPoint);
        const radius = Math.max((this.editableMesh.boundsRadius || 1) * 0.012, 0.8);
        this.measurementHeightPointMarker.scale.setScalar(radius);
        this.measurementHeightPointMarker.visible = true;
      } else {
        this.measurementHeightPointMarker.visible = false;
      }
    }
    this.updateMeasurementOverlayVisibility();
    this.updateMeasurementPointLabel();
  }

  private rebuildMeasurementPointOverlay(): void {
    if (!this.measurementPointOverlayGeometry || !this.editableMesh) {
      return;
    }

    const pointEnd = this.measurementPointPreview ?? this.measurementPointEnd;
    if (this.measurementPointStart && pointEnd) {
      this.replaceMeasurementLineGeometry(
        'point',
        new Float32Array([
          this.measurementPointStart.x,
          this.measurementPointStart.y,
          this.measurementPointStart.z,
          pointEnd.x,
          pointEnd.y,
          pointEnd.z,
        ]),
      );
      return;
    }

    this.replaceMeasurementLineGeometry('point', new Float32Array(0));
  }

  private resetMeasurementForModelAction(): void {
    if (
      !this.measurementCircumferenceVisible &&
      !this.measurementPickActive &&
      !this.measurementStartPickActive &&
      !this.measurementPointPickActive
    ) {
      return;
    }

    this.measurementCircumferenceVisible = false;
    this.cancelMeasurementPick();
    this.cancelMeasurementStartPick();
    this.cancelPointToPointMeasurementPick();
    this.clearMeasurementHeight();
    this.clearPointToPointMeasurement();
    this.updateMeasurementOverlayVisibility();
    this.callbacks.onMeasurementVisibilityChange?.(false);
    this.emitMeasurements();
  }

  private updateMeasurementOverlayVisibility(): void {
    const visible = this.editableMesh !== null && !this.rotationOverlayVisible;
    const measurementToolActive = this.isMeasurementToolActive();
    if (this.measurementOverlay) {
      this.measurementOverlay.visible =
        visible && this.measurementCircumferenceVisible && this.measurementState.rows.length > 0;
    }

    if (this.measurementHoverOverlay) {
      this.measurementHoverOverlay.visible =
        visible && this.measurementCircumferenceVisible && this.hoveredMeasurementIndex !== null;
    }

    if (this.measurementHeightOverlay) {
      this.measurementHeightOverlay.visible = visible && this.measurementState.clickedHeightMm !== null;
    }

    if (this.measurementPointOverlay) {
      this.measurementPointOverlay.visible =
        visible &&
        (this.measurementState.pointToPointDistanceMm !== null ||
          (this.measurementPointPickActive && this.measurementPointStart !== null));
    }

    if (this.measurementHeightPointMarker) {
      this.measurementHeightPointMarker.visible =
        visible && this.measurementState.clickedHeightMm !== null;
    }

    if (this.measurementGridOverlay) {
      this.measurementGridOverlay.visible = visible && measurementToolActive;
    }

    if (this.measurementAxisOverlay) {
      this.measurementAxisOverlay.visible = visible && measurementToolActive;
    }

    if (!visible || !this.measurementCircumferenceVisible || this.hoveredMeasurementIndex === null) {
      this.measurementHoverLabel.hidden = true;
    }
    if (!visible || this.measurementState.clickedHeightMm === null) {
      this.measurementHeightLabel.hidden = true;
    }
    if (
      !visible ||
      (!this.measurementPointPickActive && this.measurementState.pointToPointDistanceMm === null)
    ) {
      this.measurementPointLabel.hidden = true;
    }
    this.updateMeasurementPointMarkers();
  }

  private isMeasurementToolActive(): boolean {
    return (
      this.measurementCircumferenceVisible ||
      this.measurementPickActive ||
      this.measurementStartPickActive ||
      this.measurementPointPickActive
    );
  }

  private updatePositiveDirectionGuideForCurrentState(): void {
    if (this.interactionMode !== 'positive' || this.rotationOverlayVisible || this.isMeasurementToolActive()) {
      this.hidePositiveDirectionGuide();
      return;
    }

    this.showPositiveDirectionGuide();
  }

  private replaceMeasurementLineGeometry(
    target: 'measurement' | 'hover' | 'height' | 'point' | 'grid' | 'axis',
    positions: Float32Array,
  ): void {
    let overlay: LineSegments2 | null = null;
    let previous: LineSegmentsGeometry | null = null;
    const next = new LineSegmentsGeometry();
    next.setPositions(positions);

    if (target === 'measurement') {
      overlay = this.measurementOverlay;
      previous = this.measurementOverlayGeometry;
      this.measurementOverlayGeometry = next;
    } else if (target === 'hover') {
      overlay = this.measurementHoverOverlay;
      previous = this.measurementHoverOverlayGeometry;
      this.measurementHoverOverlayGeometry = next;
    } else if (target === 'height') {
      overlay = this.measurementHeightOverlay;
      previous = this.measurementHeightOverlayGeometry;
      this.measurementHeightOverlayGeometry = next;
    } else if (target === 'point') {
      overlay = this.measurementPointOverlay;
      previous = this.measurementPointOverlayGeometry;
      this.measurementPointOverlayGeometry = next;
    } else if (target === 'grid') {
      overlay = this.measurementGridOverlay;
      previous = this.measurementGridOverlayGeometry;
      this.measurementGridOverlayGeometry = next;
    } else {
      overlay = this.measurementAxisOverlay;
      previous = this.measurementAxisOverlayGeometry;
      this.measurementAxisOverlayGeometry = next;
    }

    if (overlay) {
      overlay.geometry = next;
    } else {
      next.dispose();
    }
    previous?.dispose();
  }

  private rebuildMeasurementHoverOverlay(): void {
    const section =
      this.hoveredMeasurementIndex === null ? null : this.measurementSections[this.hoveredMeasurementIndex] ?? null;
    this.replaceMeasurementLineGeometry('hover', section?.positions ?? new Float32Array(0));
    this.updateMeasurementOverlayVisibility();
  }

  private rebuildMeasurementGridOverlay(): void {
    if (!this.editableMesh) {
      this.replaceMeasurementLineGeometry('grid', new Float32Array(0));
      this.replaceMeasurementLineGeometry('axis', new Float32Array(0));
      return;
    }

    const bounds = computeAxisBoundsXYZ(this.editableMesh.positions);
    const spanX = Math.max(bounds.maxX - bounds.minX, 1);
    const spanY = Math.max(bounds.maxY - bounds.minY, 1);
    const padding = Math.max(Math.max(spanX, spanY) * 0.08, 5);
    const minX = bounds.minX - padding;
    const maxX = bounds.maxX + padding;
    const minY = bounds.minY - padding;
    const maxY = bounds.maxY + padding;
    const z = bounds.minZ;
    const step = chooseMeasurementGridStep(Math.max(spanX, spanY));
    const gridPositions: number[] = [];

    for (let x = Math.ceil(minX / step) * step; x <= maxX + 1e-6; x += step) {
      gridPositions.push(x, minY, z, x, maxY, z);
    }
    for (let y = Math.ceil(minY / step) * step; y <= maxY + 1e-6; y += step) {
      gridPositions.push(minX, y, z, maxX, y, z);
    }

    this.replaceMeasurementLineGeometry('grid', new Float32Array(gridPositions));
    this.replaceMeasurementLineGeometry(
      'axis',
      new Float32Array([
        minX,
        0,
        z,
        maxX,
        0,
        z,
        0,
        minY,
        z,
        0,
        maxY,
        z,
      ]),
    );
  }

  private updateMeasurementHoverFromPointer(): void {
    if (
      !this.pointerInside ||
      !this.measurementCircumferenceVisible ||
      this.measurementSections.length === 0 ||
      this.measurementStartPickActive ||
      this.measurementPickActive ||
      this.measurementPointPickActive
    ) {
      this.setHoveredMeasurementIndex(null);
      return;
    }

    let bestIndex: number | null = null;
    let bestDistanceSq = 16 * 16;
    for (let index = 0; index < this.measurementSections.length; index += 1) {
      const distanceSq = this.measurementSectionScreenDistanceSq(this.measurementSections[index]);
      if (distanceSq < bestDistanceSq) {
        bestDistanceSq = distanceSq;
        bestIndex = index;
      }
    }

    this.setHoveredMeasurementIndex(bestIndex);
  }

  private measurementSectionScreenDistanceSq(section: MeasurementSection): number {
    const positions = section.positions;
    let bestDistanceSq = Number.POSITIVE_INFINITY;
    const a = new Vector3();
    const b = new Vector3();
    const screenA = new Vector2();
    const screenB = new Vector2();
    for (let offset = 0; offset < positions.length; offset += 6) {
      a.set(positions[offset], positions[offset + 1], positions[offset + 2]);
      b.set(positions[offset + 3], positions[offset + 4], positions[offset + 5]);
      this.localPointToScreen(a, screenA);
      this.localPointToScreen(b, screenB);
      const distanceSq = pointToSegmentDistanceSq(this.pointerClient, screenA, screenB);
      if (distanceSq < bestDistanceSq) {
        bestDistanceSq = distanceSq;
      }
    }
    return bestDistanceSq;
  }

  private updateMeasurementHoverLabel(): void {
    const section =
      this.hoveredMeasurementIndex === null ? null : this.measurementSections[this.hoveredMeasurementIndex] ?? null;
    if (!section || !this.measurementCircumferenceVisible) {
      this.measurementHoverLabel.hidden = true;
      return;
    }

    this.measurementHoverLabel.textContent = `${section.circumferenceMm.toFixed(1)} mm`;
    const rect = this.container.getBoundingClientRect();
    const anchor = this.computeMeasurementLabelAnchor(section);
    const x = Math.min(Math.max(anchor.x - 112, 8), Math.max(rect.width - 104, 8));
    const y = Math.min(Math.max(anchor.y - 14, 8), Math.max(rect.height - 34, 8));
    this.measurementHoverLabel.style.left = `${x}px`;
    this.measurementHoverLabel.style.top = `${y}px`;
    this.measurementHoverLabel.hidden = false;
  }

  private updateMeasurementHeightLabel(): void {
    if (
      !this.measurementHeightPoint ||
      this.measurementState.clickedHeightMm === null ||
      !this.editableMesh ||
      this.rotationOverlayVisible
    ) {
      this.measurementHeightLabel.hidden = true;
      return;
    }

    this.measurementHeightLabel.textContent = `${this.measurementState.clickedHeightMm.toFixed(1)} mm`;
    const bottom = new Vector3(
      this.measurementHeightPoint.x,
      this.measurementHeightPoint.y,
      this.measurementDistalZ,
    );
    const middle = new Vector3().lerpVectors(bottom, this.measurementHeightPoint, 0.5);
    const screen = this.localPointToScreen(middle, new Vector2());
    const rect = this.container.getBoundingClientRect();
    const x = Math.min(Math.max(screen.x + 10, 8), Math.max(rect.width - 104, 8));
    const y = Math.min(Math.max(screen.y - 14, 8), Math.max(rect.height - 34, 8));
    this.measurementHeightLabel.style.left = `${x}px`;
    this.measurementHeightLabel.style.top = `${y}px`;
    this.measurementHeightLabel.hidden = false;
  }

  private updateMeasurementPointLabel(): void {
    const endPoint = this.measurementPointPreview ?? this.measurementPointEnd;
    if (
      !this.measurementPointStart ||
      !endPoint ||
      !this.editableMesh ||
      this.rotationOverlayVisible
    ) {
      this.measurementPointLabel.hidden = true;
      return;
    }

    const distanceMm =
      this.measurementState.pointToPointDistanceMm ?? this.measurementPointStart.distanceTo(endPoint);
    this.measurementPointLabel.textContent = `${distanceMm.toFixed(1)} mm`;
    const middle = new Vector3().lerpVectors(this.measurementPointStart, endPoint, 0.5);
    const screen = this.localPointToScreen(middle, new Vector2());
    const rect = this.container.getBoundingClientRect();
    const x = Math.min(Math.max(screen.x + 10, 8), Math.max(rect.width - 104, 8));
    const y = Math.min(Math.max(screen.y - 14, 8), Math.max(rect.height - 34, 8));
    this.measurementPointLabel.style.left = `${x}px`;
    this.measurementPointLabel.style.top = `${y}px`;
    this.measurementPointLabel.hidden = false;
  }

  private updateMeasurementPointMarkers(): void {
    const radius = Math.max((this.editableMesh?.boundsRadius || 1) * 0.012, 0.8);
    const startPoint = this.measurementPointStart ?? (this.measurementPointPickActive ? this.measurementPointPreview : null);
    const endPoint = this.measurementPointStart ? this.measurementPointPreview ?? this.measurementPointEnd : null;
    const visible = this.editableMesh !== null && !this.rotationOverlayVisible;

    if (this.measurementPointStartMarker) {
      if (visible && startPoint) {
        this.measurementPointStartMarker.position.copy(startPoint);
        this.measurementPointStartMarker.scale.setScalar(radius);
        this.measurementPointStartMarker.visible = true;
      } else {
        this.measurementPointStartMarker.visible = false;
      }
    }

    if (this.measurementPointEndMarker) {
      if (visible && endPoint) {
        this.measurementPointEndMarker.position.copy(endPoint);
        this.measurementPointEndMarker.scale.setScalar(radius);
        this.measurementPointEndMarker.visible = true;
      } else {
        this.measurementPointEndMarker.visible = false;
      }
    }
  }

  private computeMeasurementLabelAnchor(section: MeasurementSection): Vector2 {
    const positions = section.positions;
    const point = new Vector3();
    const screen = new Vector2();
    const anchor = new Vector2(this.pointerClient.x, this.pointerClient.y);
    let minX = Number.POSITIVE_INFINITY;

    for (let offset = 0; offset < positions.length; offset += 3) {
      point.set(positions[offset], positions[offset + 1], positions[offset + 2]);
      this.localPointToScreen(point, screen);
      if (!Number.isFinite(screen.x) || !Number.isFinite(screen.y)) {
        continue;
      }
      if (screen.x < minX) {
        minX = screen.x;
        anchor.copy(screen);
      }
    }

    return anchor;
  }

  private localPointToScreen(point: Vector3, target: Vector2): Vector2 {
    const world = point.clone();
    if (this.mesh) {
      this.mesh.localToWorld(world);
    }
    world.project(this.camera);
    return target.set(
      (world.x * 0.5 + 0.5) * this.overlayCanvas.clientWidth,
      (-world.y * 0.5 + 0.5) * this.overlayCanvas.clientHeight,
    );
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

  private emitMeasurements(): void {
    this.callbacks.onMeasurementChange?.(this.getMeasurementState());
  }

  private async reportPositiveLimbProgress(message: string, visible = true): Promise<void> {
    this.callbacks.onPositiveLimbProgress?.({ visible, message });
    await waitForBrowserPaint();
  }

  private showPositiveDirectionGuide(capPlaneZ?: number): void {
    this.hidePositiveDirectionGuide();
    if (!this.editableMesh) {
      return;
    }

    this.editableMesh.geometry.computeBoundingBox();
    const bounds = this.editableMesh.geometry.boundingBox;
    if (!bounds) {
      return;
    }

    const start = new Vector3(
      (bounds.min.x + bounds.max.x) * 0.5,
      (bounds.min.y + bounds.max.y) * 0.5,
      (bounds.min.z + bounds.max.z) * 0.5,
    );
    const endZ = Math.max(capPlaneZ ?? Number.NEGATIVE_INFINITY, bounds.max.z + POSITIVE_AUTO_Z_PLANE_OFFSET_MM);
    const length = endZ - start.z;
    if (!Number.isFinite(length) || length <= 1e-6) {
      return;
    }

    const group = new Group();
    group.renderOrder = 13;
    const direction = new Vector3(0, 0, 1);
    const color = '#1f6feb';
    const headLength = Math.min(Math.max(length * 0.08, 8), 18);
    const headWidth = Math.min(Math.max(headLength * 0.42, 3), 8);

    const mainArrow = new ArrowHelper(direction, start, length, color, headLength, headWidth);
    mainArrow.renderOrder = 13;
    this.configurePositiveDirectionArrow(mainArrow);
    group.add(mainArrow);

    for (const fraction of [0.38, 0.68]) {
      const markerLength = Math.min(Math.max(length * 0.14, 10), 24);
      const markerStart = start.clone().addScaledVector(direction, Math.max(length * fraction - markerLength, 0));
      const markerArrow = new ArrowHelper(
        direction,
        markerStart,
        markerLength,
        color,
        Math.min(headLength * 0.75, markerLength * 0.8),
        headWidth * 0.82,
      );
      markerArrow.renderOrder = 13;
      this.configurePositiveDirectionArrow(markerArrow);
      group.add(markerArrow);
    }

    this.positiveDirectionGuide = group;
    this.scene.add(group);
  }

  private configurePositiveDirectionArrow(arrow: ArrowHelper): void {
    arrow.traverse((object) => {
      if (!(object instanceof Mesh) && object.type !== 'Line') {
        return;
      }

      const material = (object as { material?: Material | Material[] }).material;
      const materials = Array.isArray(material) ? material : material ? [material] : [];
      for (const item of materials) {
        item.depthTest = false;
        item.depthWrite = false;
        item.transparent = true;
        item.opacity = 0.96;
      }
      object.renderOrder = 13;
    });
  }

  private hidePositiveDirectionGuide(): void {
    if (!this.positiveDirectionGuide) {
      return;
    }

    this.positiveDirectionGuide.removeFromParent();
    this.positiveDirectionGuide.traverse((object) => {
      if (object instanceof Mesh) {
        object.geometry.dispose();
        disposeMaterial(object.material);
      }
      const line = object as { line?: { geometry?: { dispose?: () => void }; material?: Material | Material[] } };
      line.line?.geometry?.dispose?.();
      disposeMaterial(line.line?.material ?? null);
    });
    this.positiveDirectionGuide = null;
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

    const holeLoopSet = buildHoleLoopSet(this.editableMesh.indices, this.editableMesh.referencePositions);
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
    this.applyUiTheme();

    return {
      loopCount: holeLoopSet.loops.length,
      edgeCount: holeLoopSet.edgeCount,
    };
  }

  private updateHoleLoopOverlayVisibility(): void {
    const visible =
      this.holeFillMode && !this.rotationOverlayVisible && !this.isMeasurementToolActive() && this.holeLoops.length > 0;
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
      this.rotationOverlayVisible ||
      this.isMeasurementToolActive() ||
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

    if (
      !this.holeFillMode ||
      this.rotationOverlayVisible ||
      this.isMeasurementToolActive() ||
      this.holeLoops.length === 0
    ) {
      this.holeLoopOverlayGeometry.setPositions(new Float32Array(0));
      this.holeLoopOverlay.visible = false;
      return;
    }

    this.holeLoopOverlayGeometry.setPositions(
      createLoopSegmentPositionArray(this.editableMesh.positions, this.holeLoops),
    );
    this.holeLoopOverlay.visible =
      this.holeLoops.length > 0 && !this.rotationOverlayVisible && !this.isMeasurementToolActive();
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
    if (
      this.rotationOverlayVisible ||
      this.isMeasurementToolActive() ||
      !this.holeFillMode ||
      highlightIndex < 0 ||
      !this.holeLoops[highlightIndex]
    ) {
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
    const resolvedLoop = resolveUsableBoundaryLoop(loop, this.editableMesh.referencePositions);
    if (!resolvedLoop) {
      this.callbacks.onHoleFill?.({
        success: false,
        message: diagnoseUnfillableHoleLoop(loop, this.editableMesh.referencePositions),
      });
      return false;
    }

    try {
      const fillMesh = createHoleFillMesh(
        this.editableMesh.positions,
        this.editableMesh.indices,
        this.editableMesh.referencePositions,
      );
      this.debugHoleFill('fill-kernel-start', {
        orderedVertexCount: resolvedLoop.orderedVertexIds.length,
      });
      const result = executeHoleFill(fillMesh, Array.from(resolvedLoop.orderedVertexIds), {
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
          message: describeHoleFillFailure(
            result.message,
            result.reason,
            resolvedLoop.loop,
            this.editableMesh.referencePositions,
          ),
        });
        return false;
      }

      const filledBoundaryVertexIds = Array.from(result.patch?.boundaryVertexIds ?? resolvedLoop.orderedVertexIds);
      const filledNewVertexIds = Array.from(result.patch?.newVertexIds ?? []);
      let closedSecondaryLoopCount = 0;
      for (const secondaryLoop of resolvedLoop.secondaryLoops) {
        const secondaryResult = executeHoleFill(fillMesh, Array.from(secondaryLoop.orderedVertexIds), {
          ignoreSharpFeatureValidation: true,
          debugLog: (stage, details) => this.debugHoleFill(`secondary-${stage}`, details),
        });
        if (!secondaryResult.success) {
          this.debugHoleFill('secondary-loop-fill-failed', {
            message: secondaryResult.message,
            reason: secondaryResult.reason,
            vertexCount: secondaryLoop.orderedVertexIds.length,
          });
          continue;
        }

        closedSecondaryLoopCount += 1;
        filledBoundaryVertexIds.push(...Array.from(secondaryResult.patch?.boundaryVertexIds ?? secondaryLoop.orderedVertexIds));
        filledNewVertexIds.push(...Array.from(secondaryResult.patch?.newVertexIds ?? []));
      }

      this.sculptEngine?.discardRedoHistory();
      const beforeSnapshot = this.captureSessionSnapshot();
      const viewState = this.captureViewState();
      const referencePositions = createHoleFillReferencePositions(
        this.editableMesh.referencePositions,
        fillMesh.positions,
        this.editableMesh.vertexCount,
      );
      const fillUvs = createHoleFillUvs(
        copyGeometryUvs(this.editableMesh.geometry),
        fillMesh.positions,
        filledBoundaryVertexIds,
        filledNewVertexIds,
      );
      const fillColors = this.bakedVertexColorsActive
        ? createHoleFillVertexColors(
            this.editableMesh.colors,
            fillMesh.positions,
            this.editableMesh.vertexCount,
            filledNewVertexIds,
          )
        : null;
      const geometry = createGeometryFromMeshArrays(fillMesh.positions, fillMesh.indices, fillUvs);
      const fillFaceMaterialIndices = createHoleFillFaceMaterialIndices(
        this.faceMaterialIndices,
        this.editableMesh.triangleCount,
        fillMesh.indices.length / 3,
      );
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
      if (this.bakedVertexColorsActive && fillColors) {
        applyVertexColors(editable, fillColors);
      }
      const engine = new SculptEngine(editable, SCULPT_HISTORY_LIMIT);
      const nextSessionId = this.allocateSessionId();
      const afterSnapshot = {
        sessionId: nextSessionId,
        positions: editable.positions.slice(),
        indices: editable.indices.slice(),
        referencePositions: editable.referencePositions.slice(),
        uvs: copyGeometryUvs(editable.geometry),
        colors: this.bakedVertexColorsActive ? editable.colors.slice() : null,
        bakedVertexColorsActive: this.bakedVertexColorsActive,
        history: engine.exportHistorySnapshot(),
        selectedTriangleMask: new Uint8Array(editable.triangleCount),
        selectedTriangleCount: 0,
        faceMaterialIndices: fillFaceMaterialIndices,
        meshViewMode: this.meshViewMode,
        rotationSessionAngles: { ...this.rotationSessionAngles },
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
        faceMaterialIndices: fillFaceMaterialIndices,
        bakedVertexColorsActive: this.bakedVertexColorsActive,
      });
      this.restoreViewState(viewState);
      this.debugHoleFill('fill-session-updated', {
        restoredCamera: true,
      });

      this.callbacks.onHoleFill?.({
        success: true,
        message:
          `${resolvedLoop.autoMessage ? `${resolvedLoop.autoMessage} ` : ''}` +
          `${closedSecondaryLoopCount > 0 ? `Closed ${closedSecondaryLoopCount} shorter touching boundary loop${closedSecondaryLoopCount === 1 ? '' : 's'}. ` : ''}` +
          result.message,
      });
      return true;
    } catch (error) {
      console.error(error);
      this.callbacks.onHoleFill?.({
        success: false,
        message: error instanceof Error ? error.message : 'Fill Hole failed unexpectedly.',
      });
      return false;
    }
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

    if (this.holeFillMode || this.rotationOverlayVisible || this.isMeasurementToolActive()) {
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
    this.selectionOverlay.visible =
      selectedTriangleCount > 0 && !this.rotationOverlayVisible && !this.isMeasurementToolActive();
    this.selectionDirty = false;
  }

  private drawSelectionPreview(): void {
    if (
      !this.selectionGestureActive ||
      this.selectionTool === 'sphere' ||
      this.isMeasurementToolActive() ||
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

  private clearSceneMesh(disposeTexture = true): void {
    this.finishRotationDraft(false);
    this.hidePositiveDirectionGuide();
    if (this.rotationOverlay) {
      this.rotationOverlay.visible = false;
    }

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

    if (this.measurementOverlay) {
      this.measurementOverlay.removeFromParent();
      this.measurementOverlayGeometry?.dispose();
      (this.measurementOverlay.material as LineMaterial).dispose();
      this.measurementOverlay = null;
      this.measurementOverlayGeometry = null;
    }

    if (this.measurementHoverOverlay) {
      this.measurementHoverOverlay.removeFromParent();
      this.measurementHoverOverlayGeometry?.dispose();
      (this.measurementHoverOverlay.material as LineMaterial).dispose();
      this.measurementHoverOverlay = null;
      this.measurementHoverOverlayGeometry = null;
    }

    if (this.measurementHeightOverlay) {
      this.measurementHeightOverlay.removeFromParent();
      this.measurementHeightOverlayGeometry?.dispose();
      (this.measurementHeightOverlay.material as LineMaterial).dispose();
      this.measurementHeightOverlay = null;
      this.measurementHeightOverlayGeometry = null;
    }
    if (this.measurementPointOverlay) {
      this.measurementPointOverlay.removeFromParent();
      this.measurementPointOverlayGeometry?.dispose();
      (this.measurementPointOverlay.material as LineMaterial).dispose();
      this.measurementPointOverlay = null;
      this.measurementPointOverlayGeometry = null;
    }

    if (this.measurementGridOverlay) {
      this.measurementGridOverlay.removeFromParent();
      this.measurementGridOverlayGeometry?.dispose();
      (this.measurementGridOverlay.material as LineMaterial).dispose();
      this.measurementGridOverlay = null;
      this.measurementGridOverlayGeometry = null;
    }

    if (this.measurementAxisOverlay) {
      this.measurementAxisOverlay.removeFromParent();
      this.measurementAxisOverlayGeometry?.dispose();
      (this.measurementAxisOverlay.material as LineMaterial).dispose();
      this.measurementAxisOverlay = null;
      this.measurementAxisOverlayGeometry = null;
    }

    if (this.measurementHeightPointMarker) {
      this.measurementHeightPointMarker.removeFromParent();
      this.measurementHeightPointMarker.geometry.dispose();
      disposeMaterial(this.measurementHeightPointMarker.material);
      this.measurementHeightPointMarker = null;
    }
    if (this.measurementPointStartMarker) {
      this.measurementPointStartMarker.removeFromParent();
      this.measurementPointStartMarker.geometry.dispose();
      disposeMaterial(this.measurementPointStartMarker.material);
      this.measurementPointStartMarker = null;
    }
    if (this.measurementPointEndMarker) {
      this.measurementPointEndMarker.removeFromParent();
      this.measurementPointEndMarker.geometry.dispose();
      disposeMaterial(this.measurementPointEndMarker.material);
      this.measurementPointEndMarker = null;
    }

    this.clearHoleLoopOverlays();

    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.mesh.removeFromParent();
      disposeMaterial(this.meshMaterial);
      if (disposeTexture) {
        this.meshTexture?.dispose();
      }
      this.mesh = null;
      this.meshMaterial = null;
      if (disposeTexture) {
        this.meshTexture = null;
      }
    }

    this.editableMesh = null;
    this.sculptEngine = null;
    this.hoverHit = null;
    this.selectedTriangleMask = null;
    this.selectedTriangleCount = 0;
    this.faceMaterialIndices = null;
    this.bakedVertexColorsActive = false;
    this.measurementState = createEmptyMeasurementState();
    this.measurementSections = [];
    this.setHoveredMeasurementIndex(null);
    this.measurementDistalZ = 0;
    this.measurementStartZ = null;
    this.measurementHeightPoint = null;
    this.measurementPointStart = null;
    this.measurementPointEnd = null;
    this.measurementPointPreview = null;
    if (this.measurementPickActive) {
      this.measurementPickActive = false;
      this.callbacks.onMeasurementPickStateChange?.(false);
    }
    if (this.measurementStartPickActive) {
      this.measurementStartPickActive = false;
      this.callbacks.onMeasurementStartPickStateChange?.(false);
    }
    if (this.measurementPointPickActive) {
      this.measurementPointPickActive = false;
      this.callbacks.onPointToPointPickStateChange?.(false);
    }
    this.emitMeasurements();
    this.selectionDirty = false;
  }

  private disposeRotationOverlay(): void {
    if (!this.rotationOverlay) {
      return;
    }

    this.setHoveredRotationRing(null);
    this.rotationOverlay.removeFromParent();
    this.rotationOverlay.traverse((object) => {
      if (object instanceof Mesh) {
        object.geometry.dispose();
        disposeMaterial(object.material);
      } else if (object instanceof GridHelper) {
        object.geometry.dispose();
        disposeMaterial(object.material);
      }
    });
    this.rotationOverlay = null;
    this.rotationRings = [];
    this.rotationPickRings = [];
    this.rotationHoveredRing = null;
  }
}

function disposeMaterial(material: Material | Material[] | null): void {
  if (Array.isArray(material)) {
    for (let i = 0; i < material.length; i += 1) {
      material[i].dispose();
    }
    return;
  }

  material?.dispose();
}

function waitForBrowserPaint(): Promise<void> {
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => {
      window.setTimeout(resolve, 0);
    });
  });
}

function createEmptyMeasurementState(): MeasurementState {
  return {
    rows: [],
    totalHeightMm: 0,
    clickedHeightMm: null,
    pointToPointDistanceMm: null,
  };
}

function rotatePositionsEulerInto(
  source: Float32Array,
  target: Float32Array,
  angles: Record<MeshRotationAxis, number>,
  center: Vector3,
): void {
  const rx = angles.x * Math.PI / 180;
  const ry = angles.y * Math.PI / 180;
  const rz = angles.z * Math.PI / 180;
  const cosX = Math.cos(rx);
  const sinX = Math.sin(rx);
  const cosY = Math.cos(ry);
  const sinY = Math.sin(ry);
  const cosZ = Math.cos(rz);
  const sinZ = Math.sin(rz);

  for (let offset = 0; offset < source.length; offset += 3) {
    let x = source[offset] - center.x;
    let y = source[offset + 1] - center.y;
    let z = source[offset + 2] - center.z;

    const yAfterX = y * cosX - z * sinX;
    const zAfterX = y * sinX + z * cosX;
    y = yAfterX;
    z = zAfterX;

    const xAfterY = x * cosY + z * sinY;
    const zAfterY = -x * sinY + z * cosY;
    x = xAfterY;
    z = zAfterY;

    const xAfterZ = x * cosZ - y * sinZ;
    const yAfterZ = x * sinZ + y * cosZ;
    target[offset] = xAfterZ + center.x;
    target[offset + 1] = yAfterZ + center.y;
    target[offset + 2] = z + center.z;
  }
}

function getRotationAxisVector(axis: MeshRotationAxis): Vector3 {
  if (axis === 'x') {
    return new Vector3(1, 0, 0);
  }
  if (axis === 'y') {
    return new Vector3(0, 1, 0);
  }
  return new Vector3(0, 0, 1);
}

function signedAngleDegrees(from: Vector3, to: Vector3, axis: Vector3): number {
  const cross = from.clone().cross(to);
  const sin = axis.dot(cross);
  const cos = Math.min(Math.max(from.dot(to), -1), 1);
  return Math.atan2(sin, cos) * 180 / Math.PI;
}

function pointToSegmentDistanceSq(point: Vector2, a: Vector2, b: Vector2): number {
  const abX = b.x - a.x;
  const abY = b.y - a.y;
  const lengthSq = abX * abX + abY * abY;
  if (lengthSq <= 1e-8) {
    return point.distanceToSquared(a);
  }

  const t = Math.min(Math.max(((point.x - a.x) * abX + (point.y - a.y) * abY) / lengthSq, 0), 1);
  const x = a.x + abX * t;
  const y = a.y + abY * t;
  const dx = point.x - x;
  const dy = point.y - y;
  return dx * dx + dy * dy;
}

function chooseMeasurementGridStep(span: number): number {
  if (span <= 80) {
    return 10;
  }
  if (span <= 220) {
    return 25;
  }
  return 50;
}

function computeHighestZ(
  positions: Float32Array,
  vertexIds: Uint32Array,
): number {
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < vertexIds.length; i += 1) {
    const offset = vertexIds[i] * 3;
    maxZ = Math.max(maxZ, positions[offset + 2]);
  }

  return Number.isFinite(maxZ) ? maxZ : 0;
}

function computeAxisBoundsZ(positions: Float32Array): { minZ: number; maxZ: number } {
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let offset = 2; offset < positions.length; offset += 3) {
    minZ = Math.min(minZ, positions[offset]);
    maxZ = Math.max(maxZ, positions[offset]);
  }

  if (!Number.isFinite(minZ) || !Number.isFinite(maxZ)) {
    return { minZ: 0, maxZ: 0 };
  }

  return { minZ, maxZ };
}

function computeAxisBoundsXYZ(
  positions: Float32Array,
): { minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number } {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let offset = 0; offset < positions.length; offset += 3) {
    minX = Math.min(minX, positions[offset]);
    maxX = Math.max(maxX, positions[offset]);
    minY = Math.min(minY, positions[offset + 1]);
    maxY = Math.max(maxY, positions[offset + 1]);
    minZ = Math.min(minZ, positions[offset + 2]);
    maxZ = Math.max(maxZ, positions[offset + 2]);
  }

  if (
    !Number.isFinite(minX) ||
    !Number.isFinite(maxX) ||
    !Number.isFinite(minY) ||
    !Number.isFinite(maxY) ||
    !Number.isFinite(minZ) ||
    !Number.isFinite(maxZ)
  ) {
    return { minX: 0, maxX: 0, minY: 0, maxY: 0, minZ: 0, maxZ: 0 };
  }

  return { minX, maxX, minY, maxY, minZ, maxZ };
}

function computeSectionSegmentsAtZ(
  positions: Float32Array,
  indices: Uint32Array,
  z: number,
): { circumferenceMm: number; positions: number[] } {
  const segmentPositions: number[] = [];
  let circumferenceMm = 0;

  for (let triangle = 0; triangle < indices.length / 3; triangle += 1) {
    const triOffset = triangle * 3;
    const intersections = collectTrianglePlaneIntersectionsZ(
      positions,
      indices[triOffset],
      indices[triOffset + 1],
      indices[triOffset + 2],
      z,
    );
    if (intersections.length < 2) {
      continue;
    }

    const pair = pickLongestPointPair(intersections);
    if (!pair) {
      continue;
    }

    const [a, b] = pair;
    const length = a.distanceTo(b);
    if (length <= 1e-5) {
      continue;
    }

    circumferenceMm += length;
    segmentPositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }

  return { circumferenceMm, positions: segmentPositions };
}

function collectTrianglePlaneIntersectionsZ(
  positions: Float32Array,
  a: number,
  b: number,
  c: number,
  z: number,
): Vector3[] {
  const vertices = [
    new Vector3(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]),
    new Vector3(positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]),
    new Vector3(positions[c * 3], positions[c * 3 + 1], positions[c * 3 + 2]),
  ];
  const intersections: Vector3[] = [];
  const seen = new Set<string>();
  for (let edge = 0; edge < 3; edge += 1) {
    const start = vertices[edge];
    const end = vertices[(edge + 1) % 3];
    const startDistance = start.z - z;
    const endDistance = end.z - z;
    if (Math.abs(startDistance) <= 1e-7 && Math.abs(endDistance) <= 1e-7) {
      continue;
    }

    if (startDistance * endDistance > 0) {
      continue;
    }

    const denominator = start.z - end.z;
    if (Math.abs(denominator) <= 1e-10) {
      continue;
    }

    const t = (start.z - z) / denominator;
    if (t < -1e-7 || t > 1 + 1e-7) {
      continue;
    }

    const point = new Vector3().lerpVectors(start, end, Math.min(Math.max(t, 0), 1));
    point.z = z;
    const key = `${point.x.toFixed(5)},${point.y.toFixed(5)},${point.z.toFixed(5)}`;
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    intersections.push(point);
  }

  return intersections;
}

function pickLongestPointPair(points: Vector3[]): [Vector3, Vector3] | null {
  let bestPair: [Vector3, Vector3] | null = null;
  let bestDistanceSq = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      const distanceSq = points[i].distanceToSquared(points[j]);
      if (distanceSq > bestDistanceSq) {
        bestDistanceSq = distanceSq;
        bestPair = [points[i], points[j]];
      }
    }
  }

  return bestPair;
}

interface ObjSerializationOptions {
  objectName: string;
  materialFilename: string;
  unit: MeshExportUnit;
  coordinateScale: number;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  uvs: Float32Array | null;
  triangleUvIndices: Uint32Array | null;
  faceMaterialIndices: Uint8Array | null;
  scanMaterialName: string;
  fillMaterialName: string;
}

interface MtlSerializationOptions {
  scanMaterialName: string;
  fillMaterialName: string;
  textureFilename: string | null;
}

interface ObjTextureAtlasOptions {
  vertexCount: number;
  indices: Uint32Array;
  sourceUvs: Float32Array | null;
  sourceColors: Float32Array | null;
  textureSampler: TextureColorSampler | null;
  faceMaterialIndices: Uint8Array | null;
}

interface ObjTextureAtlas {
  blob: Blob;
  uvs: Float32Array;
  triangleUvIndices: Uint32Array | null;
}

function createObjSourceTextureExport(texture: Texture, uvs: Float32Array): ObjTextureAtlas {
  const image = texture.image as CanvasImageSource | undefined;
  const width = getCanvasImageWidth(image);
  const height = getCanvasImageHeight(image);
  if (!image || width <= 0 || height <= 0) {
    throw new Error('The source texture image is not available for PNG export.');
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('The source PNG texture canvas could not be created.');
  }
  context.drawImage(image, 0, 0, width, height);
  return {
    blob: canvasDataUrlToBlob(canvas.toDataURL('image/png')),
    uvs,
    triangleUvIndices: null,
  };
}

function createObjTextureAtlas(options: ObjTextureAtlasOptions): ObjTextureAtlas {
  const triangleCount = Math.floor(options.indices.length / 3);
  if (triangleCount === 0) {
    throw new Error('The mesh has no triangles to texture.');
  }

  const tileColumns = Math.ceil(Math.sqrt(triangleCount));
  const tileRows = Math.ceil(triangleCount / tileColumns);
  const availableTileSize = Math.floor(OBJ_TEXTURE_ATLAS_MAX_SIZE / Math.max(tileColumns, tileRows));
  const tileSize = Math.min(OBJ_TEXTURE_ATLAS_PREFERRED_TILE_SIZE, availableTileSize);
  if (tileSize < OBJ_TEXTURE_ATLAS_MIN_TILE_SIZE) {
    throw new Error(
      `The mesh has too many triangles for a ${OBJ_TEXTURE_ATLAS_MAX_SIZE}px OBJ texture atlas. Export STL or reduce the mesh density first.`,
    );
  }

  const width = tileColumns * tileSize;
  const height = tileRows * tileSize;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('The PNG texture canvas could not be created.');
  }

  const image = context.createImageData(width, height);
  const vertexColors = resolveObjExportVertexColors(options);
  const uvs = new Float32Array(triangleCount * 6);
  const triangleUvIndices = new Uint32Array(triangleCount * 3);
  const cornerInset = tileSize >= 6 ? 1.5 : 0.75;
  const cornerSpan = tileSize - cornerInset * 2;

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const tileX = (triangle % tileColumns) * tileSize;
    const tileY = Math.floor(triangle / tileColumns) * tileSize;
    const triangleOffset = triangle * 3;
    const a = options.indices[triangleOffset];
    const b = options.indices[triangleOffset + 1];
    const c = options.indices[triangleOffset + 2];
    const isFillFace = options.faceMaterialIndices?.[triangle] === 1;

    for (let localY = 0; localY < tileSize; localY += 1) {
      for (let localX = 0; localX < tileSize; localX += 1) {
        const rawB = (localX + 0.5 - cornerInset) / cornerSpan;
        const rawC = (localY + 0.5 - cornerInset) / cornerSpan;
        const rawA = 1 - rawB - rawC;
        const clampedA = clamp01(rawA);
        const clampedB = clamp01(rawB);
        const clampedC = clamp01(rawC);
        const weightTotal = clampedA + clampedB + clampedC || 1;
        const weightA = clampedA / weightTotal;
        const weightB = clampedB / weightTotal;
        const weightC = clampedC / weightTotal;
        const pixelOffset = ((tileY + localY) * width + tileX + localX) * 4;

        if (isFillFace) {
          image.data[pixelOffset] = 217;
          image.data[pixelOffset + 1] = 222;
          image.data[pixelOffset + 2] = 230;
        } else {
          writeInterpolatedAtlasColor(
            image.data,
            pixelOffset,
            vertexColors,
            options.vertexCount,
            a,
            b,
            c,
            weightA,
            weightB,
            weightC,
          );
        }
        image.data[pixelOffset + 3] = 255;
      }
    }

    const uvOffset = triangle * 6;
    const leftU = (tileX + cornerInset) / width;
    const rightU = (tileX + tileSize - cornerInset) / width;
    const topV = 1 - (tileY + cornerInset) / height;
    const bottomV = 1 - (tileY + tileSize - cornerInset) / height;
    uvs[uvOffset] = leftU;
    uvs[uvOffset + 1] = topV;
    uvs[uvOffset + 2] = rightU;
    uvs[uvOffset + 3] = topV;
    uvs[uvOffset + 4] = leftU;
    uvs[uvOffset + 5] = bottomV;
    triangleUvIndices[triangleOffset] = triangleOffset;
    triangleUvIndices[triangleOffset + 1] = triangleOffset + 1;
    triangleUvIndices[triangleOffset + 2] = triangleOffset + 2;
  }

  context.putImageData(image, 0, 0);
  return {
    blob: canvasDataUrlToBlob(canvas.toDataURL('image/png')),
    uvs,
    triangleUvIndices,
  };
}

function resolveObjExportVertexColors(options: ObjTextureAtlasOptions): Float32Array | null {
  if (options.sourceColors && options.sourceColors.length >= options.vertexCount * 3) {
    return options.sourceColors;
  }
  if (!options.textureSampler || !options.sourceUvs || options.sourceUvs.length < options.vertexCount * 2) {
    return null;
  }

  const colors = new Float32Array(options.vertexCount * 3);
  const sampledColor = new Vector3();
  for (let vertex = 0; vertex < options.vertexCount; vertex += 1) {
    const uvOffset = vertex * 2;
    sampleTextureColor(options.textureSampler, options.sourceUvs[uvOffset], options.sourceUvs[uvOffset + 1], sampledColor);
    const colorOffset = vertex * 3;
    colors[colorOffset] = sampledColor.x;
    colors[colorOffset + 1] = sampledColor.y;
    colors[colorOffset + 2] = sampledColor.z;
  }
  return colors;
}

function writeInterpolatedAtlasColor(
  output: Uint8ClampedArray,
  outputOffset: number,
  colors: Float32Array | null,
  vertexCount: number,
  a: number,
  b: number,
  c: number,
  weightA: number,
  weightB: number,
  weightC: number,
): void {
  if (!colors || a >= vertexCount || b >= vertexCount || c >= vertexCount) {
    output[outputOffset] = 255;
    output[outputOffset + 1] = 255;
    output[outputOffset + 2] = 255;
    return;
  }

  const ao = a * 3;
  const bo = b * 3;
  const co = c * 3;
  output[outputOffset] = Math.round(
    clamp01(colors[ao] * weightA + colors[bo] * weightB + colors[co] * weightC) * 255,
  );
  output[outputOffset + 1] = Math.round(
    clamp01(colors[ao + 1] * weightA + colors[bo + 1] * weightB + colors[co + 1] * weightC) * 255,
  );
  output[outputOffset + 2] = Math.round(
    clamp01(colors[ao + 2] * weightA + colors[bo + 2] * weightB + colors[co + 2] * weightC) * 255,
  );
}

function canvasDataUrlToBlob(dataUrl: string): Blob {
  const separatorIndex = dataUrl.indexOf(',');
  if (separatorIndex < 0) {
    throw new Error('The PNG texture could not be encoded.');
  }

  const binary = window.atob(dataUrl.slice(separatorIndex + 1));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: 'image/png' });
}

function serializeAsciiStl(
  name: string,
  positions: Float32Array,
  indices: Uint32Array,
  coordinateScale: number,
): string {
  const lines = [`solid ${name}`];
  for (let triangle = 0; triangle < indices.length / 3; triangle += 1) {
    const triOffset = triangle * 3;
    const a = indices[triOffset];
    const b = indices[triOffset + 1];
    const c = indices[triOffset + 2];
    const normal = computeFacetNormal(positions, a, b, c);
    lines.push(
      `  facet normal ${formatExportNumber(normal.x)} ${formatExportNumber(normal.y)} ${formatExportNumber(normal.z)}`,
      '    outer loop',
      `      vertex ${formatVertex(positions, a, coordinateScale)}`,
      `      vertex ${formatVertex(positions, b, coordinateScale)}`,
      `      vertex ${formatVertex(positions, c, coordinateScale)}`,
      '    endloop',
      '  endfacet',
    );
  }

  lines.push(`endsolid ${name}`);
  return `${lines.join('\n')}\n`;
}

function serializeObj(options: ObjSerializationOptions): string {
  const { positions, normals, indices, uvs, triangleUvIndices, faceMaterialIndices } = options;
  const vertexCount = positions.length / 3;
  const textureCoordinateCount = uvs ? Math.floor(uvs.length / 2) : 0;
  const hasUvs = Boolean(
    uvs &&
      textureCoordinateCount > 0 &&
      (triangleUvIndices
        ? triangleUvIndices.length >= indices.length
        : textureCoordinateCount >= vertexCount),
  );
  const hasNormals = normals.length >= vertexCount * 3;
  const lines = [
    '# Exported by NouraSoft',
    `# Coordinate units: ${formatExportUnitComment(options.unit)}`,
    `mtllib ${options.materialFilename}`,
    `o ${options.objectName}`,
  ];

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    lines.push(`v ${formatVertex(positions, vertex, options.coordinateScale)}`);
  }

  if (hasUvs && uvs) {
    for (let textureCoordinate = 0; textureCoordinate < textureCoordinateCount; textureCoordinate += 1) {
      const offset = textureCoordinate * 2;
      lines.push(`vt ${formatExportNumber(uvs[offset])} ${formatExportNumber(uvs[offset + 1])}`);
    }
  }

  if (hasNormals) {
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      lines.push(`vn ${formatVertex(normals, vertex)}`);
    }
    lines.push('s 1');
  }

  const triangleCount = Math.floor(indices.length / 3);
  const triangleOrder = createObjTriangleOrder(triangleCount, faceMaterialIndices);
  let currentMaterial = '';
  for (let orderIndex = 0; orderIndex < triangleOrder.length; orderIndex += 1) {
    const triangle = triangleOrder[orderIndex];
    const materialName = faceMaterialIndices?.[triangle] === 1 ? options.fillMaterialName : options.scanMaterialName;
    if (materialName !== currentMaterial) {
      lines.push(`usemtl ${materialName}`);
      currentMaterial = materialName;
    }

    const triOffset = triangle * 3;
    const a = indices[triOffset] + 1;
    const b = indices[triOffset + 1] + 1;
    const c = indices[triOffset + 2] + 1;
    const uvA = (triangleUvIndices?.[triOffset] ?? indices[triOffset]) + 1;
    const uvB = (triangleUvIndices?.[triOffset + 1] ?? indices[triOffset + 1]) + 1;
    const uvC = (triangleUvIndices?.[triOffset + 2] ?? indices[triOffset + 2]) + 1;
    if (hasUvs && hasNormals) {
      lines.push(`f ${a}/${uvA}/${a} ${b}/${uvB}/${b} ${c}/${uvC}/${c}`);
    } else if (hasUvs) {
      lines.push(`f ${a}/${uvA} ${b}/${uvB} ${c}/${uvC}`);
    } else if (hasNormals) {
      lines.push(`f ${a}//${a} ${b}//${b} ${c}//${c}`);
    } else {
      lines.push(`f ${a} ${b} ${c}`);
    }
  }

  return `${lines.join('\n')}\n`;
}

function createObjTriangleOrder(triangleCount: number, faceMaterialIndices: Uint8Array | null): number[] {
  if (!faceMaterialIndices || faceMaterialIndices.length < triangleCount) {
    return Array.from({ length: triangleCount }, (_, triangle) => triangle);
  }

  const scanTriangles: number[] = [];
  const fillTriangles: number[] = [];
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    if (faceMaterialIndices[triangle] === 1) {
      fillTriangles.push(triangle);
    } else {
      scanTriangles.push(triangle);
    }
  }
  return [...scanTriangles, ...fillTriangles];
}

function serializeMtl(options: MtlSerializationOptions): string {
  const lines = [
    '# Exported by NouraSoft',
    `newmtl ${options.scanMaterialName}`,
    'Ka 1.000000 1.000000 1.000000',
    'Kd 1.000000 1.000000 1.000000',
    'Ks 0.000000 0.000000 0.000000',
    'Ns 0.000000',
    'd 1.000000',
    'illum 1',
  ];

  if (options.textureFilename) {
    lines.push(`map_Kd ${options.textureFilename}`);
  }

  lines.push(
    '',
    `newmtl ${options.fillMaterialName}`,
    'Ka 0.850000 0.870000 0.900000',
    'Kd 0.850000 0.870000 0.900000',
    'Ks 0.000000 0.000000 0.000000',
    'Ns 0.000000',
    'd 1.000000',
    'illum 1',
  );

  return `${lines.join('\n')}\n`;
}

function computeFacetNormal(positions: Float32Array, a: number, b: number, c: number): Vector3 {
  const ax = positions[a * 3];
  const ay = positions[a * 3 + 1];
  const az = positions[a * 3 + 2];
  const bx = positions[b * 3];
  const by = positions[b * 3 + 1];
  const bz = positions[b * 3 + 2];
  const cx = positions[c * 3];
  const cy = positions[c * 3 + 1];
  const cz = positions[c * 3 + 2];

  const normal = new Vector3(
    (by - ay) * (cz - az) - (bz - az) * (cy - ay),
    (bz - az) * (cx - ax) - (bx - ax) * (cz - az),
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax),
  );
  if (normal.lengthSq() <= 1e-16) {
    return normal.set(0, 0, 0);
  }

  return normal.normalize();
}

function formatVertex(positions: Float32Array, vertex: number, scale = 1): string {
  const offset = vertex * 3;
  return `${formatExportNumber(positions[offset] * scale)} ${formatExportNumber(
    positions[offset + 1] * scale,
  )} ${formatExportNumber(
    positions[offset + 2] * scale,
  )}`;
}

function getMillimeterExportScale(unit: MeshExportUnit): number {
  switch (unit) {
    case 'cm':
      return 0.1;
    case 'm':
      return 0.001;
    case 'in':
      return 1 / 25.4;
    default:
      return 1;
  }
}

function formatExportUnitComment(unit: MeshExportUnit): string {
  switch (unit) {
    case 'cm':
      return 'centimetres (cm)';
    case 'm':
      return 'metres (m)';
    case 'in':
      return 'inches (in)';
    default:
      return 'millimetres (mm)';
  }
}

function formatExportNumber(value: number): string {
  return Number.isFinite(value) ? Number(value.toFixed(6)).toString() : '0';
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.min(Math.max(value, 0), 1);
}

function sanitizeExportName(name: string): string {
  const withoutExtension = name.replace(/\.[^.\\/]+$/, '');
  const sanitized = withoutExtension.replace(/[<>:"/\\|?*\u0000-\u001f]+/g, '_').trim();
  return sanitized || 'NouraSoft_export';
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

function diagnoseUnfillableHoleLoop(loop: HoleLoop | undefined, positions: Float32Array): string {
  if (!loop) {
    return 'No highlighted contour was found. Move the cursor directly over a blue open edge until it turns purple, then click.';
  }

  const topology = analyzeHoleLoopTopology(loop, positions);
  if (loop.edgeCount < 3 || loop.boundaryEdgeCount < 3) {
    return `Selected contour has only ${loop.boundaryEdgeCount} usable open edge${loop.boundaryEdgeCount === 1 ? '' : 's'}. Fill Hole needs at least 3 connected boundary edges.`;
  }

  if (topology.nonBoundaryEdgeCount > 0) {
    return `Selected contour includes ${topology.nonBoundaryEdgeCount} non-manifold edge${topology.nonBoundaryEdgeCount === 1 ? '' : 's'} that are not true open boundary edges. One or more edges are shared by too many faces, often from duplicate faces or a remesh/delete artifact.`;
  }

  if (topology.highDegreeVertices.length > 0) {
    const maxDegree = Math.max(...topology.highDegreeVertices.map((entry) => entry.degree));
    if (maxDegree >= 4) {
      return `Two holes or cracks appear to touch at a point. ${topology.highDegreeVertices.length} junction vertex${topology.highDegreeVertices.length === 1 ? '' : 'es'} has more than two boundary edges, so this is not one clean ring.`;
    }

    return `A small branch or broken edge segment is attached to the main contour. ${topology.highDegreeVertices.length} junction vertex${topology.highDegreeVertices.length === 1 ? '' : 'es'} connects to more than two contour edges.`;
  }

  if (topology.endpointCount > 0) {
    return `Selected contour is broken instead of closed. It has ${topology.endpointCount} loose endpoint${topology.endpointCount === 1 ? '' : 's'}, which usually means a tiny crack or broken edge segment was created by delete/remesh.`;
  }

  if (topology.duplicateEdgeCount > 0) {
    return `Selected contour contains ${topology.duplicateEdgeCount} duplicate edge${topology.duplicateEdgeCount === 1 ? '' : 's'}. This usually means overlapping faces or a doubled crack near the hole.`;
  }

  if (topology.nearVertexPair) {
    return `Two boundary vertices are only ${topology.nearVertexPair.distance.toFixed(4)} mm apart but are not welded. Run/redo weld or remesh around the hole, then try Fill Hole again.`;
  }

  if (loop.edgeCount > loop.boundaryEdgeCount) {
    return 'Selected contour is a mixed loop group: it visually highlights together, but internally it combines open edges with non-boundary edges. Fill Hole needs one clean open rim.';
  }

  return 'Selected contour is not one clean closed boundary ring. It may be a combined loop group, a tiny crack, duplicate edge, or a remesh/delete artifact. Try welding/remeshing near the hole or selecting a simpler open loop.';
}

interface ResolvedBoundaryLoop {
  orderedVertexIds: Uint32Array;
  loop: HoleLoop;
  secondaryLoops: BoundaryCycleCandidate[];
  autoMessage: string | null;
}

function resolveUsableBoundaryLoop(
  loop: HoleLoop | undefined,
  positions: Float32Array,
): ResolvedBoundaryLoop | null {
  if (!loop) {
    return null;
  }

  if (loop.isBoundaryLoop && loop.orderedVertexIds && loop.orderedVertexIds.length >= 3) {
    return {
      orderedVertexIds: loop.orderedVertexIds,
      loop,
      secondaryLoops: [],
      autoMessage: null,
    };
  }

  const cycles = findSimpleClosedBoundaryCycles(loop, positions);
  if (cycles.length === 0) {
    return null;
  }

  const [selectedCycle] = cycles;
  const secondaryLoops = cycles.slice(1);
  return {
    orderedVertexIds: selectedCycle.orderedVertexIds,
    loop: createHoleLoopFromOrderedVertices(selectedCycle.orderedVertexIds),
    secondaryLoops,
    autoMessage:
      cycles.length > 1
        ? `Auto-selected the longest clean closed open-edge loop (${selectedCycle.orderedVertexIds.length} edges) from ${cycles.length} touching loops.`
        : `Auto-selected a clean closed open-edge loop (${selectedCycle.orderedVertexIds.length} edges) from the highlighted contour.`,
  };
}

interface BoundaryCycleCandidate {
  orderedVertexIds: Uint32Array;
  perimeter: number;
}

function findSimpleClosedBoundaryCycles(loop: HoleLoop, positions: Float32Array): BoundaryCycleCandidate[] {
  const graph = new Map<number, number[]>();
  const pairs = loop.segmentVertexPairs;
  for (let i = 0; i < pairs.length; i += 2) {
    addGraphNeighbor(graph, pairs[i], pairs[i + 1]);
    addGraphNeighbor(graph, pairs[i + 1], pairs[i]);
  }

  for (const neighbors of graph.values()) {
    neighbors.sort((a, b) => a - b);
  }

  const vertices = Array.from(graph.keys()).sort((a, b) => a - b);
  const cycles = new Map<string, Uint32Array>();
  const maxCycles = 128;
  const maxDepth = Math.min(Math.max(loop.edgeCount + 1, 3), 4000);

  for (const start of vertices) {
    const startNeighbors = graph.get(start) ?? [];
    for (const next of startNeighbors) {
      if (next < start) {
        continue;
      }
      traceBoundaryCycleDfs(graph, start, next, [start, next], new Set([createUndirectedEdgeKey(start, next)]), cycles, maxCycles, maxDepth);
      if (cycles.size >= maxCycles) {
        break;
      }
    }
    if (cycles.size >= maxCycles) {
      break;
    }
  }

  return Array.from(cycles.values())
    .map((orderedVertexIds) => ({
      orderedVertexIds,
      perimeter: computeCyclePerimeter(orderedVertexIds, positions),
    }))
    .filter((cycle) => cycle.orderedVertexIds.length >= 3 && cycle.perimeter > 0)
    .sort((a, b) => b.perimeter - a.perimeter);
}

function traceBoundaryCycleDfs(
  graph: Map<number, number[]>,
  start: number,
  current: number,
  path: number[],
  usedEdges: Set<string>,
  cycles: Map<string, Uint32Array>,
  maxCycles: number,
  maxDepth: number,
): void {
  if (cycles.size >= maxCycles || path.length > maxDepth) {
    return;
  }

  const neighbors = graph.get(current) ?? [];
  for (const next of neighbors) {
    const edgeKey = createUndirectedEdgeKey(current, next);
    if (usedEdges.has(edgeKey)) {
      continue;
    }

    if (next === start) {
      if (path.length >= 3) {
        const cycle = new Uint32Array(path);
        cycles.set(createCanonicalCycleKey(cycle), cycle);
      }
      continue;
    }

    if (next < start || path.includes(next)) {
      continue;
    }

    const nextUsedEdges = new Set(usedEdges);
    nextUsedEdges.add(edgeKey);
    traceBoundaryCycleDfs(
      graph,
      start,
      next,
      [...path, next],
      nextUsedEdges,
      cycles,
      maxCycles,
      maxDepth,
    );
  }
}

function addGraphNeighbor(graph: Map<number, number[]>, from: number, to: number): void {
  const neighbors = graph.get(from);
  if (neighbors) {
    if (!neighbors.includes(to)) {
      neighbors.push(to);
    }
    return;
  }

  graph.set(from, [to]);
}

function createHoleLoopFromOrderedVertices(orderedVertexIds: Uint32Array): HoleLoop {
  const pairs = new Uint32Array(orderedVertexIds.length * 2);
  for (let i = 0; i < orderedVertexIds.length; i += 1) {
    pairs[i * 2] = orderedVertexIds[i];
    pairs[i * 2 + 1] = orderedVertexIds[(i + 1) % orderedVertexIds.length];
  }

  return {
    segmentVertexPairs: pairs,
    edgeCount: orderedVertexIds.length,
    boundaryEdgeCount: orderedVertexIds.length,
    orderedVertexIds,
    isBoundaryLoop: true,
  };
}

function computeCyclePerimeter(orderedVertexIds: Uint32Array, positions: Float32Array): number {
  let perimeter = 0;
  for (let i = 0; i < orderedVertexIds.length; i += 1) {
    const a = orderedVertexIds[i] * 3;
    const b = orderedVertexIds[(i + 1) % orderedVertexIds.length] * 3;
    const dx = positions[a] - positions[b];
    const dy = positions[a + 1] - positions[b + 1];
    const dz = positions[a + 2] - positions[b + 2];
    perimeter += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return perimeter;
}

function createCanonicalCycleKey(cycle: Uint32Array): string {
  let minIndex = 0;
  for (let i = 1; i < cycle.length; i += 1) {
    if (cycle[i] < cycle[minIndex]) {
      minIndex = i;
    }
  }

  const forward: number[] = [];
  const backward: number[] = [];
  for (let i = 0; i < cycle.length; i += 1) {
    forward.push(cycle[(minIndex + i) % cycle.length]);
    backward.push(cycle[(minIndex - i + cycle.length) % cycle.length]);
  }

  const forwardKey = forward.join(':');
  const backwardKey = backward.join(':');
  return forwardKey < backwardKey ? forwardKey : backwardKey;
}

function createUndirectedEdgeKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function describeHoleFillFailure(
  fallbackMessage: string,
  reason: string | null,
  loop: HoleLoop,
  positions: Float32Array,
): string {
  switch (reason) {
    case 'loop_too_short':
      return `Boundary loop is too short: it has ${loop.orderedVertexIds?.length ?? loop.boundaryEdgeCount} vertices, but Fill Hole needs at least 3.`;
    case 'duplicate_vertices':
      return 'Boundary loop contains duplicated neighboring vertices. This usually means overlapping or unwelded vertices on the hole rim.';
    case 'non_boundary_edge':
    case 'non_manifold_boundary':
      return diagnoseUnfillableHoleLoop(loop, positions);
    case 'non_simple_projection':
      return 'Boundary loop crosses over itself when projected to the fill plane. This can happen when two holes touch, the rim folds over, or the contour is not one simple ring.';
    case 'insufficient_support':
      return 'Not enough surrounding surface was found to estimate a stable patch. The hole may be too close to a trimmed edge or has a very thin support band.';
    case 'hole_too_large':
      return 'The selected hole is larger than the safe fill limit for this tool.';
    case 'sharp_feature':
      return 'The hole rim is too sharp or irregular for a safe tangent fill.';
    case 'triangulation_failed':
      return 'The hole boundary could not be triangulated into clean faces. This often means the contour is concave, self-crossing, or has very uneven edge spacing.';
    case 'surface_fit_failed':
      return 'The local surface around the hole was unstable, so the app could not estimate a clean patch surface.';
    case 'triangle_quality':
      return 'The proposed fill would create very skinny or badly angled triangles at the seam.';
    case 'fairing_unstable':
      return fallbackMessage || 'The filled patch became unstable during smoothing/fairing.';
    default:
      return fallbackMessage || diagnoseUnfillableHoleLoop(loop, positions);
  }
}

interface HoleLoopTopology {
  endpointCount: number;
  duplicateEdgeCount: number;
  nonBoundaryEdgeCount: number;
  highDegreeVertices: Array<{ vertex: number; degree: number }>;
  nearVertexPair: { a: number; b: number; distance: number } | null;
}

function analyzeHoleLoopTopology(loop: HoleLoop, positions: Float32Array): HoleLoopTopology {
  const degreeByVertex = new Map<number, number>();
  const edgeCounts = new Map<string, number>();
  const loopVertices = new Set<number>();
  const pairs = loop.segmentVertexPairs;

  for (let i = 0; i < pairs.length; i += 2) {
    const a = pairs[i];
    const b = pairs[i + 1];
    loopVertices.add(a);
    loopVertices.add(b);
    degreeByVertex.set(a, (degreeByVertex.get(a) ?? 0) + 1);
    degreeByVertex.set(b, (degreeByVertex.get(b) ?? 0) + 1);
    const low = Math.min(a, b);
    const high = Math.max(a, b);
    const key = `${low}:${high}`;
    edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
  }

  let endpointCount = 0;
  const highDegreeVertices: Array<{ vertex: number; degree: number }> = [];
  for (const [vertex, degree] of degreeByVertex) {
    if (degree === 1) {
      endpointCount += 1;
    } else if (degree > 2) {
      highDegreeVertices.push({ vertex, degree });
    }
  }

  let duplicateEdgeCount = 0;
  for (const count of edgeCounts.values()) {
    if (count > 1) {
      duplicateEdgeCount += count - 1;
    }
  }

  return {
    endpointCount,
    duplicateEdgeCount,
    nonBoundaryEdgeCount: Math.max(0, loop.edgeCount - loop.boundaryEdgeCount),
    highDegreeVertices,
    nearVertexPair: findNearUnweldedLoopVertices(Array.from(loopVertices), positions),
  };
}

function findNearUnweldedLoopVertices(
  vertices: number[],
  positions: Float32Array,
): { a: number; b: number; distance: number } | null {
  const cellSize = HOLE_DIAGNOSTIC_NEAR_WELD_MM;
  const maxDistanceSq = HOLE_DIAGNOSTIC_NEAR_WELD_MM * HOLE_DIAGNOSTIC_NEAR_WELD_MM;
  const cells = new Map<string, number[]>();
  let best: { a: number; b: number; distance: number } | null = null;

  for (const vertex of vertices) {
    const offset = vertex * 3;
    const x = positions[offset];
    const y = positions[offset + 1];
    const z = positions[offset + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      continue;
    }

    const cx = Math.floor(x / cellSize);
    const cy = Math.floor(y / cellSize);
    const cz = Math.floor(z / cellSize);
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const candidates = cells.get(`${cx + dx}:${cy + dy}:${cz + dz}`);
          if (!candidates) {
            continue;
          }

          for (const otherVertex of candidates) {
            if (otherVertex === vertex) {
              continue;
            }

            const otherOffset = otherVertex * 3;
            const ddx = x - positions[otherOffset];
            const ddy = y - positions[otherOffset + 1];
            const ddz = z - positions[otherOffset + 2];
            const distanceSq = ddx * ddx + ddy * ddy + ddz * ddz;
            if (distanceSq <= 1e-12 || distanceSq > maxDistanceSq) {
              continue;
            }

            const distance = Math.sqrt(distanceSq);
            if (!best || distance < best.distance) {
              best = { a: vertex, b: otherVertex, distance };
            }
          }
        }
      }
    }

    const key = `${cx}:${cy}:${cz}`;
    const cell = cells.get(key);
    if (cell) {
      cell.push(vertex);
    } else {
      cells.set(key, [vertex]);
    }
  }

  return best;
}

interface MaterializedMeshData {
  positions: Float32Array;
  indices: Uint32Array;
  faceMaterialIndices: Uint8Array;
}

interface FinalizedPositiveLimbGeometry {
  geometry: BufferGeometry;
  faceMaterialIndices: Uint8Array;
  topology: MeshTopologySummary;
}

interface MaterializedEdgeUse {
  triangle: number;
  a: number;
  b: number;
}

interface MeshTopologySummary {
  boundaryEdges: number;
  nonManifoldEdges: number;
  inconsistentWindingEdges: number;
}

function finalizePositiveLimbGeometry(
  source: BufferGeometry,
  sourceFaceMaterialIndices: Uint8Array,
): FinalizedPositiveLimbGeometry {
  let mesh = readMaterializedMesh(source, sourceFaceMaterialIndices);
  source.dispose();
  mesh = weldMaterializedMeshByProximity(mesh, POSITIVE_FINAL_WELD_TOLERANCE_MM);
  mesh = keepLargestMaterializedComponent(mesh);

  for (let pass = 0; pass < POSITIVE_FINAL_REPAIR_PASSES; pass += 1) {
    mesh = removeExcessNonManifoldFaces(mesh);
    mesh = weldMaterializedMeshByProximity(mesh, POSITIVE_FINAL_WELD_TOLERANCE_MM);
    makeTriangleWindingCoherent(mesh.indices);

    const beforeFillTriangleCount = mesh.indices.length / 3;
    const geometryForFill = createMaterializedGeometry(mesh);
    const filledGeometry = fillAllGeometryHoles(geometryForFill);
    const filledTriangleCount = Math.floor((filledGeometry.getIndex()?.count ?? 0) / 3);
    const filledMaterials = new Uint8Array(filledTriangleCount);
    filledMaterials.set(
      mesh.faceMaterialIndices.subarray(0, Math.min(beforeFillTriangleCount, filledTriangleCount)),
    );
    if (filledTriangleCount > beforeFillTriangleCount) {
      filledMaterials.fill(1, beforeFillTriangleCount);
    }

    mesh = readMaterializedMesh(filledGeometry, filledMaterials);
    filledGeometry.dispose();
    mesh = weldMaterializedMeshByProximity(mesh, POSITIVE_FINAL_WELD_TOLERANCE_MM);
    if (analyzeMaterializedTopology(mesh).boundaryEdges > 0) {
      mesh = capSimpleMaterializedBoundaryLoops(mesh);
      mesh = weldMaterializedMeshByProximity(mesh, POSITIVE_FINAL_WELD_TOLERANCE_MM);
    }
    mesh = keepLargestMaterializedComponent(mesh);

    const topology = analyzeMaterializedTopology(mesh);
    if (topology.boundaryEdges === 0 && topology.nonManifoldEdges === 0) {
      break;
    }
  }

  const geometry = createMaterializedGeometry(mesh);
  orientGeometryOutward(geometry);
  mesh = readMaterializedMesh(geometry, mesh.faceMaterialIndices);
  const topology = analyzeMaterializedTopology(mesh);

  return {
    geometry,
    faceMaterialIndices: mesh.faceMaterialIndices,
    topology,
  };
}

function readMaterializedMesh(
  geometry: BufferGeometry,
  faceMaterialIndices: Uint8Array | null,
): MaterializedMeshData {
  const positionAttribute = geometry.getAttribute('position');
  const indexAttribute = geometry.getIndex();
  if (!positionAttribute || !indexAttribute) {
    throw new Error('Positive Limb finalization requires indexed triangle geometry.');
  }

  const positions = new Float32Array(positionAttribute.count * 3);
  for (let vertex = 0; vertex < positionAttribute.count; vertex += 1) {
    const offset = vertex * 3;
    positions[offset] = positionAttribute.getX(vertex);
    positions[offset + 1] = positionAttribute.getY(vertex);
    positions[offset + 2] = positionAttribute.getZ(vertex);
  }

  const indices = new Uint32Array(indexAttribute.count);
  for (let index = 0; index < indexAttribute.count; index += 1) {
    indices[index] = indexAttribute.getX(index);
  }

  const triangleCount = Math.floor(indices.length / 3);
  const materials = new Uint8Array(triangleCount);
  if (faceMaterialIndices) {
    materials.set(faceMaterialIndices.subarray(0, Math.min(faceMaterialIndices.length, triangleCount)));
  }

  return { positions, indices, faceMaterialIndices: materials };
}

function createMaterializedGeometry(mesh: MaterializedMeshData): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(mesh.positions.slice(), 3));
  geometry.setIndex(new BufferAttribute(mesh.indices.slice(), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function weldMaterializedMeshByProximity(
  source: MaterializedMeshData,
  tolerance: number,
): MaterializedMeshData {
  const sourceVertexCount = source.positions.length / 3;
  const sourceToWelded = new Int32Array(sourceVertexCount);
  sourceToWelded.fill(-1);
  const weldedPositions: number[] = [];
  const cells = new Map<string, number[]>();
  const safeTolerance = Math.max(tolerance, 1e-9);
  const toleranceSq = safeTolerance * safeTolerance;

  for (let vertex = 0; vertex < sourceVertexCount; vertex += 1) {
    const offset = vertex * 3;
    const x = source.positions[offset];
    const y = source.positions[offset + 1];
    const z = source.positions[offset + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      continue;
    }

    const cx = Math.floor(x / safeTolerance);
    const cy = Math.floor(y / safeTolerance);
    const cz = Math.floor(z / safeTolerance);
    let closestVertex = -1;
    let closestDistanceSq = toleranceSq;
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const candidates = cells.get(`${cx + dx}:${cy + dy}:${cz + dz}`);
          if (!candidates) {
            continue;
          }

          for (let i = 0; i < candidates.length; i += 1) {
            const candidate = candidates[i];
            const candidateOffset = candidate * 3;
            const ddx = x - weldedPositions[candidateOffset];
            const ddy = y - weldedPositions[candidateOffset + 1];
            const ddz = z - weldedPositions[candidateOffset + 2];
            const distanceSq = ddx * ddx + ddy * ddy + ddz * ddz;
            if (distanceSq <= closestDistanceSq) {
              closestDistanceSq = distanceSq;
              closestVertex = candidate;
            }
          }
        }
      }
    }

    if (closestVertex === -1) {
      closestVertex = weldedPositions.length / 3;
      weldedPositions.push(x, y, z);
      const cellKey = `${cx}:${cy}:${cz}`;
      const cell = cells.get(cellKey);
      if (cell) {
        cell.push(closestVertex);
      } else {
        cells.set(cellKey, [closestVertex]);
      }
    }
    sourceToWelded[vertex] = closestVertex;
  }

  const indices: number[] = [];
  const materials: number[] = [];
  const faceByVertexKey = new Map<string, number>();
  const triangleCount = Math.floor(source.indices.length / 3);
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3;
    const sourceA = source.indices[offset];
    const sourceB = source.indices[offset + 1];
    const sourceC = source.indices[offset + 2];
    if (sourceA >= sourceVertexCount || sourceB >= sourceVertexCount || sourceC >= sourceVertexCount) {
      continue;
    }

    const a = sourceToWelded[sourceA];
    const b = sourceToWelded[sourceB];
    const c = sourceToWelded[sourceC];
    if (a < 0 || b < 0 || c < 0 || a === b || b === c || c === a) {
      continue;
    }
    if (triangleAreaSquared(weldedPositions, a, b, c) <= 1e-16) {
      continue;
    }

    const sorted = [a, b, c].sort((left, right) => left - right);
    const faceKey = `${sorted[0]}:${sorted[1]}:${sorted[2]}`;
    const duplicate = faceByVertexKey.get(faceKey);
    const material = source.faceMaterialIndices[triangle] ?? 0;
    if (duplicate !== undefined) {
      materials[duplicate] = Math.max(materials[duplicate], material);
      continue;
    }

    faceByVertexKey.set(faceKey, materials.length);
    indices.push(a, b, c);
    materials.push(material);
  }

  return compactMaterializedMesh({
    positions: new Float32Array(weldedPositions),
    indices: new Uint32Array(indices),
    faceMaterialIndices: new Uint8Array(materials),
  });
}

function triangleAreaSquared(
  positions: ArrayLike<number>,
  a: number,
  b: number,
  c: number,
): number {
  const ao = a * 3;
  const bo = b * 3;
  const co = c * 3;
  const abx = positions[bo] - positions[ao];
  const aby = positions[bo + 1] - positions[ao + 1];
  const abz = positions[bo + 2] - positions[ao + 2];
  const acx = positions[co] - positions[ao];
  const acy = positions[co + 1] - positions[ao + 1];
  const acz = positions[co + 2] - positions[ao + 2];
  const nx = aby * acz - abz * acy;
  const ny = abz * acx - abx * acz;
  const nz = abx * acy - aby * acx;
  return nx * nx + ny * ny + nz * nz;
}

function compactMaterializedMesh(source: MaterializedMeshData): MaterializedMeshData {
  const vertexMap = new Int32Array(source.positions.length / 3);
  vertexMap.fill(-1);
  const positions: number[] = [];
  const indices = new Uint32Array(source.indices.length);

  for (let index = 0; index < source.indices.length; index += 1) {
    const sourceVertex = source.indices[index];
    let targetVertex = vertexMap[sourceVertex];
    if (targetVertex === -1) {
      targetVertex = positions.length / 3;
      vertexMap[sourceVertex] = targetVertex;
      const offset = sourceVertex * 3;
      positions.push(source.positions[offset], source.positions[offset + 1], source.positions[offset + 2]);
    }
    indices[index] = targetVertex;
  }

  return {
    positions: new Float32Array(positions),
    indices,
    faceMaterialIndices: source.faceMaterialIndices.slice(),
  };
}

function keepLargestMaterializedComponent(source: MaterializedMeshData): MaterializedMeshData {
  const triangleCount = source.indices.length / 3;
  if (triangleCount <= 1) {
    return source;
  }

  const trianglesByVertex: number[][] = Array.from(
    { length: source.positions.length / 3 },
    () => [],
  );
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3;
    trianglesByVertex[source.indices[offset]].push(triangle);
    trianglesByVertex[source.indices[offset + 1]].push(triangle);
    trianglesByVertex[source.indices[offset + 2]].push(triangle);
  }

  const visited = new Uint8Array(triangleCount);
  let largest: number[] = [];
  for (let start = 0; start < triangleCount; start += 1) {
    if (visited[start] !== 0) {
      continue;
    }

    const component: number[] = [];
    const stack = [start];
    visited[start] = 1;
    while (stack.length > 0) {
      const triangle = stack.pop() as number;
      component.push(triangle);
      const offset = triangle * 3;
      for (let corner = 0; corner < 3; corner += 1) {
        const linked = trianglesByVertex[source.indices[offset + corner]];
        for (let i = 0; i < linked.length; i += 1) {
          const neighbor = linked[i];
          if (visited[neighbor] === 0) {
            visited[neighbor] = 1;
            stack.push(neighbor);
          }
        }
      }
    }

    if (component.length > largest.length) {
      largest = component;
    }
  }

  if (largest.length === triangleCount) {
    return source;
  }

  const keep = new Uint8Array(triangleCount);
  for (let i = 0; i < largest.length; i += 1) {
    keep[largest[i]] = 1;
  }
  return filterMaterializedTriangles(source, keep);
}

function removeExcessNonManifoldFaces(source: MaterializedMeshData): MaterializedMeshData {
  const edgeUses = buildMaterializedEdgeUses(source.indices);
  const triangleCount = source.indices.length / 3;
  const keep = new Uint8Array(triangleCount);
  keep.fill(1);
  const normals = computeMaterializedFaceNormals(source);
  let removed = false;

  for (const uses of edgeUses.values()) {
    if (uses.length <= 2) {
      continue;
    }

    let bestLeft = 0;
    let bestRight = 1;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let left = 0; left < uses.length; left += 1) {
      for (let right = left + 1; right < uses.length; right += 1) {
        const leftUse = uses[left];
        const rightUse = uses[right];
        const leftOffset = leftUse.triangle * 3;
        const rightOffset = rightUse.triangle * 3;
        const normalAlignment = Math.abs(
          normals[leftOffset] * normals[rightOffset] +
            normals[leftOffset + 1] * normals[rightOffset + 1] +
            normals[leftOffset + 2] * normals[rightOffset + 2],
        );
        const oppositeDirection = leftUse.a === rightUse.b && leftUse.b === rightUse.a;
        const sameMaterial =
          source.faceMaterialIndices[leftUse.triangle] === source.faceMaterialIndices[rightUse.triangle];
        const score = normalAlignment + (oppositeDirection ? 0.25 : 0) + (sameMaterial ? 0.02 : 0);
        if (score > bestScore) {
          bestScore = score;
          bestLeft = left;
          bestRight = right;
        }
      }
    }

    for (let i = 0; i < uses.length; i += 1) {
      if (i !== bestLeft && i !== bestRight) {
        keep[uses[i].triangle] = 0;
        removed = true;
      }
    }
  }

  return removed ? filterMaterializedTriangles(source, keep) : source;
}

function computeMaterializedFaceNormals(source: MaterializedMeshData): Float32Array {
  const triangleCount = source.indices.length / 3;
  const normals = new Float32Array(triangleCount * 3);
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const indexOffset = triangle * 3;
    const a = source.indices[indexOffset] * 3;
    const b = source.indices[indexOffset + 1] * 3;
    const c = source.indices[indexOffset + 2] * 3;
    const abx = source.positions[b] - source.positions[a];
    const aby = source.positions[b + 1] - source.positions[a + 1];
    const abz = source.positions[b + 2] - source.positions[a + 2];
    const acx = source.positions[c] - source.positions[a];
    const acy = source.positions[c + 1] - source.positions[a + 1];
    const acz = source.positions[c + 2] - source.positions[a + 2];
    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;
    const length = Math.hypot(nx, ny, nz);
    if (length > 1e-12) {
      nx /= length;
      ny /= length;
      nz /= length;
    }
    normals[indexOffset] = nx;
    normals[indexOffset + 1] = ny;
    normals[indexOffset + 2] = nz;
  }
  return normals;
}

function filterMaterializedTriangles(
  source: MaterializedMeshData,
  keep: Uint8Array,
): MaterializedMeshData {
  const indices: number[] = [];
  const materials: number[] = [];
  const triangleCount = source.indices.length / 3;
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    if (keep[triangle] === 0) {
      continue;
    }
    const offset = triangle * 3;
    indices.push(source.indices[offset], source.indices[offset + 1], source.indices[offset + 2]);
    materials.push(source.faceMaterialIndices[triangle] ?? 0);
  }

  return compactMaterializedMesh({
    positions: source.positions,
    indices: new Uint32Array(indices),
    faceMaterialIndices: new Uint8Array(materials),
  });
}

function buildMaterializedEdgeUses(indices: Uint32Array): Map<string, MaterializedEdgeUse[]> {
  const edgeUses = new Map<string, MaterializedEdgeUse[]>();
  for (let triangle = 0; triangle < indices.length / 3; triangle += 1) {
    const offset = triangle * 3;
    addMaterializedEdgeUse(edgeUses, triangle, indices[offset], indices[offset + 1]);
    addMaterializedEdgeUse(edgeUses, triangle, indices[offset + 1], indices[offset + 2]);
    addMaterializedEdgeUse(edgeUses, triangle, indices[offset + 2], indices[offset]);
  }
  return edgeUses;
}

function addMaterializedEdgeUse(
  edgeUses: Map<string, MaterializedEdgeUse[]>,
  triangle: number,
  a: number,
  b: number,
): void {
  const key = a < b ? `${a}:${b}` : `${b}:${a}`;
  const uses = edgeUses.get(key);
  const use = { triangle, a, b };
  if (uses) {
    uses.push(use);
  } else {
    edgeUses.set(key, [use]);
  }
}

function analyzeMaterializedTopology(source: MaterializedMeshData): MeshTopologySummary {
  const edgeUses = buildMaterializedEdgeUses(source.indices);
  let boundaryEdges = 0;
  let nonManifoldEdges = 0;
  let inconsistentWindingEdges = 0;
  for (const uses of edgeUses.values()) {
    if (uses.length === 1) {
      boundaryEdges += 1;
    } else if (uses.length > 2) {
      nonManifoldEdges += 1;
    } else if (uses.length === 2 && uses[0].a === uses[1].a && uses[0].b === uses[1].b) {
      inconsistentWindingEdges += 1;
    }
  }
  return { boundaryEdges, nonManifoldEdges, inconsistentWindingEdges };
}

function capSimpleMaterializedBoundaryLoops(source: MaterializedMeshData): MaterializedMeshData {
  const edgeUses = buildMaterializedEdgeUses(source.indices);
  const boundaryUseByKey = new Map<string, MaterializedEdgeUse>();
  const boundaryNeighbors = new Map<number, number[]>();
  for (const [key, uses] of edgeUses) {
    if (uses.length !== 1) {
      continue;
    }
    const use = uses[0];
    boundaryUseByKey.set(key, use);
    appendBoundaryNeighbor(boundaryNeighbors, use.a, use.b);
    appendBoundaryNeighbor(boundaryNeighbors, use.b, use.a);
  }

  if (boundaryUseByKey.size === 0) {
    return source;
  }

  const processedVertices = new Set<number>();
  const positions = Array.from(source.positions);
  const appendedIndices: number[] = [];
  for (const startVertex of boundaryNeighbors.keys()) {
    if (processedVertices.has(startVertex)) {
      continue;
    }

    const componentVertices: number[] = [];
    const componentStack = [startVertex];
    processedVertices.add(startVertex);
    while (componentStack.length > 0) {
      const vertex = componentStack.pop() as number;
      componentVertices.push(vertex);
      const neighbors = boundaryNeighbors.get(vertex) ?? [];
      for (let i = 0; i < neighbors.length; i += 1) {
        if (!processedVertices.has(neighbors[i])) {
          processedVertices.add(neighbors[i]);
          componentStack.push(neighbors[i]);
        }
      }
    }

    if (
      componentVertices.length < 3 ||
      componentVertices.some((vertex) => (boundaryNeighbors.get(vertex)?.length ?? 0) !== 2)
    ) {
      continue;
    }

    const loop = traceSimpleBoundaryLoop(startVertex, boundaryNeighbors, componentVertices.length);
    if (!loop || loop.length !== componentVertices.length) {
      continue;
    }

    let centerX = 0;
    let centerY = 0;
    let centerZ = 0;
    for (let i = 0; i < loop.length; i += 1) {
      const offset = loop[i] * 3;
      centerX += source.positions[offset];
      centerY += source.positions[offset + 1];
      centerZ += source.positions[offset + 2];
    }
    const inverseLoopLength = 1 / loop.length;
    centerX *= inverseLoopLength;
    centerY *= inverseLoopLength;
    centerZ *= inverseLoopLength;
    const centerVertex = positions.length / 3;

    const loopTriangles: number[] = [];
    positions.push(centerX, centerY, centerZ);
    for (let i = 0; i < loop.length; i += 1) {
      const next = (i + 1) % loop.length;
      if (triangleAreaSquared(positions, loop[i], loop[next], centerVertex) <= 1e-16) {
        loopTriangles.length = 0;
        positions.length -= 3;
        break;
      }
      loopTriangles.push(loop[i], loop[next], centerVertex);
    }
    if (loopTriangles.length < 3) {
      continue;
    }

    const firstEdgeKey = createUndirectedEdgeKey(loop[0], loop[1]);
    const existingUse = boundaryUseByKey.get(firstEdgeKey);
    if (existingUse && patchUsesDirectedEdge(loopTriangles, existingUse.a, existingUse.b)) {
      for (let offset = 0; offset < loopTriangles.length; offset += 3) {
        const swap = loopTriangles[offset + 1];
        loopTriangles[offset + 1] = loopTriangles[offset + 2];
        loopTriangles[offset + 2] = swap;
      }
    }
    appendedIndices.push(...loopTriangles);
  }

  if (appendedIndices.length === 0) {
    return source;
  }

  const indices = new Uint32Array(source.indices.length + appendedIndices.length);
  indices.set(source.indices);
  indices.set(appendedIndices, source.indices.length);
  const faceMaterialIndices = new Uint8Array(indices.length / 3);
  faceMaterialIndices.set(source.faceMaterialIndices);
  faceMaterialIndices.fill(1, source.faceMaterialIndices.length);
  return {
    positions: new Float32Array(positions),
    indices,
    faceMaterialIndices,
  };
}

function appendBoundaryNeighbor(neighbors: Map<number, number[]>, vertex: number, neighbor: number): void {
  const linked = neighbors.get(vertex);
  if (linked) {
    linked.push(neighbor);
  } else {
    neighbors.set(vertex, [neighbor]);
  }
}

function traceSimpleBoundaryLoop(
  startVertex: number,
  neighbors: Map<number, number[]>,
  expectedLength: number,
): number[] | null {
  const loop = [startVertex];
  let previous = -1;
  let current = startVertex;
  for (let step = 0; step < expectedLength; step += 1) {
    const linked = neighbors.get(current);
    if (!linked || linked.length !== 2) {
      return null;
    }
    const next = linked[0] === previous ? linked[1] : linked[0];
    previous = current;
    current = next;
    if (current === startVertex) {
      return loop.length === expectedLength ? loop : null;
    }
    if (loop.includes(current)) {
      return null;
    }
    loop.push(current);
  }
  return null;
}

function patchUsesDirectedEdge(indices: number[], a: number, b: number): boolean {
  for (let offset = 0; offset < indices.length; offset += 3) {
    const triangle = [indices[offset], indices[offset + 1], indices[offset + 2]];
    for (let corner = 0; corner < 3; corner += 1) {
      if (triangle[corner] === a && triangle[(corner + 1) % 3] === b) {
        return true;
      }
    }
  }
  return false;
}

function fillAllGeometryHoles(geometry: BufferGeometry): BufferGeometry {
  const positionAttribute = geometry.getAttribute('position');
  const indexAttribute = geometry.getIndex();
  if (!positionAttribute || !indexAttribute) {
    return geometry;
  }

  const sourcePositions = positionAttribute.array as ArrayLike<number>;
  const sourceIndices = indexAttribute.array as ArrayLike<number>;
  const positions = new Float32Array(sourcePositions.length);
  const indices = new Uint32Array(sourceIndices.length);
  for (let i = 0; i < sourcePositions.length; i += 1) {
    positions[i] = sourcePositions[i];
  }
  for (let i = 0; i < sourceIndices.length; i += 1) {
    indices[i] = sourceIndices[i];
  }

  const fillMesh = createHoleFillMesh(positions, indices, positions.slice());
  const failedLoopKeys = new Set<string>();
  let filledCount = 0;

  for (let attempt = 0; attempt < 100; attempt += 1) {
    const loops = buildOpenBoundaryLoopCandidates(
      new Uint32Array(fillMesh.indices),
      new Float32Array(fillMesh.positions),
    )
      .filter((loop) => loop.isBoundaryLoop && loop.orderedVertexIds && loop.orderedVertexIds.length >= 3);
    if (loops.length === 0) {
      break;
    }

    let filledThisPass = false;
    for (let i = 0; i < loops.length; i += 1) {
      const loop = loops[i];
      if (!loop.orderedVertexIds) {
        continue;
      }

      const loopKey = createHoleLoopKey(loop.orderedVertexIds);
      if (failedLoopKeys.has(loopKey)) {
        continue;
      }

      const result = executeHoleFill(fillMesh, Array.from(loop.orderedVertexIds), {
        ignoreSharpFeatureValidation: true,
      });
      if (!result.success) {
        failedLoopKeys.add(loopKey);
        continue;
      }

      filledCount += 1;
      filledThisPass = true;
      break;
    }

    if (!filledThisPass) {
      break;
    }
  }

  if (filledCount === 0) {
    return geometry;
  }

  const filledGeometry = createGeometryFromMeshArrays(fillMesh.positions, fillMesh.indices);
  geometry.dispose();
  return weldGeometryByDistance(filledGeometry);
}

function createHoleLoopKey(vertexIds: ArrayLike<number>): string {
  const sorted = Array.from(vertexIds).sort((a, b) => a - b);
  return sorted.join(',');
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
  uvs: Float32Array | null = null,
  faceMaterialIndices: Uint8Array | null = null,
  colors: Float32Array | null = null,
): {
  geometry: BufferGeometry | null;
  referencePositions: Float32Array | null;
  faceMaterialIndices: Uint8Array | null;
  colors: Float32Array | null;
} {
  const vertexMap = new Int32Array(positions.length / 3);
  vertexMap.fill(-1);

  const nextPositions: number[] = [];
  const nextReferencePositions: number[] = [];
  const nextUvs: number[] = [];
  const nextColors: number[] = [];
  const nextIndices: number[] = [];
  const nextFaceMaterialIndices: number[] = [];
  const shouldCopyUvs = uvs !== null && uvs.length >= vertexMap.length * 2;
  const shouldCopyColors = colors !== null && colors.length >= vertexMap.length * 3;
  const shouldCopyFaceMaterials =
    faceMaterialIndices !== null && faceMaterialIndices.length >= indices.length / 3;

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
        if (shouldCopyUvs) {
          const uvOffset = sourceVertex * 2;
          nextUvs.push(uvs[uvOffset], uvs[uvOffset + 1]);
        }
        if (shouldCopyColors) {
          const colorOffset = sourceVertex * 3;
          nextColors.push(colors[colorOffset], colors[colorOffset + 1], colors[colorOffset + 2]);
        }
      }

      nextIndices.push(targetVertex);
    }
    if (shouldCopyFaceMaterials) {
      nextFaceMaterialIndices.push(faceMaterialIndices[triangle]);
    }
  }

  if (nextIndices.length === 0) {
    return {
      geometry: null,
      referencePositions: null,
      faceMaterialIndices: null,
      colors: null,
    };
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(nextPositions), 3));
  if (shouldCopyUvs) {
    geometry.setAttribute('uv', new BufferAttribute(new Float32Array(nextUvs), 2));
  }
  geometry.setIndex(new BufferAttribute(new Uint32Array(nextIndices), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return {
    geometry,
    referencePositions: new Float32Array(nextReferencePositions),
    faceMaterialIndices:
      shouldCopyFaceMaterials && nextFaceMaterialIndices.some((materialIndex) => materialIndex !== 0)
        ? new Uint8Array(nextFaceMaterialIndices)
        : null,
    colors: shouldCopyColors ? new Float32Array(nextColors) : null,
  };
}

function createGeometryFromMeshArrays(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  uvs: ArrayLike<number> | null = null,
): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  if (uvs && uvs.length >= Math.floor(positions.length / 3) * 2) {
    geometry.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
  }
  geometry.setIndex(new BufferAttribute(new Uint32Array(indices), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function keepLargestConnectedComponentGeometry(source: BufferGeometry): BufferGeometry {
  const positionAttribute = source.getAttribute('position');
  const indexAttribute = source.getIndex();
  if (!positionAttribute || !indexAttribute || indexAttribute.count < 3) {
    return source;
  }

  const sourcePositions = positionAttribute.array as ArrayLike<number>;
  const sourceIndices = indexAttribute.array as ArrayLike<number>;
  const triangleCount = Math.floor(indexAttribute.count / 3);
  const vertexCount = positionAttribute.count;
  const trianglesByVertex: number[][] = Array.from({ length: vertexCount }, () => []);

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3;
    trianglesByVertex[sourceIndices[offset]].push(triangle);
    trianglesByVertex[sourceIndices[offset + 1]].push(triangle);
    trianglesByVertex[sourceIndices[offset + 2]].push(triangle);
  }

  const visited = new Uint8Array(triangleCount);
  const stack: number[] = [];
  let bestTriangles: number[] = [];

  for (let startTriangle = 0; startTriangle < triangleCount; startTriangle += 1) {
    if (visited[startTriangle] !== 0) {
      continue;
    }

    const component: number[] = [];
    stack.length = 0;
    stack.push(startTriangle);
    visited[startTriangle] = 1;

    while (stack.length > 0) {
      const triangle = stack.pop() as number;
      component.push(triangle);
      const offset = triangle * 3;

      for (let corner = 0; corner < 3; corner += 1) {
        const linkedTriangles = trianglesByVertex[sourceIndices[offset + corner]];
        for (let i = 0; i < linkedTriangles.length; i += 1) {
          const neighbor = linkedTriangles[i];
          if (visited[neighbor] !== 0) {
            continue;
          }

          visited[neighbor] = 1;
          stack.push(neighbor);
        }
      }
    }

    if (component.length > bestTriangles.length) {
      bestTriangles = component;
    }
  }

  if (bestTriangles.length === triangleCount) {
    return source;
  }

  const vertexMap = new Int32Array(vertexCount);
  vertexMap.fill(-1);
  const nextPositions: number[] = [];
  const nextIndices: number[] = [];

  for (let i = 0; i < bestTriangles.length; i += 1) {
    const triangleOffset = bestTriangles[i] * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const sourceVertex = sourceIndices[triangleOffset + corner];
      let targetVertex = vertexMap[sourceVertex];
      if (targetVertex === -1) {
        targetVertex = nextPositions.length / 3;
        vertexMap[sourceVertex] = targetVertex;
        const sourceOffset = sourceVertex * 3;
        nextPositions.push(
          sourcePositions[sourceOffset],
          sourcePositions[sourceOffset + 1],
          sourcePositions[sourceOffset + 2],
        );
      }

      nextIndices.push(targetVertex);
    }
  }

  return createGeometryFromMeshArrays(nextPositions, nextIndices);
}

function createGuideOrderedBoundaryComponent(
  positions: ArrayLike<number>,
  segmentVertexPairs: Uint32Array,
  guide: Float32Array,
): Uint32Array | null {
  if (segmentVertexPairs.length < 6 || guide.length < 9) {
    return null;
  }

  const uniqueVertices = new Set<number>();
  for (let i = 0; i < segmentVertexPairs.length; i += 1) {
    uniqueVertices.add(segmentVertexPairs[i]);
  }

  const scoredVertices = Array.from(uniqueVertices)
    .map((vertex) => ({
      vertex,
      score: scoreVertexAlongGuide(positions, vertex, guide),
    }))
    .filter((entry) => Number.isFinite(entry.score.distanceSq))
    .sort((left, right) => left.score.along - right.score.along);

  if (scoredVertices.length < 3) {
    return null;
  }

  const distances = scoredVertices.map((entry) => entry.score.distanceSq).sort((left, right) => left - right);
  const medianDistance = distances[Math.floor(distances.length / 2)] ?? 0;
  const keepDistanceSq = Math.max(medianDistance * 9, 16);
  const filtered = scoredVertices.filter((entry) => entry.score.distanceSq <= keepDistanceSq);
  const ordered = (filtered.length >= 3 ? filtered : scoredVertices).map((entry) => entry.vertex);

  return ordered.length >= 3 ? new Uint32Array(ordered) : null;
}

function scoreVertexAlongGuide(
  positions: ArrayLike<number>,
  vertex: number,
  guide: Float32Array,
): { along: number; distanceSq: number } {
  const offset = vertex * 3;
  let bestAlong = 0;
  let bestDistanceSq = Infinity;
  const segmentCount = guide.length / 3;

  for (let i = 0; i < segmentCount; i += 1) {
    const next = (i + 1) % segmentCount;
    const guideOffset = i * 3;
    const nextOffset = next * 3;
    const ax = guide[guideOffset];
    const ay = guide[guideOffset + 1];
    const az = guide[guideOffset + 2];
    const bx = guide[nextOffset];
    const by = guide[nextOffset + 1];
    const bz = guide[nextOffset + 2];
    const abx = bx - ax;
    const aby = by - ay;
    const abz = bz - az;
    const apx = positions[offset] - ax;
    const apy = positions[offset + 1] - ay;
    const apz = positions[offset + 2] - az;
    const lengthSq = abx * abx + aby * aby + abz * abz;
    const t = lengthSq > 1e-12 ? Math.min(Math.max((apx * abx + apy * aby + apz * abz) / lengthSq, 0), 1) : 0;
    const closestX = ax + abx * t;
    const closestY = ay + aby * t;
    const closestZ = az + abz * t;
    const dx = positions[offset] - closestX;
    const dy = positions[offset + 1] - closestY;
    const dz = positions[offset + 2] - closestZ;
    const distanceSq = dx * dx + dy * dy + dz * dz;
    if (distanceSq < bestDistanceSq) {
      bestDistanceSq = distanceSq;
      bestAlong = i + t;
    }
  }

  return { along: bestAlong, distanceSq: bestDistanceSq };
}

function applyVertexColors(editable: EditableMeshData, colors: Float32Array): void {
  if (colors.length < editable.vertexCount * 3) {
    return;
  }

  editable.colors.set(colors.subarray(0, editable.vertexCount * 3));
  editable.colorAttribute.needsUpdate = true;
  editable.geometry.getAttribute('color').needsUpdate = true;
}

function sampleBakedColorFromSource(
  source: BakedColorSource,
  faceIndex: number,
  point: Vector3,
  target: Vector3,
): Vector3 {
  if (source.faceMaterialIndices?.[faceIndex] === 1) {
    return target.set(0.85, 0.87, 0.9);
  }

  const triOffset = faceIndex * 3;
  const a = source.indices[triOffset];
  const b = source.indices[triOffset + 1];
  const c = source.indices[triOffset + 2];
  if (a === undefined || b === undefined || c === undefined) {
    return target.set(0.85, 0.87, 0.9);
  }

  const triangleA = getArrayVertex(source.positions, a, new Vector3());
  const triangleB = getArrayVertex(source.positions, b, new Vector3());
  const triangleC = getArrayVertex(source.positions, c, new Vector3());
  const barycentric = Triangle.getBarycoord(point, triangleA, triangleB, triangleC, new Vector3());
  if (!barycentric) {
    return target.set(0.85, 0.87, 0.9);
  }

  if (source.textureSampler && source.uvs) {
    const uv = interpolateSourceUv(source.uvs, a, b, c, barycentric, new Vector2());
    return sampleTextureColor(source.textureSampler, uv.x, uv.y, target);
  }

  if (source.colors && source.colors.length >= Math.max(a, b, c) * 3 + 3) {
    return interpolateSourceColor(source.colors, a, b, c, barycentric, target);
  }

  return target.set(0.85, 0.87, 0.9);
}

function getArrayVertex(positions: Float32Array, vertex: number, target: Vector3): Vector3 {
  const offset = vertex * 3;
  return target.set(positions[offset], positions[offset + 1], positions[offset + 2]);
}

function interpolateSourceUv(
  uvs: Float32Array,
  a: number,
  b: number,
  c: number,
  barycentric: Vector3,
  target: Vector2,
): Vector2 {
  const ao = a * 2;
  const bo = b * 2;
  const co = c * 2;
  return target.set(
    uvs[ao] * barycentric.x + uvs[bo] * barycentric.y + uvs[co] * barycentric.z,
    uvs[ao + 1] * barycentric.x + uvs[bo + 1] * barycentric.y + uvs[co + 1] * barycentric.z,
  );
}

function interpolateSourceColor(
  colors: Float32Array,
  a: number,
  b: number,
  c: number,
  barycentric: Vector3,
  target: Vector3,
): Vector3 {
  const ao = a * 3;
  const bo = b * 3;
  const co = c * 3;
  return target.set(
    colors[ao] * barycentric.x + colors[bo] * barycentric.y + colors[co] * barycentric.z,
    colors[ao + 1] * barycentric.x + colors[bo + 1] * barycentric.y + colors[co + 1] * barycentric.z,
    colors[ao + 2] * barycentric.x + colors[bo + 2] * barycentric.y + colors[co + 2] * barycentric.z,
  );
}

function createTextureColorSampler(texture: Texture): TextureColorSampler | null {
  const image = texture.image as CanvasImageSource | undefined;
  const width = getCanvasImageWidth(image);
  const height = getCanvasImageHeight(image);
  if (!image || width <= 0 || height <= 0) {
    return null;
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    return null;
  }

  try {
    context.drawImage(image, 0, 0, width, height);
    return {
      width,
      height,
      flipY: texture.flipY,
      data: context.getImageData(0, 0, width, height).data,
    };
  } catch (error) {
    console.warn('Unable to sample the scan texture for Positive Limb color baking.', error);
    return null;
  }
}

function getCanvasImageWidth(image: CanvasImageSource | undefined): number {
  if (!image) {
    return 0;
  }

  const candidate = image as { naturalWidth?: number; videoWidth?: number; width?: number };
  return Math.floor(candidate.videoWidth ?? candidate.naturalWidth ?? candidate.width ?? 0);
}

function getCanvasImageHeight(image: CanvasImageSource | undefined): number {
  if (!image) {
    return 0;
  }

  const candidate = image as { naturalHeight?: number; videoHeight?: number; height?: number };
  return Math.floor(candidate.videoHeight ?? candidate.naturalHeight ?? candidate.height ?? 0);
}

function sampleTextureColor(sampler: TextureColorSampler, u: number, v: number, target: Vector3): Vector3 {
  const x = Math.round(clamp01(u) * (sampler.width - 1));
  const textureV = sampler.flipY ? 1 - clamp01(v) : clamp01(v);
  const y = Math.round(textureV * (sampler.height - 1));
  const offset = (y * sampler.width + x) * 4;
  return target.set(
    sampler.data[offset] / 255,
    sampler.data[offset + 1] / 255,
    sampler.data[offset + 2] / 255,
  );
}

function createMeshMaterials(
  texture: Texture | null,
  sculptMatcapTexture: CanvasTexture,
  faceMaterialIndices: Uint8Array | null,
  viewMode: MeshViewMode,
  useVertexColors = false,
  uiTheme: ViewportUiTheme = 'light',
): Material | Material[] {
  const baseMaterial = createScanViewMaterial(texture, sculptMatcapTexture, viewMode, useVertexColors, uiTheme);

  if (useVertexColors) {
    return baseMaterial;
  }

  if (!faceMaterialIndices?.some((materialIndex) => materialIndex !== 0)) {
    return baseMaterial;
  }

  const fillMaterial =
    viewMode === 'wireframe'
      ? new MeshBasicMaterial({
          color: uiTheme === 'dark' ? '#ffffff' : '#4b5563',
          side: DoubleSide,
          wireframe: true,
        })
      : new MeshMatcapMaterial({
          color: '#d8dde3',
          matcap: sculptMatcapTexture,
          side: DoubleSide,
        });
  return [baseMaterial, fillMaterial];
}

function createScanViewMaterial(
  texture: Texture | null,
  sculptMatcapTexture: CanvasTexture,
  viewMode: MeshViewMode,
  useVertexColors = false,
  uiTheme: ViewportUiTheme = 'light',
): Material {
  if (viewMode === 'wireframe') {
    return new MeshBasicMaterial({
      color: uiTheme === 'dark' ? '#ffffff' : '#26313d',
      side: DoubleSide,
      wireframe: true,
    });
  }

  if (useVertexColors) {
    return viewMode === 'shaded'
      ? new MeshMatcapMaterial({
          color: '#ffffff',
          matcap: sculptMatcapTexture,
          side: DoubleSide,
          vertexColors: true,
        })
      : new MeshBasicMaterial({
          side: DoubleSide,
          vertexColors: true,
        });
  }

  if (viewMode === 'shaded') {
    return new MeshMatcapMaterial({
      color: '#e8ebef',
      matcap: sculptMatcapTexture,
      side: DoubleSide,
    });
  }

  return texture
    ? new MeshBasicMaterial({
        map: texture,
        side: DoubleSide,
      })
    : new MeshMatcapMaterial({
        color: '#e8ebef',
        matcap: sculptMatcapTexture,
        side: DoubleSide,
        vertexColors: true,
      });
}

function applyFaceMaterialGroups(
  geometry: BufferGeometry,
  faceMaterialIndices: Uint8Array | null,
): void {
  geometry.clearGroups();
  if (!faceMaterialIndices?.some((materialIndex) => materialIndex !== 0)) {
    return;
  }

  let startTriangle = 0;
  let currentMaterial = faceMaterialIndices[0] ?? 0;
  for (let triangle = 1; triangle < faceMaterialIndices.length; triangle += 1) {
    const material = faceMaterialIndices[triangle] ?? 0;
    if (material === currentMaterial) {
      continue;
    }

    geometry.addGroup(startTriangle * 3, (triangle - startTriangle) * 3, currentMaterial);
    startTriangle = triangle;
    currentMaterial = material;
  }

  geometry.addGroup(
    startTriangle * 3,
    (faceMaterialIndices.length - startTriangle) * 3,
    currentMaterial,
  );
}

function createHoleFillFaceMaterialIndices(
  sourceFaceMaterialIndices: Uint8Array | null,
  previousTriangleCount: number,
  nextTriangleCount: number,
): Uint8Array | null {
  if (nextTriangleCount <= previousTriangleCount) {
    return sourceFaceMaterialIndices?.slice() ?? null;
  }

  const faceMaterialIndices = new Uint8Array(nextTriangleCount);
  if (sourceFaceMaterialIndices) {
    faceMaterialIndices.set(
      sourceFaceMaterialIndices.subarray(0, Math.min(sourceFaceMaterialIndices.length, previousTriangleCount)),
    );
  }

  faceMaterialIndices.fill(1, previousTriangleCount);
  return faceMaterialIndices;
}

function createHoleFillVertexColors(
  sourceColors: Float32Array,
  positions: ArrayLike<number>,
  preservedVertexCount: number,
  newVertexIds: ArrayLike<number>,
): Float32Array | null {
  const vertexCount = Math.floor(positions.length / 3);
  if (vertexCount === 0 || sourceColors.length < preservedVertexCount * 3) {
    return null;
  }

  const colors = new Float32Array(vertexCount * 3);
  const preservedLength = Math.min(sourceColors.length, preservedVertexCount * 3, colors.length);
  colors.set(sourceColors.subarray(0, preservedLength));

  if (vertexCount <= preservedVertexCount || newVertexIds.length === 0) {
    return colors;
  }

  for (let i = preservedVertexCount * 3; i < colors.length; i += 3) {
    colors[i] = 0.85;
    colors[i + 1] = 0.87;
    colors[i + 2] = 0.9;
  }

  for (let i = 0; i < newVertexIds.length; i += 1) {
    const vertex = newVertexIds[i];
    if (vertex < 0 || vertex >= vertexCount) {
      continue;
    }

    const offset = vertex * 3;
    colors[offset] = 0.85;
    colors[offset + 1] = 0.87;
    colors[offset + 2] = 0.9;
  }

  return colors;
}

function copyGeometryUvs(geometry: BufferGeometry): Float32Array | null {
  const uvAttribute = geometry.getAttribute('uv');
  if (!uvAttribute || uvAttribute.itemSize < 2) {
    return null;
  }

  const uvs = new Float32Array(uvAttribute.count * 2);
  for (let vertex = 0; vertex < uvAttribute.count; vertex += 1) {
    const offset = vertex * 2;
    uvs[offset] = uvAttribute.getX(vertex);
    uvs[offset + 1] = uvAttribute.getY(vertex);
  }

  return uvs;
}

function createHoleFillUvs(
  sourceUvs: Float32Array | null,
  positions: ArrayLike<number>,
  boundaryVertexIds: ArrayLike<number>,
  newVertexIds: ArrayLike<number>,
): Float32Array | null {
  if (!sourceUvs) {
    return null;
  }

  const vertexCount = Math.floor(positions.length / 3);
  const previousVertexCount = Math.floor(sourceUvs.length / 2);
  if (vertexCount <= previousVertexCount || newVertexIds.length === 0) {
    return sourceUvs.length === vertexCount * 2 ? sourceUvs.slice() : null;
  }

  const uvs = new Float32Array(vertexCount * 2);
  uvs.set(sourceUvs.subarray(0, Math.min(sourceUvs.length, uvs.length)));

  const samples = collectBoundaryUvSamples(positions, sourceUvs, boundaryVertexIds);
  if (samples.length === 0) {
    return uvs;
  }

  const uvFit = samples.length >= 3 ? fitBoundaryUvPlane(samples) : null;
  for (let i = 0; i < newVertexIds.length; i += 1) {
    const vertex = newVertexIds[i];
    if (vertex < 0 || vertex >= vertexCount) {
      continue;
    }

    const positionOffset = vertex * 3;
    const uvOffset = vertex * 2;
    if (uvFit) {
      const x =
        (positions[positionOffset] - uvFit.origin.x) * uvFit.axisU.x +
        (positions[positionOffset + 1] - uvFit.origin.y) * uvFit.axisU.y +
        (positions[positionOffset + 2] - uvFit.origin.z) * uvFit.axisU.z;
      const y =
        (positions[positionOffset] - uvFit.origin.x) * uvFit.axisV.x +
        (positions[positionOffset + 1] - uvFit.origin.y) * uvFit.axisV.y +
        (positions[positionOffset + 2] - uvFit.origin.z) * uvFit.axisV.z;
      uvs[uvOffset] = uvFit.u[0] * x + uvFit.u[1] * y + uvFit.u[2];
      uvs[uvOffset + 1] = uvFit.v[0] * x + uvFit.v[1] * y + uvFit.v[2];
    } else {
      const nearest = estimateUvByDistance(samples, positions, vertex);
      uvs[uvOffset] = nearest.x;
      uvs[uvOffset + 1] = nearest.y;
    }
  }

  return uvs;
}

interface BoundaryUvSample {
  position: Vector3;
  uv: Vector2;
}

interface BoundaryUvFit {
  origin: Vector3;
  axisU: Vector3;
  axisV: Vector3;
  u: [number, number, number];
  v: [number, number, number];
}

function collectBoundaryUvSamples(
  positions: ArrayLike<number>,
  sourceUvs: Float32Array,
  boundaryVertexIds: ArrayLike<number>,
): BoundaryUvSample[] {
  const vertexCount = Math.floor(positions.length / 3);
  const uvVertexCount = Math.floor(sourceUvs.length / 2);
  const samples: BoundaryUvSample[] = [];
  const seen = new Set<number>();

  for (let i = 0; i < boundaryVertexIds.length; i += 1) {
    const vertex = boundaryVertexIds[i];
    if (vertex < 0 || vertex >= vertexCount || vertex >= uvVertexCount || seen.has(vertex)) {
      continue;
    }

    seen.add(vertex);
    const positionOffset = vertex * 3;
    const uvOffset = vertex * 2;
    samples.push({
      position: new Vector3(
        positions[positionOffset],
        positions[positionOffset + 1],
        positions[positionOffset + 2],
      ),
      uv: new Vector2(sourceUvs[uvOffset], sourceUvs[uvOffset + 1]),
    });
  }

  return samples;
}

function fitBoundaryUvPlane(samples: BoundaryUvSample[]): BoundaryUvFit | null {
  const origin = new Vector3();
  for (let i = 0; i < samples.length; i += 1) {
    origin.add(samples[i].position);
  }
  origin.multiplyScalar(1 / samples.length);

  const normal = new Vector3();
  for (let i = 0; i < samples.length; i += 1) {
    const current = samples[i].position.clone().sub(origin);
    const next = samples[(i + 1) % samples.length].position.clone().sub(origin);
    normal.add(current.cross(next));
  }
  if (normal.lengthSq() <= 1e-12) {
    return null;
  }
  normal.normalize();

  let axisU = new Vector3();
  let farthestDistanceSq = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const candidate = samples[i].position.clone().sub(origin);
    const distanceSq = candidate.lengthSq();
    if (distanceSq > farthestDistanceSq) {
      farthestDistanceSq = distanceSq;
      axisU = candidate;
    }
  }
  if (axisU.lengthSq() <= 1e-12) {
    return null;
  }
  axisU.normalize();
  const axisV = normal.clone().cross(axisU).normalize();

  const matrix = [0, 0, 0, 0, 0, 0, 0, 0, samples.length] as [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  const rhsU: [number, number, number] = [0, 0, 0];
  const rhsV: [number, number, number] = [0, 0, 0];

  for (let i = 0; i < samples.length; i += 1) {
    const delta = samples[i].position.clone().sub(origin);
    const x = delta.dot(axisU);
    const y = delta.dot(axisV);
    matrix[0] += x * x;
    matrix[1] += x * y;
    matrix[2] += x;
    matrix[3] += x * y;
    matrix[4] += y * y;
    matrix[5] += y;
    matrix[6] += x;
    matrix[7] += y;
    rhsU[0] += x * samples[i].uv.x;
    rhsU[1] += y * samples[i].uv.x;
    rhsU[2] += samples[i].uv.x;
    rhsV[0] += x * samples[i].uv.y;
    rhsV[1] += y * samples[i].uv.y;
    rhsV[2] += samples[i].uv.y;
  }

  const u = solveThreeByThree(matrix, rhsU);
  const v = solveThreeByThree(matrix, rhsV);
  if (!u || !v) {
    return null;
  }

  return { origin, axisU, axisV, u, v };
}

function solveThreeByThree(
  matrix: [number, number, number, number, number, number, number, number, number],
  rhs: [number, number, number],
): [number, number, number] | null {
  const determinant =
    matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
    matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
    matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  if (Math.abs(determinant) <= 1e-12) {
    return null;
  }

  const detX =
    rhs[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
    matrix[1] * (rhs[1] * matrix[8] - matrix[5] * rhs[2]) +
    matrix[2] * (rhs[1] * matrix[7] - matrix[4] * rhs[2]);
  const detY =
    matrix[0] * (rhs[1] * matrix[8] - matrix[5] * rhs[2]) -
    rhs[0] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
    matrix[2] * (matrix[3] * rhs[2] - rhs[1] * matrix[6]);
  const detZ =
    matrix[0] * (matrix[4] * rhs[2] - rhs[1] * matrix[7]) -
    matrix[1] * (matrix[3] * rhs[2] - rhs[1] * matrix[6]) +
    rhs[0] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);

  return [detX / determinant, detY / determinant, detZ / determinant];
}

function estimateUvByDistance(
  samples: BoundaryUvSample[],
  positions: ArrayLike<number>,
  vertex: number,
): Vector2 {
  const offset = vertex * 3;
  const target = new Vector3(positions[offset], positions[offset + 1], positions[offset + 2]);
  let sumWeight = 0;
  const uv = new Vector2();

  for (let i = 0; i < samples.length; i += 1) {
    const distanceSq = target.distanceToSquared(samples[i].position);
    const weight = 1 / Math.max(distanceSq, 1e-10);
    uv.addScaledVector(samples[i].uv, weight);
    sumWeight += weight;
  }

  if (sumWeight > 0) {
    uv.multiplyScalar(1 / sumWeight);
  }

  return uv;
}

function orientGeometryOutward(geometry: BufferGeometry): void {
  const positionAttribute = geometry.getAttribute('position');
  const indexAttribute = geometry.getIndex();
  if (!positionAttribute || !indexAttribute) {
    return;
  }

  const positions = positionAttribute.array as ArrayLike<number>;
  const indexArray = indexAttribute.array as Uint16Array | Uint32Array;
  if (positionAttribute.count === 0 || indexArray.length < 3) {
    return;
  }

  makeTriangleWindingCoherent(indexArray);

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
  let signedVolume = 0;
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
    signedVolume +=
      ax * (by * cz - bz * cy) +
      bx * (cy * az - cz * ay) +
      cx * (ay * bz - az * by);
  }

  if (Math.abs(signedVolume) > 1e-8 ? signedVolume < 0 : orientationScore < 0) {
    flipAllTriangles(indexArray);
  }

  geometry.deleteAttribute('normal');
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  indexAttribute.needsUpdate = true;
}

interface TriangleEdgeUse {
  triangle: number;
  a: number;
  b: number;
}

function makeTriangleWindingCoherent(indices: Uint16Array | Uint32Array): void {
  const triangleCount = Math.floor(indices.length / 3);
  const edgeUsesByKey = new Map<string, TriangleEdgeUse[]>();
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 3;
    collectTriangleEdgeUse(edgeUsesByKey, triangle, indices[offset], indices[offset + 1]);
    collectTriangleEdgeUse(edgeUsesByKey, triangle, indices[offset + 1], indices[offset + 2]);
    collectTriangleEdgeUse(edgeUsesByKey, triangle, indices[offset + 2], indices[offset]);
  }

  const neighbors: Array<Array<{ triangle: number; sameDirection: boolean }>> = Array.from(
    { length: triangleCount },
    () => [],
  );
  for (const uses of edgeUsesByKey.values()) {
    for (let i = 0; i < uses.length; i += 1) {
      for (let j = i + 1; j < uses.length; j += 1) {
        const left = uses[i];
        const right = uses[j];
        const sameDirection = left.a === right.a && left.b === right.b;
        neighbors[left.triangle].push({ triangle: right.triangle, sameDirection });
        neighbors[right.triangle].push({ triangle: left.triangle, sameDirection });
      }
    }
  }

  const flipped = new Int8Array(triangleCount);
  flipped.fill(-1);
  const stack: number[] = [];
  for (let start = 0; start < triangleCount; start += 1) {
    if (flipped[start] !== -1) {
      continue;
    }

    flipped[start] = 0;
    stack.push(start);
    while (stack.length > 0) {
      const triangle = stack.pop() as number;
      const currentFlip = flipped[triangle];
      const linked = neighbors[triangle];
      for (let i = 0; i < linked.length; i += 1) {
        const neighbor = linked[i];
        const targetFlip = currentFlip ^ (neighbor.sameDirection ? 1 : 0);
        if (flipped[neighbor.triangle] !== -1) {
          continue;
        }

        flipped[neighbor.triangle] = targetFlip;
        stack.push(neighbor.triangle);
      }
    }
  }

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    if (flipped[triangle] === 1) {
      flipTriangle(indices, triangle);
    }
  }
}

function collectTriangleEdgeUse(
  edgeUsesByKey: Map<string, TriangleEdgeUse[]>,
  triangle: number,
  a: number,
  b: number,
): void {
  const low = Math.min(a, b);
  const high = Math.max(a, b);
  const key = `${low}:${high}`;
  const uses = edgeUsesByKey.get(key);
  const use = { triangle, a, b };
  if (uses) {
    uses.push(use);
    return;
  }

  edgeUsesByKey.set(key, [use]);
}

function flipAllTriangles(indices: Uint16Array | Uint32Array): void {
  for (let triangle = 0; triangle < indices.length / 3; triangle += 1) {
    flipTriangle(indices, triangle);
  }
}

function flipTriangle(indices: Uint16Array | Uint32Array, triangle: number): void {
  const offset = triangle * 3;
  const swap = indices[offset + 1];
  indices[offset + 1] = indices[offset + 2];
  indices[offset + 2] = swap;
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
    uvs: null,
    colors: null,
    bakedVertexColorsActive: false,
    history: null,
    selectedTriangleMask: null,
    selectedTriangleCount: 0,
    faceMaterialIndices: null,
    meshViewMode: 'colored',
    rotationSessionAngles: { x: 0, y: 0, z: 0 },
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
