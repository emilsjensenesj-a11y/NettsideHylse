import './styles.css';
import './three-bvh';

import { createEditableMeshData } from './sculpt/editable-mesh';
import { loadMeshFile } from './io/mesh-loader';
import type { MeshUnit } from './io/mesh-loader';
import { ViewportController } from './render/viewport';
import { SculptEngine } from './sculpt/sculpt-engine';
import type {
  BoundaryWorkflowState,
  BrushType,
  HistoryState,
  InteractionMode,
  MeshStats,
  SelectionState,
  SelectionTool,
} from './sculpt/types';

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) {
  throw new Error('App root not found.');
}

app.innerHTML = `
  <div class="app-shell">
    <aside class="toolbar">
      <div class="panel-header">
        <p class="eyebrow">Fast V1 Mesh Sculptor</p>
        <h1>Low-latency browser sculpting and face selection for local meshes.</h1>
      </div>

      <div class="control-group">
        <button id="open-file" class="primary-button">Open STL / OBJ / PLY</button>
        <input id="file-input" type="file" accept=".stl,.obj,.ply" hidden />
        <button id="reset-view" class="secondary-button">Reset View</button>
      </div>

      <div class="control-group split">
        <label class="field">
          <span>Mode</span>
          <select id="mode-select">
            <option value="sculpt">Sculpt</option>
            <option value="select">Select</option>
            <option value="fill">Fill Hole</option>
            <option value="boundary">Socket Model</option>
            <option value="positive">Positive Socket</option>
            <option value="remesh">Remesh</option>
            <option value="thicken">Thicken</option>
          </select>
        </label>
        <label class="field">
          <span>Wireframe</span>
          <input id="wireframe-toggle" type="checkbox" />
        </label>
      </div>

      <div id="sculpt-controls" class="control-group">
        <label class="field">
          <span>Brush</span>
          <select id="brush-select">
            <option value="bump">Bump / Inflate</option>
            <option value="smooth">Smooth</option>
            <option value="flatten">Flatten</option>
          </select>
        </label>

        <label class="field range-field">
          <span>Brush Radius <strong id="radius-value">5.0 mm</strong></span>
          <input id="radius-slider" type="range" min="0.1" max="100" step="0.1" value="5" />
        </label>

        <label class="field range-field">
          <span>Strength <strong id="strength-value">0.35</strong></span>
          <input id="strength-slider" type="range" min="0.02" max="1" step="0.01" value="0.35" />
        </label>
      </div>

      <div id="selection-controls" class="control-group" hidden>
        <label class="field">
          <span>Selection Tool</span>
          <select id="selection-tool-select">
            <option value="sphere">Sphere</option>
            <option value="box">Box</option>
            <option value="snip">Snip / Lasso</option>
          </select>
        </label>

        <label id="selection-radius-field" class="field range-field">
          <span>Selection Radius <strong id="selection-radius-value">6.0 mm</strong></span>
          <input id="selection-radius-slider" type="range" min="0.1" max="120" step="0.1" value="6" />
        </label>

        <label class="field range-field">
          <span>Smooth Intensity <strong id="selection-smooth-strength-value">0.35</strong></span>
          <input id="selection-smooth-strength-slider" type="range" min="0.05" max="1" step="0.01" value="0.35" />
        </label>

        <label class="field range-field">
          <span>Smooth Iterations <strong id="selection-smooth-iterations-value">6</strong></span>
          <input id="selection-smooth-iterations-slider" type="range" min="1" max="20" step="1" value="6" />
        </label>

        <label class="field range-field">
          <span>Remesh Edge Size <strong id="selection-remesh-edge-value">0.250 mm</strong></span>
          <input id="selection-remesh-edge-slider" type="range" min="0.05" max="1" step="0.01" value="0.25" />
        </label>

        <div class="control-group split">
          <button id="clear-selection-button" class="secondary-button" disabled>Clear</button>
          <button id="delete-selection-button" class="secondary-button" disabled>Delete</button>
        </div>

        <div class="control-group split">
          <button id="smooth-selection-button" class="secondary-button" disabled>Smooth</button>
          <button id="refine-selection-button" class="secondary-button" disabled>Refine</button>
        </div>

        <div class="control-group">
          <button id="smooth-boundary-button" class="secondary-button" disabled>Smooth Boundary</button>
        </div>

        <div class="control-group">
          <button id="remesh-selection-button" class="secondary-button" disabled>Remesh Selected</button>
        </div>
      </div>

      <div id="fill-controls" class="control-group" hidden>
        <p class="inline-note">
          Hover a bright-blue boundary loop until it turns purple, then left click to patch it.
        </p>
      </div>

      <div id="boundary-controls" class="control-group" hidden>
        <div class="stepper-header">
          <div>
            <p class="mini-heading">Socket Model</p>
            <p id="socket-model-step-label" class="inline-note">Step 1 of 6</p>
          </div>
          <div class="stepper-actions">
            <button id="socket-model-prev-button" class="secondary-button">Previous</button>
            <button id="socket-model-next-button" class="secondary-button">Next</button>
          </div>
        </div>
        <p id="socket-model-step-title" class="inline-note">Target Boundary</p>
        <p id="boundary-target-status" class="inline-note">No boundary targeted yet.</p>

        <div id="socket-step-target" class="socket-step-panel control-group">
          <p class="inline-note">
            Hover a bright-blue boundary loop until it turns purple, then left click to target it for the socket workflow.
          </p>
        </div>

        <div id="socket-step-boundary-smooth" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Boundary Smooth</p>
          <label class="field range-field">
            <span>Strength <strong id="boundary-smooth-strength-value">0.35</strong></span>
            <input id="boundary-smooth-strength-slider" type="range" min="0.05" max="1" step="0.01" value="0.35" />
          </label>
          <p class="inline-note">Fixed at 10 iterations. The slider previews the boundary smooth live, and Next commits it.</p>
        </div>

        <div id="socket-step-remesh" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Fixed-Boundary Remesh</p>
          <label class="field range-field">
            <span>Target Edge Size <strong id="boundary-remesh-edge-value">0.250 mm</strong></span>
            <input id="boundary-remesh-edge-slider" type="range" min="0.05" max="1" step="0.01" value="0.25" />
          </label>
          <p class="inline-note">This is the only slow step in Socket Model, so it does not preview live. Press Next to run it.</p>
        </div>

        <div id="socket-step-thicken" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Thicken</p>
          <label class="field range-field">
            <span>Thickness <strong id="boundary-thicken-value">0.250 mm</strong></span>
            <input id="boundary-thicken-slider" type="range" min="0.05" max="1" step="0.01" value="0.25" />
          </label>
          <p class="inline-note">The thickness slider previews live on the current socket shell, and Next commits it.</p>
        </div>

        <div id="socket-step-offset" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Offset</p>
          <label class="field range-field">
            <span>Band Distance <strong id="boundary-band-distance-value">1.500 mm</strong></span>
            <input id="boundary-band-distance-slider" type="range" min="0.1" max="10" step="0.05" value="1.5" />
          </label>
          <p class="inline-note">
            The band preview updates live. Press Next to commit the band and auto-remesh the selected region at thickness / 8.
          </p>
        </div>

        <div id="socket-step-final-smooth" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Selected Band Smooth</p>
          <label class="field range-field">
            <span>Strength <strong id="boundary-selected-smooth-strength-value">0.35</strong></span>
            <input id="boundary-selected-smooth-strength-slider" type="range" min="0.05" max="1" step="0.01" value="0.35" />
          </label>
          <p class="inline-note">Fixed at 40 iterations. The slider previews the final relax live, and Finish commits it.</p>
        </div>

        <div id="socket-step-complete" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Complete</p>
          <p class="inline-note">
            The current socket pass is finished. Press Next to start over on a new boundary, or Previous to inspect the earlier stages.
          </p>
        </div>
      </div>

      <div id="positive-controls" class="control-group" hidden>
        <div class="stepper-header">
          <div>
            <p class="mini-heading">Positive Socket</p>
            <p id="positive-socket-step-label" class="inline-note">Step 1 of 8</p>
          </div>
          <div class="stepper-actions">
            <button id="positive-socket-prev-button" class="secondary-button">Previous</button>
            <button id="positive-socket-next-button" class="secondary-button">Next</button>
          </div>
        </div>
        <p id="positive-socket-step-title" class="inline-note">Target Boundary</p>
        <p id="positive-boundary-target-status" class="inline-note">No boundary targeted yet.</p>

        <div id="positive-step-target" class="socket-step-panel control-group">
          <p class="inline-note">
            Hover a bright-blue boundary loop until it turns purple, then left click to target it for the positive socket workflow.
          </p>
        </div>

        <div id="positive-step-boundary-smooth" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Boundary Smooth</p>
          <label class="field range-field">
            <span>Strength <strong id="positive-boundary-smooth-strength-value">0.35</strong></span>
            <input id="positive-boundary-smooth-strength-slider" type="range" min="0.05" max="1" step="0.01" value="0.35" />
          </label>
          <p class="inline-note">Fixed at 10 iterations. The slider previews the boundary smooth live, and Next commits it.</p>
        </div>

        <div id="positive-step-remesh" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Fixed-Boundary Remesh</p>
          <label class="field range-field">
            <span>Target Edge Size <strong id="positive-remesh-edge-value">0.250 mm</strong></span>
            <input id="positive-remesh-edge-slider" type="range" min="0.05" max="1" step="0.01" value="0.25" />
          </label>
          <p class="inline-note">This is the only slow step in Positive Socket, so it does not preview live. Press Next to run it.</p>
        </div>

        <div id="positive-step-extrude" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Extrude Boundary</p>
          <label class="field range-field">
            <span>Extrude Distance <strong id="positive-extrude-distance-value">0.250 mm</strong></span>
            <input id="positive-extrude-distance-slider" type="range" min="0.05" max="1" step="0.01" value="0.25" />
          </label>
          <p class="inline-note">The extrude distance previews live on the remeshed boundary, and Next commits it.</p>
        </div>

        <div id="positive-step-offset" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Offset</p>
          <label class="field range-field">
            <span>Band Distance <strong id="positive-band-distance-value">1.500 mm</strong></span>
            <input id="positive-band-distance-slider" type="range" min="0.1" max="10" step="0.05" value="1.5" />
          </label>
          <p class="inline-note">
            The band preview updates live. Press Next to commit the band and auto-remesh the selected region at extrude distance / 8.
          </p>
        </div>

        <div id="positive-step-final-smooth" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Band Smooth</p>
          <label class="field range-field">
            <span>Strength <strong id="positive-selected-smooth-strength-value">0.35</strong></span>
            <input id="positive-selected-smooth-strength-slider" type="range" min="0.05" max="1" step="0.01" value="0.35" />
          </label>
          <p class="inline-note">Fixed at 40 iterations. The slider previews the final relax live while keeping the selected patch boundary fixed.</p>
        </div>

        <div id="positive-step-directional-extrude" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Directional Wall Extrude</p>
          <label class="field range-field">
            <span>X Tilt <strong id="positive-directional-tilt-x-value">0.0°</strong></span>
            <input id="positive-directional-tilt-x-slider" type="range" min="-45" max="45" step="1" value="0" />
          </label>
          <label class="field range-field">
            <span>Y Tilt <strong id="positive-directional-tilt-y-value">0.0°</strong></span>
            <input id="positive-directional-tilt-y-slider" type="range" min="-45" max="45" step="1" value="0" />
          </label>
          <p class="inline-note">
            This final wall extrusion uses the current outer boundary, the unified boundary normals, and a plane test to pick the outward side. The wall length matches the largest edge of the mesh bounding box.
          </p>
        </div>

        <div id="positive-step-complete" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Complete</p>
          <p class="inline-note">
            The current positive socket pass is finished. Press Next to start over on a new boundary, or Previous to inspect the earlier stages.
          </p>
        </div>
      </div>

      <div id="remesh-controls" class="control-group" hidden>
        <label class="field range-field">
          <span>Target Edge Size <strong id="remesh-edge-value">0.250 mm</strong></span>
          <input id="remesh-edge-slider" type="range" min="0.05" max="1" step="0.01" value="0.25" />
        </label>
        <label class="field">
          <span>Boundary</span>
          <select id="remesh-boundary-select">
            <option value="fixed">Fixed Boundary</option>
            <option value="refined" selected>Refined Boundary</option>
            <option value="free">Free Boundary</option>
          </select>
        </label>
        <button id="apply-remesh-button" class="secondary-button">Apply Remesh</button>
      </div>

      <div id="thicken-controls" class="control-group" hidden>
        <label class="field range-field">
          <span>Thickness <strong id="thicken-value">0.250 mm</strong></span>
          <input id="thicken-slider" type="range" min="0.05" max="1" step="0.01" value="0.25" />
        </label>
        <button id="apply-thicken-button" class="secondary-button">Apply Thicken</button>
      </div>

      <div id="history-controls" class="control-group split">
        <button id="undo-button" class="secondary-button" disabled>Undo</button>
        <button id="redo-button" class="secondary-button" disabled>Redo</button>
      </div>

      <div class="hint-box">
        <p><strong>Controls</strong></p>
        <p id="mode-hint-primary">Left drag sculpts and middle drag orbits.</p>
        <p id="mode-hint-secondary">Mouse wheel zooms and right drag pans through OrbitControls.</p>
        <p id="mode-hint-tertiary">In Select mode, <kbd>Shift</kbd> adds and <kbd>Ctrl</kbd> subtracts.</p>
      </div>

      <div class="stats-card">
        <p class="stats-title">Mesh Stats</p>
        <div class="stats-grid">
          <span>File</span>
          <strong id="file-name">None loaded</strong>
          <span>Vertices</span>
          <strong id="vertex-count">0</strong>
          <span>Triangles</span>
          <strong id="triangle-count">0</strong>
          <span>Selected Faces</span>
          <strong id="selected-triangle-count">0</strong>
        </div>
      </div>

      <p id="status" class="status">Open a local STL, OBJ, or PLY mesh to begin sculpting.</p>
    </aside>

    <main class="viewport-panel">
      <div id="viewport" class="viewport-host"></div>
    </main>
  </div>

  <div id="import-unit-modal" class="modal-backdrop" hidden>
    <div class="modal-card">
      <p class="modal-eyebrow">Import Units</p>
      <h2 id="import-unit-title">Choose the source unit for this mesh</h2>
      <p class="modal-body">
        The app will convert the mesh to internal millimeters so brush size, remesh size, and thickness stay unit-based.
      </p>
      <div class="modal-actions vertical">
        <button id="import-unit-mm" class="primary-button">This mesh is in millimeters (mm)</button>
        <button id="import-unit-cm" class="secondary-button">This mesh is in centimeters (cm)</button>
        <button id="import-unit-m" class="secondary-button">This mesh is in meters (m)</button>
      </div>
      <div class="modal-actions">
        <button id="import-unit-cancel" class="secondary-button">Cancel</button>
      </div>
    </div>
  </div>
`;

const openFileButton = requireElement<HTMLButtonElement>('open-file');
const resetViewButton = requireElement<HTMLButtonElement>('reset-view');
const fileInput = requireElement<HTMLInputElement>('file-input');
const modeSelect = requireElement<HTMLSelectElement>('mode-select');
const brushSelect = requireElement<HTMLSelectElement>('brush-select');
const radiusSlider = requireElement<HTMLInputElement>('radius-slider');
const radiusValue = requireElement<HTMLElement>('radius-value');
const strengthSlider = requireElement<HTMLInputElement>('strength-slider');
const strengthValue = requireElement<HTMLElement>('strength-value');
const selectionToolSelect = requireElement<HTMLSelectElement>('selection-tool-select');
const selectionRadiusField = requireElement<HTMLElement>('selection-radius-field');
const selectionRadiusSlider = requireElement<HTMLInputElement>('selection-radius-slider');
const selectionRadiusValue = requireElement<HTMLElement>('selection-radius-value');
const selectionSmoothStrengthSlider = requireElement<HTMLInputElement>('selection-smooth-strength-slider');
const selectionSmoothStrengthValue = requireElement<HTMLElement>('selection-smooth-strength-value');
const selectionSmoothIterationsSlider = requireElement<HTMLInputElement>('selection-smooth-iterations-slider');
const selectionSmoothIterationsValue = requireElement<HTMLElement>('selection-smooth-iterations-value');
const selectionRemeshEdgeSlider = requireElement<HTMLInputElement>('selection-remesh-edge-slider');
const selectionRemeshEdgeValue = requireElement<HTMLElement>('selection-remesh-edge-value');
const wireframeToggle = requireElement<HTMLInputElement>('wireframe-toggle');
const clearSelectionButton = requireElement<HTMLButtonElement>('clear-selection-button');
const deleteSelectionButton = requireElement<HTMLButtonElement>('delete-selection-button');
const smoothSelectionButton = requireElement<HTMLButtonElement>('smooth-selection-button');
const refineSelectionButton = requireElement<HTMLButtonElement>('refine-selection-button');
const smoothBoundaryButton = requireElement<HTMLButtonElement>('smooth-boundary-button');
const remeshSelectionButton = requireElement<HTMLButtonElement>('remesh-selection-button');
const fillControls = requireElement<HTMLElement>('fill-controls');
const boundaryControls = requireElement<HTMLElement>('boundary-controls');
const positiveControls = requireElement<HTMLElement>('positive-controls');
const socketModelStepLabel = requireElement<HTMLElement>('socket-model-step-label');
const socketModelStepTitle = requireElement<HTMLElement>('socket-model-step-title');
const socketModelPrevButton = requireElement<HTMLButtonElement>('socket-model-prev-button');
const socketModelNextButton = requireElement<HTMLButtonElement>('socket-model-next-button');
const socketStepTarget = requireElement<HTMLElement>('socket-step-target');
const socketStepBoundarySmooth = requireElement<HTMLElement>('socket-step-boundary-smooth');
const socketStepRemesh = requireElement<HTMLElement>('socket-step-remesh');
const socketStepThicken = requireElement<HTMLElement>('socket-step-thicken');
const socketStepOffset = requireElement<HTMLElement>('socket-step-offset');
const socketStepFinalSmooth = requireElement<HTMLElement>('socket-step-final-smooth');
const socketStepComplete = requireElement<HTMLElement>('socket-step-complete');
const boundaryTargetStatus = requireElement<HTMLElement>('boundary-target-status');
const boundarySmoothStrengthSlider = requireElement<HTMLInputElement>('boundary-smooth-strength-slider');
const boundarySmoothStrengthValue = requireElement<HTMLElement>('boundary-smooth-strength-value');
const boundaryRemeshEdgeSlider = requireElement<HTMLInputElement>('boundary-remesh-edge-slider');
const boundaryRemeshEdgeValue = requireElement<HTMLElement>('boundary-remesh-edge-value');
const boundaryThickenSlider = requireElement<HTMLInputElement>('boundary-thicken-slider');
const boundaryThickenValue = requireElement<HTMLElement>('boundary-thicken-value');
const boundaryBandDistanceSlider = requireElement<HTMLInputElement>('boundary-band-distance-slider');
const boundaryBandDistanceValue = requireElement<HTMLElement>('boundary-band-distance-value');
const boundarySelectedSmoothStrengthSlider = requireElement<HTMLInputElement>('boundary-selected-smooth-strength-slider');
const boundarySelectedSmoothStrengthValue = requireElement<HTMLElement>('boundary-selected-smooth-strength-value');
const positiveSocketStepLabel = requireElement<HTMLElement>('positive-socket-step-label');
const positiveSocketStepTitle = requireElement<HTMLElement>('positive-socket-step-title');
const positiveSocketPrevButton = requireElement<HTMLButtonElement>('positive-socket-prev-button');
const positiveSocketNextButton = requireElement<HTMLButtonElement>('positive-socket-next-button');
const positiveStepTarget = requireElement<HTMLElement>('positive-step-target');
const positiveStepBoundarySmooth = requireElement<HTMLElement>('positive-step-boundary-smooth');
const positiveStepRemesh = requireElement<HTMLElement>('positive-step-remesh');
const positiveStepExtrude = requireElement<HTMLElement>('positive-step-extrude');
const positiveStepOffset = requireElement<HTMLElement>('positive-step-offset');
const positiveStepFinalSmooth = requireElement<HTMLElement>('positive-step-final-smooth');
const positiveStepDirectionalExtrude = requireElement<HTMLElement>('positive-step-directional-extrude');
const positiveStepComplete = requireElement<HTMLElement>('positive-step-complete');
const positiveBoundaryTargetStatus = requireElement<HTMLElement>('positive-boundary-target-status');
const positiveBoundarySmoothStrengthSlider = requireElement<HTMLInputElement>('positive-boundary-smooth-strength-slider');
const positiveBoundarySmoothStrengthValue = requireElement<HTMLElement>('positive-boundary-smooth-strength-value');
const positiveRemeshEdgeSlider = requireElement<HTMLInputElement>('positive-remesh-edge-slider');
const positiveRemeshEdgeValue = requireElement<HTMLElement>('positive-remesh-edge-value');
const positiveExtrudeDistanceSlider = requireElement<HTMLInputElement>('positive-extrude-distance-slider');
const positiveExtrudeDistanceValue = requireElement<HTMLElement>('positive-extrude-distance-value');
const positiveBandDistanceSlider = requireElement<HTMLInputElement>('positive-band-distance-slider');
const positiveBandDistanceValue = requireElement<HTMLElement>('positive-band-distance-value');
const positiveSelectedSmoothStrengthSlider = requireElement<HTMLInputElement>('positive-selected-smooth-strength-slider');
const positiveSelectedSmoothStrengthValue = requireElement<HTMLElement>('positive-selected-smooth-strength-value');
const positiveDirectionalTiltXSlider = requireElement<HTMLInputElement>('positive-directional-tilt-x-slider');
const positiveDirectionalTiltXValue = requireElement<HTMLElement>('positive-directional-tilt-x-value');
const positiveDirectionalTiltYSlider = requireElement<HTMLInputElement>('positive-directional-tilt-y-slider');
const positiveDirectionalTiltYValue = requireElement<HTMLElement>('positive-directional-tilt-y-value');
const remeshControls = requireElement<HTMLElement>('remesh-controls');
const remeshEdgeSlider = requireElement<HTMLInputElement>('remesh-edge-slider');
const remeshEdgeValue = requireElement<HTMLElement>('remesh-edge-value');
const remeshBoundarySelect = requireElement<HTMLSelectElement>('remesh-boundary-select');
const applyRemeshButton = requireElement<HTMLButtonElement>('apply-remesh-button');
const thickenControls = requireElement<HTMLElement>('thicken-controls');
const thickenSlider = requireElement<HTMLInputElement>('thicken-slider');
const thickenValue = requireElement<HTMLElement>('thicken-value');
const applyThickenButton = requireElement<HTMLButtonElement>('apply-thicken-button');
const importUnitModal = requireElement<HTMLElement>('import-unit-modal');
const importUnitTitle = requireElement<HTMLElement>('import-unit-title');
const importUnitMillimetersButton = requireElement<HTMLButtonElement>('import-unit-mm');
const importUnitCentimetersButton = requireElement<HTMLButtonElement>('import-unit-cm');
const importUnitMetersButton = requireElement<HTMLButtonElement>('import-unit-m');
const importUnitCancelButton = requireElement<HTMLButtonElement>('import-unit-cancel');
const historyControls = requireElement<HTMLElement>('history-controls');
const undoButton = requireElement<HTMLButtonElement>('undo-button');
const redoButton = requireElement<HTMLButtonElement>('redo-button');
const sculptControls = requireElement<HTMLElement>('sculpt-controls');
const selectionControls = requireElement<HTMLElement>('selection-controls');
const fileName = requireElement<HTMLElement>('file-name');
const vertexCount = requireElement<HTMLElement>('vertex-count');
const triangleCount = requireElement<HTMLElement>('triangle-count');
const selectedTriangleCount = requireElement<HTMLElement>('selected-triangle-count');
const status = requireElement<HTMLElement>('status');
const modeHintPrimary = requireElement<HTMLElement>('mode-hint-primary');
const modeHintSecondary = requireElement<HTMLElement>('mode-hint-secondary');
const modeHintTertiary = requireElement<HTMLElement>('mode-hint-tertiary');
const viewportHost = requireElement<HTMLElement>('viewport');

let currentFilename = 'None loaded';
const SOCKET_MODEL_TARGET_STEP_INDEX = 0;
const SOCKET_MODEL_BOUNDARY_SMOOTH_STEP_INDEX = 1;
const SOCKET_MODEL_REMESH_STEP_INDEX = 2;
const SOCKET_MODEL_THICKEN_STEP_INDEX = 3;
const SOCKET_MODEL_OFFSET_STEP_INDEX = 4;
const SOCKET_MODEL_FINAL_SMOOTH_STEP_INDEX = 5;
const SOCKET_MODEL_COMPLETE_STEP_INDEX = 6;
const SOCKET_MODEL_STEP_TITLES = [
  'Target Boundary',
  'Boundary Smooth',
  'Fixed-Boundary Remesh',
  'Thicken',
  'Offset',
  'Final Smooth',
  'Complete',
] as const;
const SOCKET_MODEL_STEP_PANELS = [
  socketStepTarget,
  socketStepBoundarySmooth,
  socketStepRemesh,
  socketStepThicken,
  socketStepOffset,
  socketStepFinalSmooth,
  socketStepComplete,
] as const;
let socketModelStepIndex = 0;
const POSITIVE_SOCKET_TARGET_STEP_INDEX = 0;
const POSITIVE_SOCKET_BOUNDARY_SMOOTH_STEP_INDEX = 1;
const POSITIVE_SOCKET_REMESH_STEP_INDEX = 2;
const POSITIVE_SOCKET_EXTRUDE_STEP_INDEX = 3;
const POSITIVE_SOCKET_OFFSET_STEP_INDEX = 4;
const POSITIVE_SOCKET_FINAL_SMOOTH_STEP_INDEX = 5;
const POSITIVE_SOCKET_DIRECTIONAL_EXTRUDE_STEP_INDEX = 6;
const POSITIVE_SOCKET_COMPLETE_STEP_INDEX = 7;
const POSITIVE_SOCKET_STEP_TITLES = [
  'Target Boundary',
  'Boundary Smooth',
  'Fixed-Boundary Remesh',
  'Extrude Boundary',
  'Offset',
  'Final Smooth',
  'Directional Wall Extrude',
  'Complete',
] as const;
const POSITIVE_SOCKET_STEP_PANELS = [
  positiveStepTarget,
  positiveStepBoundarySmooth,
  positiveStepRemesh,
  positiveStepExtrude,
  positiveStepOffset,
  positiveStepFinalSmooth,
  positiveStepDirectionalExtrude,
  positiveStepComplete,
] as const;
let positiveSocketStepIndex = 0;

const viewport = new ViewportController(viewportHost, {
  onHistoryChange: updateHistoryButtons,
  onSelectionChange: updateSelectionUi,
  onBoundaryWorkflowChange: updateBoundaryWorkflowUi,
  onBoundaryAction: ({ success, message }) => setStatus(message, !success),
  onMeshStatsChange: updateMeshStats,
  onHoleFill: ({ success, message }) => setStatus(message, !success),
});

viewport.setBrushType(brushSelect.value as BrushType);
viewport.setBrushRadiusMm(Number(radiusSlider.value));
viewport.setBrushStrength(Number(strengthSlider.value));
viewport.setInteractionMode(modeSelect.value as InteractionMode);
viewport.setSelectionTool(selectionToolSelect.value as SelectionTool);
viewport.setSelectionRadiusMm(Number(selectionRadiusSlider.value));
viewport.setWireframe(wireframeToggle.checked);
syncModeUi();
syncSelectionUi();
configureOperationSliders(0.5);
updateBoundaryWorkflowUi(viewport.getBoundaryWorkflowState());
syncSocketModelStepUi();
syncPositiveSocketStepUi();

openFileButton.addEventListener('click', () => fileInput.click());
resetViewButton.addEventListener('click', () => viewport.resetView());
undoButton.addEventListener('click', () => viewport.undo());
redoButton.addEventListener('click', () => viewport.redo());
clearSelectionButton.addEventListener('click', () => {
  if (viewport.clearSelection()) {
    setStatus('Selection cleared.');
  }
});
deleteSelectionButton.addEventListener('click', () => {
  deleteSelectedFaces();
});
smoothSelectionButton.addEventListener('click', () => {
  const result = viewport.smoothSelection(
    Number(selectionSmoothStrengthSlider.value),
    Number(selectionSmoothIterationsSlider.value),
  );
  setStatus(result.message, !result.success);
});
smoothBoundaryButton.addEventListener('click', () => {
  const result = viewport.smoothSelectionBoundary(
    Number(selectionSmoothStrengthSlider.value),
    Number(selectionSmoothIterationsSlider.value),
  );
  setStatus(result.message, !result.success);
});
remeshSelectionButton.addEventListener('click', () => {
  const result = viewport.remeshSelection(Number(selectionRemeshEdgeSlider.value));
  setStatus(result.message, !result.success);
});
refineSelectionButton.addEventListener('click', () => {
  const result = viewport.refineSelection();
  setStatus(result.message, !result.success);
});
socketModelPrevButton.addEventListener('click', () => {
  socketModelStepIndex = Math.max(0, socketModelStepIndex - 1);
  syncSocketModelStepUi(false);
});
socketModelNextButton.addEventListener('click', () => {
  advanceSocketModelStep();
});
positiveSocketPrevButton.addEventListener('click', () => {
  positiveSocketStepIndex = Math.max(0, positiveSocketStepIndex - 1);
  syncPositiveSocketStepUi(false);
});
positiveSocketNextButton.addEventListener('click', () => {
  advancePositiveSocketStep();
});

modeSelect.addEventListener('change', () => {
  const mode = modeSelect.value as InteractionMode;
  viewport.setInteractionMode(mode);
  syncModeUi();
  if (mode === 'fill') {
    const summary = viewport.getHoleLoopSummary();
    if (!summary) {
      setStatus('Load a mesh before using Fill Hole.');
    } else if (summary.loopCount === 0) {
      setStatus('Fill Hole mode active. No open or non-manifold edge loops were found.');
    } else {
      setStatus(`Fill Hole mode active. Highlighting ${formatCount(summary.loopCount)} loop groups in blue.`);
    }
  } else if (mode === 'boundary') {
    socketModelStepIndex = 0;
    syncSocketModelStepUi();
    const summary = viewport.getHoleLoopSummary();
    if (!summary) {
      setStatus('Load a mesh before using Socket Model.');
    } else if (summary.loopCount === 0) {
      setStatus('Socket Model is active. No open or non-manifold edge loops were found.');
    } else {
      setStatus(
        `Socket Model is active. Hover a loop, left click to target it, then step through the socket workflow.`,
      );
    }
  } else if (mode === 'positive') {
    positiveSocketStepIndex = 0;
    syncPositiveSocketStepUi(false);
    const summary = viewport.getHoleLoopSummary();
    if (!summary) {
      setStatus('Load a mesh before using Positive Socket.');
    } else if (summary.loopCount === 0) {
      setStatus('Positive Socket is active. No open or non-manifold edge loops were found.');
    } else {
      setStatus(
        'Positive Socket is active. Hover a loop, left click to target it, then step through the positive workflow.',
      );
    }
  }
});

brushSelect.addEventListener('change', () => {
  viewport.setBrushType(brushSelect.value as BrushType);
});

radiusSlider.addEventListener('input', () => {
  const value = Number(radiusSlider.value);
  radiusValue.textContent = formatMillimeters(value, 1);
  viewport.setBrushRadiusMm(value);
});

strengthSlider.addEventListener('input', () => {
  const value = Number(strengthSlider.value);
  strengthValue.textContent = value.toFixed(2);
  viewport.setBrushStrength(value);
});

selectionToolSelect.addEventListener('change', () => {
  viewport.setSelectionTool(selectionToolSelect.value as SelectionTool);
  syncSelectionUi();
});

selectionRadiusSlider.addEventListener('input', () => {
  const value = Number(selectionRadiusSlider.value);
  selectionRadiusValue.textContent = formatMillimeters(value, 1);
  viewport.setSelectionRadiusMm(value);
});

selectionSmoothStrengthSlider.addEventListener('input', () => {
  selectionSmoothStrengthValue.textContent = Number(selectionSmoothStrengthSlider.value).toFixed(2);
});

selectionSmoothIterationsSlider.addEventListener('input', () => {
  selectionSmoothIterationsValue.textContent = `${Math.max(
    1,
    Math.round(Number(selectionSmoothIterationsSlider.value)),
  )}`;
});

boundarySmoothStrengthSlider.addEventListener('input', () => {
  boundarySmoothStrengthValue.textContent = Number(boundarySmoothStrengthSlider.value).toFixed(2);
  previewBoundarySmoothFromUi();
});

positiveBoundarySmoothStrengthSlider.addEventListener('input', () => {
  positiveBoundarySmoothStrengthValue.textContent = Number(positiveBoundarySmoothStrengthSlider.value).toFixed(2);
  previewPositiveBoundarySmoothFromUi();
});

selectionRemeshEdgeSlider.addEventListener('input', () => {
  selectionRemeshEdgeValue.textContent = formatMillimeters(Number(selectionRemeshEdgeSlider.value), 3);
});

boundaryRemeshEdgeSlider.addEventListener('input', () => {
  boundaryRemeshEdgeValue.textContent = formatMillimeters(Number(boundaryRemeshEdgeSlider.value), 3);
});

positiveRemeshEdgeSlider.addEventListener('input', () => {
  positiveRemeshEdgeValue.textContent = formatMillimeters(Number(positiveRemeshEdgeSlider.value), 3);
});

boundaryThickenSlider.addEventListener('input', () => {
  boundaryThickenValue.textContent = formatMillimeters(Number(boundaryThickenSlider.value), 3);
  previewBoundaryThickenFromUi();
});

positiveExtrudeDistanceSlider.addEventListener('input', () => {
  positiveExtrudeDistanceValue.textContent = formatMillimeters(Number(positiveExtrudeDistanceSlider.value), 3);
  previewPositiveBoundaryExtrudeFromUi();
});

boundaryBandDistanceSlider.addEventListener('input', () => {
  boundaryBandDistanceValue.textContent = formatMillimeters(Number(boundaryBandDistanceSlider.value), 3);
  previewBoundaryOffsetFromUi();
});

positiveBandDistanceSlider.addEventListener('input', () => {
  positiveBandDistanceValue.textContent = formatMillimeters(Number(positiveBandDistanceSlider.value), 3);
  previewPositiveBoundaryOffsetFromUi();
});

boundarySelectedSmoothStrengthSlider.addEventListener('input', () => {
  boundarySelectedSmoothStrengthValue.textContent = Number(boundarySelectedSmoothStrengthSlider.value).toFixed(2);
  previewBoundaryFinalSmoothFromUi();
});

positiveSelectedSmoothStrengthSlider.addEventListener('input', () => {
  positiveSelectedSmoothStrengthValue.textContent = Number(positiveSelectedSmoothStrengthSlider.value).toFixed(2);
  previewPositiveBoundaryFinalSmoothFromUi();
});

positiveDirectionalTiltXSlider.addEventListener('input', () => {
  positiveDirectionalTiltXValue.textContent = `${Number(positiveDirectionalTiltXSlider.value).toFixed(1)}°`;
  previewPositiveDirectionalExtrudeFromUi();
});

positiveDirectionalTiltYSlider.addEventListener('input', () => {
  positiveDirectionalTiltYValue.textContent = `${Number(positiveDirectionalTiltYSlider.value).toFixed(1)}°`;
  previewPositiveDirectionalExtrudeFromUi();
});

wireframeToggle.addEventListener('change', () => {
  viewport.setWireframe(wireframeToggle.checked);
});

remeshEdgeSlider.addEventListener('input', () => {
  remeshEdgeValue.textContent = formatMillimeters(Number(remeshEdgeSlider.value), 3);
});

thickenSlider.addEventListener('input', () => {
  thickenValue.textContent = formatMillimeters(Number(thickenSlider.value), 3);
});

applyRemeshButton.addEventListener('click', () => {
  const result = viewport.applySurfaceRemesh(
    Number(remeshEdgeSlider.value),
    remeshBoundarySelect.value as 'fixed' | 'refined' | 'free',
  );
  setStatus(result.message, !result.success);
});

applyThickenButton.addEventListener('click', () => {
  const result = viewport.applyThicken(Number(thickenSlider.value));
  setStatus(result.message, !result.success);
});

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    return;
  }

  const importUnit = await promptImportUnit(file.name);
  if (!importUnit) {
    fileInput.value = '';
    setStatus('Import cancelled.');
    return;
  }

  setStatus(`Loading ${file.name} as ${importUnit}...`);
  openFileButton.disabled = true;
  fileInput.disabled = true;

  try {
    const loaded = await loadMeshFile(file, importUnit);
    currentFilename = loaded.filename;
    loaded.geometry.computeBoundsTree({
      maxLeafSize: 20,
      setBoundingBox: false,
      indirect: true,
    });

    const editable = createEditableMeshData(loaded.geometry);
    const engine = new SculptEngine(editable, 12);
    viewport.setSession(editable, engine);
    viewport.setBrushType(brushSelect.value as BrushType);
    viewport.setBrushRadiusMm(Number(radiusSlider.value));
    viewport.setBrushStrength(Number(strengthSlider.value));
    viewport.setInteractionMode(modeSelect.value as InteractionMode);
    viewport.setSelectionTool(selectionToolSelect.value as SelectionTool);
    viewport.setSelectionRadiusMm(Number(selectionRadiusSlider.value));
    viewport.setWireframe(wireframeToggle.checked);

    fileName.textContent = currentFilename;
    setStatus(
      `Loaded ${loaded.filename} (${loaded.extension.toUpperCase()}) using ${loaded.importUnit} -> mm with ${formatCount(
        loaded.triangleCount,
      )} triangles.`,
    );
  } catch (error) {
    console.error(error);
    const message = error instanceof Error ? error.message : 'Failed to load the selected mesh.';
    setStatus(message, true);
  } finally {
    openFileButton.disabled = false;
    fileInput.disabled = false;
    fileInput.value = '';
  }
});

window.addEventListener('keydown', (event) => {
  if (event.key !== 'Delete') {
    return;
  }

  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) {
    return;
  }

  deleteSelectedFaces();
});

window.addEventListener('beforeunload', () => viewport.dispose());

function syncModeUi(): void {
  const mode = modeSelect.value as InteractionMode;
  sculptControls.hidden = mode !== 'sculpt';
  selectionControls.hidden = mode !== 'select';
  fillControls.hidden = mode !== 'fill';
  boundaryControls.hidden = mode !== 'boundary';
  positiveControls.hidden = mode !== 'positive';
  remeshControls.hidden = mode !== 'remesh';
  thickenControls.hidden = mode !== 'thicken';
  historyControls.hidden = mode === 'boundary' || mode === 'positive';

  if (mode === 'fill') {
    modeHintPrimary.textContent = 'Bright blue lines show open or non-manifold edge loops.';
    modeHintSecondary.textContent =
      'Move near a loop to preview it in purple, then left click to patch a clean boundary loop.';
    modeHintTertiary.textContent = 'Middle drag orbits while Fill Hole mode stays active.';
    return;
  }

  if (mode === 'remesh') {
    modeHintPrimary.textContent =
      'Remesh rebuilds the current surface with split, collapse, flip, and reprojection passes.';
    modeHintSecondary.textContent =
      'Refined Boundary matches Meshmixer: boundary edges can split, but they do not collapse or smooth.';
    modeHintTertiary.textContent =
      'Fixed keeps the source border, Refined up-samples it cleanly, and Free lets the border regularize too.';
    return;
  }

  if (mode === 'thicken') {
    modeHintPrimary.textContent =
      'Thicken duplicates the surface, offsets it along normals, and bridges all open boundaries.';
    modeHintSecondary.textContent =
      'This is similar to Blender Solidify: the current mesh becomes a shell with explicit rim faces on boundary edges.';
    modeHintTertiary.textContent =
      'Set the thickness in millimeters, then apply it. Orbit remains on middle drag.';
    return;
  }

  if (mode === 'boundary') {
    modeHintPrimary.textContent =
      'Hover an open boundary loop until it turns purple, then left click to target that loop.';
    modeHintSecondary.textContent =
      'Socket Model uses one step at a time with Previous and Next instead of one long scrolling stack.';
    modeHintTertiary.textContent =
      'Boundary smooth uses a fixed 10-iteration preview, final smooth uses a fixed 40-iteration preview, and the post-offset remesh runs automatically at thickness / 8.';
    return;
  }

  if (mode === 'positive') {
    modeHintPrimary.textContent =
      'Hover an open boundary loop until it turns purple, then left click to target that loop.';
    modeHintSecondary.textContent =
      'Positive Socket uses the same clean loop targeting, then extrudes the remeshed boundary outward with a live distance slider.';
    modeHintTertiary.textContent =
      'The offset stage auto-remeshes at extrude distance / 8, the band smooth keeps the selected patch boundary fixed, and the final wall extrusion uses X/Y tilt sliders with a bbox-sized length.';
    return;
  }

  if (mode === 'sculpt') {
    modeHintPrimary.innerHTML = 'Left drag sculpts and middle drag orbits.';
    modeHintSecondary.textContent = 'Mouse wheel zooms and right drag pans through OrbitControls.';
    modeHintTertiary.textContent = 'Brush and selection sizes are in millimeters after import unit conversion.';
  } else {
    modeHintPrimary.innerHTML =
      'Sphere paints local face selection. Box and Snip drag a screen-space selection.';
    modeHintSecondary.textContent =
      'Middle drag orbits, Shift adds, Ctrl subtracts, and Delete removes the selected faces.';
    modeHintTertiary.textContent =
      'Smooth relaxes the selected patch, Smooth Boundary only relaxes the border, Refine subdivides once, and Remesh Selected rebuilds the patch with a fixed boundary.';
  }
}

function syncSelectionUi(): void {
  const tool = selectionToolSelect.value as SelectionTool;
  selectionRadiusField.hidden = tool !== 'sphere';
}

function updateHistoryButtons(history: HistoryState): void {
  undoButton.disabled = !history.canUndo;
  redoButton.disabled = !history.canRedo;
}

function updateSelectionUi(selection: SelectionState): void {
  selectedTriangleCount.textContent = formatCount(selection.selectedTriangleCount);
  clearSelectionButton.disabled = selection.selectedTriangleCount === 0;
  deleteSelectionButton.disabled = !selection.canDelete;
  smoothSelectionButton.disabled = selection.selectedTriangleCount === 0;
  smoothBoundaryButton.disabled = selection.selectedTriangleCount === 0;
  refineSelectionButton.disabled = selection.selectedTriangleCount === 0;
  remeshSelectionButton.disabled = selection.selectedTriangleCount === 0;
}

function updateBoundaryWorkflowUi(state: BoundaryWorkflowState): void {
  if (state.hasSelectedBoundary) {
    boundaryTargetStatus.textContent = `Targeted boundary: ${formatCount(state.selectedBoundaryEdgeCount)} loop vertices.`;
  } else if (state.thickenApplied && !state.offsetApplied) {
    boundaryTargetStatus.textContent = 'Boundary thicken complete. Move to Offset and apply the band plus the automatic remesh.';
  } else if (state.remeshApplied && !state.thickenApplied) {
    boundaryTargetStatus.textContent = 'Fixed-boundary remesh complete. Move to Thicken next.';
  } else if (state.smoothCommitted && !state.remeshApplied) {
    boundaryTargetStatus.textContent = 'Boundary smooth committed. Move to Remesh next.';
  } else if (state.offsetApplied) {
    boundaryTargetStatus.textContent = `Offset and auto-remesh complete. ${formatCount(state.selectedTriangleCount)} faces are ready for the final smooth.`;
  } else if (modeSelect.value === 'boundary') {
    boundaryTargetStatus.textContent = 'No boundary targeted yet.';
  } else {
    boundaryTargetStatus.textContent = 'Socket Model is inactive.';
  }

  if (state.hasSelectedBoundary) {
    positiveBoundaryTargetStatus.textContent = `Targeted boundary: ${formatCount(state.selectedBoundaryEdgeCount)} loop vertices.`;
  } else if (modeSelect.value === 'positive' && positiveSocketStepIndex >= POSITIVE_SOCKET_DIRECTIONAL_EXTRUDE_STEP_INDEX) {
    positiveBoundaryTargetStatus.textContent = 'Band smooth complete. Adjust the X/Y tilt sliders and finish the final wall extrusion.';
  } else if (state.extrudeApplied && !state.offsetApplied) {
    positiveBoundaryTargetStatus.textContent = 'Positive extrusion complete. Move to Offset and apply the band plus the automatic remesh.';
  } else if (state.remeshApplied && !state.extrudeApplied) {
    positiveBoundaryTargetStatus.textContent = 'Fixed-boundary remesh complete. Move to Extrude next.';
  } else if (state.smoothCommitted && !state.remeshApplied) {
    positiveBoundaryTargetStatus.textContent = 'Boundary smooth committed. Move to Remesh next.';
  } else if (state.offsetApplied) {
    positiveBoundaryTargetStatus.textContent = `Offset and auto-remesh complete. ${formatCount(state.selectedTriangleCount)} faces are ready for the final smooth.`;
  } else if (modeSelect.value === 'positive') {
    positiveBoundaryTargetStatus.textContent = 'No boundary targeted yet.';
  } else {
    positiveBoundaryTargetStatus.textContent = 'Positive Socket is inactive.';
  }

  // Boundary workflow updates are often emitted from the preview/commit methods
  // themselves. Re-entering the live preview here creates a feedback loop where
  // a commit triggers a preview, which mutates state again and can stall the wizard.
  syncSocketModelStepUi(false);
  syncPositiveSocketStepUi(false);
}

function updateMeshStats(stats: MeshStats): void {
  vertexCount.textContent = formatCount(stats.vertexCount);
  triangleCount.textContent = formatCount(stats.triangleCount);
  applyRemeshButton.disabled = stats.triangleCount === 0;
  applyThickenButton.disabled = stats.triangleCount === 0;
  if (stats.triangleCount === 0) {
    fileName.textContent = 'None loaded';
    currentFilename = 'None loaded';
  } else {
    fileName.textContent = currentFilename;
  }

  configureOperationSliders(stats.boundsRadius);
}

function deleteSelectedFaces(): void {
  const result = viewport.deleteSelection();
  if (!result) {
    return;
  }

  if (result.triangleCount === 0) {
    setStatus('Deleted the selected faces. The mesh is now empty.');
  } else {
    setStatus(`Deleted selected faces. ${formatCount(result.triangleCount)} triangles remain.`);
  }
}

function setStatus(message: string, isError = false): void {
  status.textContent = message;
  status.dataset.state = isError ? 'error' : 'idle';
}

function previewBoundarySmoothFromUi(): void {
  if (modeSelect.value !== 'boundary') {
    return;
  }

  const result = viewport.previewBoundarySmooth(
    Number(boundarySmoothStrengthSlider.value),
    10,
  );
  if (!result.success) {
    return;
  }

  setStatus(result.message);
}

function previewBoundaryThickenFromUi(): void {
  if (modeSelect.value !== 'boundary') {
    return;
  }

  const result = viewport.previewBoundaryThicken(Number(boundaryThickenSlider.value));
  if (!result.success) {
    return;
  }

  setStatus(result.message);
}

function previewBoundaryOffsetFromUi(): void {
  if (modeSelect.value !== 'boundary') {
    return;
  }

  const result = viewport.previewBoundaryBand(Number(boundaryBandDistanceSlider.value));
  if (!result.success) {
    return;
  }

  setStatus(`Previewing offset band at ${formatMillimeters(Number(boundaryBandDistanceSlider.value), 3)}.`);
}

function previewBoundaryFinalSmoothFromUi(): void {
  if (modeSelect.value !== 'boundary') {
    return;
  }

  const result = viewport.previewBoundaryFinalSmooth(Number(boundarySelectedSmoothStrengthSlider.value));
  if (!result.success) {
    return;
  }

  setStatus(result.message);
}

function previewPositiveBoundarySmoothFromUi(): void {
  if (modeSelect.value !== 'positive') {
    return;
  }

  const result = viewport.previewBoundarySmooth(
    Number(positiveBoundarySmoothStrengthSlider.value),
    10,
  );
  if (!result.success) {
    return;
  }

  setStatus(result.message);
}

function previewPositiveBoundaryExtrudeFromUi(): void {
  if (modeSelect.value !== 'positive') {
    return;
  }

  const result = viewport.previewBoundaryExtrude(Number(positiveExtrudeDistanceSlider.value));
  if (!result.success) {
    return;
  }

  setStatus(result.message);
}

function previewPositiveBoundaryOffsetFromUi(): void {
  if (modeSelect.value !== 'positive') {
    return;
  }

  const result = viewport.previewBoundaryBand(Number(positiveBandDistanceSlider.value));
  if (!result.success) {
    return;
  }

  setStatus(`Previewing offset band at ${formatMillimeters(Number(positiveBandDistanceSlider.value), 3)}.`);
}

function previewPositiveBoundaryFinalSmoothFromUi(): void {
  if (modeSelect.value !== 'positive') {
    return;
  }

  const result = viewport.previewBoundaryFinalSmooth(Number(positiveSelectedSmoothStrengthSlider.value));
  if (!result.success) {
    return;
  }

  setStatus(result.message);
}

function previewPositiveDirectionalExtrudeFromUi(): void {
  if (modeSelect.value !== 'positive') {
    return;
  }

  const result = viewport.previewBoundaryDirectionalExtrude(
    Number(positiveDirectionalTiltXSlider.value),
    Number(positiveDirectionalTiltYSlider.value),
  );
  if (!result.success) {
    return;
  }

  setStatus(result.message);
}

function syncSocketModelStepUi(_shouldRefreshPreview = true): void {
  for (let i = 0; i < SOCKET_MODEL_STEP_PANELS.length; i += 1) {
    SOCKET_MODEL_STEP_PANELS[i].hidden = i !== socketModelStepIndex;
  }

  socketModelStepLabel.textContent = `Step ${socketModelStepIndex + 1} of ${SOCKET_MODEL_STEP_PANELS.length}`;
  socketModelStepTitle.textContent = SOCKET_MODEL_STEP_TITLES[socketModelStepIndex];
  socketModelPrevButton.disabled = socketModelStepIndex === SOCKET_MODEL_TARGET_STEP_INDEX;
  socketModelNextButton.textContent =
    socketModelStepIndex === SOCKET_MODEL_COMPLETE_STEP_INDEX
      ? 'Start Over'
      : socketModelStepIndex === SOCKET_MODEL_FINAL_SMOOTH_STEP_INDEX
        ? 'Finish'
        : 'Next';

  const state = viewport.getBoundaryWorkflowState();
  if (modeSelect.value !== 'boundary') {
    socketModelNextButton.disabled = false;
    return;
  }

  socketModelNextButton.disabled =
    socketModelStepIndex === SOCKET_MODEL_TARGET_STEP_INDEX && !state.hasSelectedBoundary;
}

function advanceSocketModelStep(): void {
  if (modeSelect.value !== 'boundary') {
    return;
  }

  const state = viewport.getBoundaryWorkflowState();
  let result:
    | {
        success: boolean;
        message: string;
      }
    | null = null;

  switch (socketModelStepIndex) {
    case SOCKET_MODEL_TARGET_STEP_INDEX:
      if (!state.hasSelectedBoundary) {
        setStatus('Target a boundary loop before continuing to Boundary Smooth.', true);
        return;
      }

      socketModelStepIndex = SOCKET_MODEL_BOUNDARY_SMOOTH_STEP_INDEX;
      syncSocketModelStepUi(false);
      return;
    case SOCKET_MODEL_BOUNDARY_SMOOTH_STEP_INDEX:
      result = viewport.commitBoundarySmooth();
      break;
    case SOCKET_MODEL_REMESH_STEP_INDEX:
      result = viewport.applyBoundaryFixedRemesh(Number(boundaryRemeshEdgeSlider.value));
      break;
    case SOCKET_MODEL_THICKEN_STEP_INDEX:
      result = viewport.commitBoundaryThicken(Number(boundaryThickenSlider.value));
      break;
    case SOCKET_MODEL_OFFSET_STEP_INDEX: {
      const offsetResult = viewport.selectBoundaryBand(Number(boundaryBandDistanceSlider.value));
      if (!offsetResult.success) {
        setStatus(offsetResult.message, true);
        return;
      }

      const autoEdgeSize = Math.max(Number(boundaryBandDistanceSlider.value) / 5, 0.05);
      const remeshResult = viewport.remeshSelection(autoEdgeSize);
      if (!remeshResult.success) {
        setStatus(remeshResult.message, true);
        return;
      }

      setStatus(
        `Offset applied and the selected band was auto-remeshed at ${formatMillimeters(autoEdgeSize, 3)}.`,
      );
      socketModelStepIndex = SOCKET_MODEL_FINAL_SMOOTH_STEP_INDEX;
      syncSocketModelStepUi(false);
      return;
    }
    case SOCKET_MODEL_FINAL_SMOOTH_STEP_INDEX:
      result = viewport.commitBoundaryFinalSmooth(Number(boundarySelectedSmoothStrengthSlider.value));
      break;
    case SOCKET_MODEL_COMPLETE_STEP_INDEX:
      socketModelStepIndex = SOCKET_MODEL_TARGET_STEP_INDEX;
      syncSocketModelStepUi(false);
      setStatus('Socket Model reset. Hover a loop and target a new boundary to begin again.');
      return;
    default:
      return;
  }

  if (!result.success) {
    setStatus(result.message, true);
    return;
  }

  setStatus(result.message);
  if (socketModelStepIndex < SOCKET_MODEL_COMPLETE_STEP_INDEX) {
    socketModelStepIndex += 1;
    syncSocketModelStepUi(false);
    if (socketModelStepIndex === SOCKET_MODEL_COMPLETE_STEP_INDEX) {
      setStatus('Socket Model finished. Press Next to start over on a new boundary, or Previous to inspect the earlier steps.');
    }
    return;
  }
}

function syncPositiveSocketStepUi(_shouldRefreshPreview = true): void {
  for (let i = 0; i < POSITIVE_SOCKET_STEP_PANELS.length; i += 1) {
    POSITIVE_SOCKET_STEP_PANELS[i].hidden = i !== positiveSocketStepIndex;
  }

  positiveSocketStepLabel.textContent = `Step ${positiveSocketStepIndex + 1} of ${POSITIVE_SOCKET_STEP_PANELS.length}`;
  positiveSocketStepTitle.textContent = POSITIVE_SOCKET_STEP_TITLES[positiveSocketStepIndex];
  positiveSocketPrevButton.disabled = positiveSocketStepIndex === POSITIVE_SOCKET_TARGET_STEP_INDEX;
  positiveSocketNextButton.textContent =
    positiveSocketStepIndex === POSITIVE_SOCKET_COMPLETE_STEP_INDEX
      ? 'Start Over'
      : positiveSocketStepIndex === POSITIVE_SOCKET_DIRECTIONAL_EXTRUDE_STEP_INDEX
        ? 'Finish'
        : 'Next';

  const state = viewport.getBoundaryWorkflowState();
  if (modeSelect.value !== 'positive') {
    positiveSocketNextButton.disabled = false;
    return;
  }

  positiveSocketNextButton.disabled =
    positiveSocketStepIndex === POSITIVE_SOCKET_TARGET_STEP_INDEX && !state.hasSelectedBoundary;
}

function advancePositiveSocketStep(): void {
  if (modeSelect.value !== 'positive') {
    return;
  }

  const state = viewport.getBoundaryWorkflowState();
  let result:
    | {
        success: boolean;
        message: string;
      }
    | null = null;

  switch (positiveSocketStepIndex) {
    case POSITIVE_SOCKET_TARGET_STEP_INDEX:
      if (!state.hasSelectedBoundary) {
        setStatus('Target a boundary loop before continuing to Boundary Smooth.', true);
        return;
      }

      positiveSocketStepIndex = POSITIVE_SOCKET_BOUNDARY_SMOOTH_STEP_INDEX;
      syncPositiveSocketStepUi(false);
      return;
    case POSITIVE_SOCKET_BOUNDARY_SMOOTH_STEP_INDEX:
      result = viewport.commitBoundarySmooth();
      break;
    case POSITIVE_SOCKET_REMESH_STEP_INDEX:
      result = viewport.applyBoundaryFixedRemesh(Number(positiveRemeshEdgeSlider.value));
      break;
    case POSITIVE_SOCKET_EXTRUDE_STEP_INDEX:
      result = viewport.commitBoundaryExtrude(Number(positiveExtrudeDistanceSlider.value));
      break;
    case POSITIVE_SOCKET_OFFSET_STEP_INDEX: {
      const offsetResult = viewport.selectBoundaryBand(Number(positiveBandDistanceSlider.value));
      if (!offsetResult.success) {
        setStatus(offsetResult.message, true);
        return;
      }

      const autoEdgeSize = Math.max(Number(positiveExtrudeDistanceSlider.value) / 8, 0.05);
      const remeshResult = viewport.remeshSelection(autoEdgeSize);
      if (!remeshResult.success) {
        setStatus(remeshResult.message, true);
        return;
      }

      setStatus(
        `Offset applied and the selected band was auto-remeshed at ${formatMillimeters(autoEdgeSize, 3)}.`,
      );
      positiveSocketStepIndex = POSITIVE_SOCKET_FINAL_SMOOTH_STEP_INDEX;
      syncPositiveSocketStepUi(false);
      return;
    }
    case POSITIVE_SOCKET_FINAL_SMOOTH_STEP_INDEX:
      result = viewport.commitBoundaryFinalSmooth(Number(positiveSelectedSmoothStrengthSlider.value));
      break;
    case POSITIVE_SOCKET_DIRECTIONAL_EXTRUDE_STEP_INDEX:
      result = viewport.commitBoundaryDirectionalExtrude(
        Number(positiveDirectionalTiltXSlider.value),
        Number(positiveDirectionalTiltYSlider.value),
      );
      break;
    case POSITIVE_SOCKET_COMPLETE_STEP_INDEX:
      positiveSocketStepIndex = POSITIVE_SOCKET_TARGET_STEP_INDEX;
      syncPositiveSocketStepUi(false);
      setStatus('Positive Socket reset. Hover a loop and target a new boundary to begin again.');
      return;
    default:
      return;
  }

  if (!result.success) {
    setStatus(result.message, true);
    return;
  }

  setStatus(result.message);
  if (positiveSocketStepIndex < POSITIVE_SOCKET_COMPLETE_STEP_INDEX) {
    positiveSocketStepIndex += 1;
    syncPositiveSocketStepUi(false);
    if (positiveSocketStepIndex === POSITIVE_SOCKET_COMPLETE_STEP_INDEX) {
      setStatus('Positive Socket finished. Press Next to start over on a new boundary, or Previous to inspect the earlier steps.');
    }
  }
}

function promptImportUnit(filename: string): Promise<MeshUnit | null> {
  importUnitTitle.textContent = `Choose the source unit for ${filename}`;
  importUnitModal.hidden = false;

  return new Promise((resolve) => {
    const complete = (unit: MeshUnit | null) => {
      importUnitModal.hidden = true;
      importUnitMillimetersButton.removeEventListener('click', handleMillimeters);
      importUnitCentimetersButton.removeEventListener('click', handleCentimeters);
      importUnitMetersButton.removeEventListener('click', handleMeters);
      importUnitCancelButton.removeEventListener('click', handleCancel);
      importUnitModal.removeEventListener('click', handleBackdropClick);
      window.removeEventListener('keydown', handleKeyDown);
      resolve(unit);
    };

    const handleMillimeters = () => complete('mm');
    const handleCentimeters = () => complete('cm');
    const handleMeters = () => complete('m');
    const handleCancel = () => complete(null);
    const handleBackdropClick = (event: MouseEvent) => {
      if (event.target === importUnitModal) {
        complete(null);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        complete(null);
      }
    };

    importUnitMillimetersButton.addEventListener('click', handleMillimeters);
    importUnitCentimetersButton.addEventListener('click', handleCentimeters);
    importUnitMetersButton.addEventListener('click', handleMeters);
    importUnitCancelButton.addEventListener('click', handleCancel);
    importUnitModal.addEventListener('click', handleBackdropClick);
    window.addEventListener('keydown', handleKeyDown);
  });
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Expected element #${id}`);
  }

  return element as T;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

function formatMillimeters(value: number, decimals: number): string {
  return `${value.toFixed(decimals)} mm`;
}

function configureOperationSliders(boundsRadius: number): void {
  const effectiveRadius = Math.max(boundsRadius, 1);

  configureAbsoluteSlider(
    radiusSlider,
    radiusValue,
    0.1,
    Math.max(effectiveRadius * 0.35, 80),
    5,
    1,
  );

  configureAbsoluteSlider(
    selectionRadiusSlider,
    selectionRadiusValue,
    0.1,
    Math.max(effectiveRadius * 0.5, 120),
    6,
    1,
  );

  configureAbsoluteSlider(
    remeshEdgeSlider,
    remeshEdgeValue,
    0.05,
    Math.max(effectiveRadius * 0.18, 40),
    0.25,
    3,
  );

  configureAbsoluteSlider(
    boundaryRemeshEdgeSlider,
    boundaryRemeshEdgeValue,
    0.05,
    Math.max(effectiveRadius * 0.18, 40),
    0.25,
    3,
  );

  configureAbsoluteSlider(
    positiveRemeshEdgeSlider,
    positiveRemeshEdgeValue,
    0.05,
    Math.max(effectiveRadius * 0.18, 40),
    0.25,
    3,
  );

  configureAbsoluteSlider(
    boundaryThickenSlider,
    boundaryThickenValue,
    0.05,
    Math.max(effectiveRadius * 0.12, 30),
    0.25,
    3,
  );

  configureAbsoluteSlider(
    selectionRemeshEdgeSlider,
    selectionRemeshEdgeValue,
    0.05,
    Math.max(effectiveRadius * 0.18, 40),
    0.25,
    3,
  );

  configureAbsoluteSlider(
    boundaryBandDistanceSlider,
    boundaryBandDistanceValue,
    0.1,
    Math.max(effectiveRadius * 0.24, 30),
    1.5,
    3,
  );

  configureAbsoluteSlider(
    positiveExtrudeDistanceSlider,
    positiveExtrudeDistanceValue,
    0.05,
    Math.max(effectiveRadius * 0.12, 30),
    0.25,
    3,
  );

  configureAbsoluteSlider(
    positiveBandDistanceSlider,
    positiveBandDistanceValue,
    0.1,
    Math.max(effectiveRadius * 0.24, 30),
    1.5,
    3,
  );

  configureAbsoluteSlider(
    thickenSlider,
    thickenValue,
    0.05,
    Math.max(effectiveRadius * 0.12, 30),
    0.25,
    3,
  );
}

function configureAbsoluteSlider(
  slider: HTMLInputElement,
  label: HTMLElement,
  min: number,
  max: number,
  fallbackValue: number,
  decimals = 3,
): void {
  const clampedMax = Math.max(max, min + 0.001);
  slider.min = min.toFixed(4);
  slider.max = clampedMax.toFixed(4);
  slider.step = Math.max((clampedMax - min) / 240, 0.0005).toFixed(4);

  const currentValue = Number(slider.value);
  const nextValue =
    Number.isFinite(currentValue) && currentValue >= min && currentValue <= clampedMax
      ? currentValue
      : Math.min(Math.max(fallbackValue, min), clampedMax);

  slider.value = nextValue.toFixed(4);
  label.textContent = formatMillimeters(nextValue, decimals);
}
