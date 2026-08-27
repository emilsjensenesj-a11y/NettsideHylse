import './styles.css';
import './three-bvh';

import { createEditableMeshData } from './sculpt/editable-mesh';
import { loadMeshFile } from './io/mesh-loader';
import {
  ViewportController,
  type MeshExportUnit,
  type MeshRotationAxis,
  type MeshViewMode,
} from './render/viewport';
import { SculptEngine } from './sculpt/sculpt-engine';
import type {
  BoundaryWorkflowState,
  HistoryState,
  InteractionMode,
  MeasurementState,
  MeshStats,
  SelectionState,
  SelectionTool,
} from './sculpt/types';

interface DesktopExportFilePayload {
  filename: string;
  mimeType: string;
  data: ArrayBuffer;
}

interface DesktopExportResult {
  canceled: boolean;
  filePaths?: string[];
}

interface NouraDesktopApi {
  saveExportedFiles?: (request: { files: DesktopExportFilePayload[] }) => Promise<DesktopExportResult>;
}

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) {
  throw new Error('App root not found.');
}

app.innerHTML = `
  <div class="app-shell">
    <aside class="toolbar">
      <div class="panel-header">
        <img
          id="brand-logo"
          class="brand-logo"
          src="./nourasoft-logo.png"
          alt="NouraSoft"
          data-fallbacks="./nourasoft-logo.jpg,./nourasoft-logo.jpeg,./nourasoft-logo.webp"
        />
      </div>

      <p id="file-name" class="file-caption">No file loaded</p>

      <div class="workflow-steps" aria-label="Clinical workflow">
        <button class="workflow-step" type="button" data-mode="select">
          <span class="workflow-number">1</span>
          <span class="workflow-text">Select</span>
        </button>
        <button class="workflow-step" type="button" data-mode="fill">
          <span class="workflow-number">2</span>
          <span class="workflow-text">Fill Hole</span>
        </button>
        <button class="workflow-step" type="button" data-mode="sculpt">
          <span class="workflow-number">3</span>
          <span class="workflow-text">Smooth Brush</span>
        </button>
        <button class="workflow-step" type="button" data-mode="positive">
          <span class="workflow-number">4</span>
          <span class="workflow-text">Positive Limb</span>
        </button>
      </div>

      <select id="mode-select" hidden>
        <option value="select" selected>Select</option>
        <option value="fill">Fill Hole</option>
        <option value="sculpt">Smooth Brush</option>
        <option value="positive">Positive Limb</option>
      </select>


      <div id="sculpt-controls" class="control-group" hidden>
        <label class="field range-field">
          <span>Brush Radius</span>
          <div class="range-input-row">
            <input id="radius-slider" type="range" min="0" max="100" step="1" value="13" />
            <label class="percent-input">
              <input id="radius-input" class="number-input" type="number" min="0" max="100" step="1" value="13" />
              <span>%</span>
            </label>
          </div>
        </label>

        <label class="field range-field">
          <span>Strength</span>
          <div class="range-input-row">
            <input id="strength-slider" type="range" min="0" max="100" step="1" value="35" />
            <label class="percent-input">
              <input id="strength-input" class="number-input" type="number" min="0" max="100" step="1" value="35" />
              <span>%</span>
            </label>
          </div>
        </label>
        <button id="trimline-only-button" class="secondary-button" type="button" aria-pressed="false">Only Trimline</button>
      </div>

      <div id="selection-controls" class="control-group">
        <label class="field">
          <span>Selection Area</span>
          <select id="selection-tool-select">
            <option value="sphere">Sphere</option>
            <option value="box">Box</option>
            <option value="snip">Snip / Lasso</option>
          </select>
        </label>

        <label class="checkbox-field">
          <span>Select Through</span>
          <input id="select-visible-toggle" type="checkbox" />
        </label>

        <label id="selection-radius-field" class="field range-field">
          <span>Selection Radius</span>
          <div class="range-input-row">
            <input id="selection-radius-slider" type="range" min="0" max="100" step="1" value="15" />
            <label class="percent-input">
              <input id="selection-radius-input" class="number-input" type="number" min="0" max="100" step="1" value="15" />
              <span>%</span>
            </label>
          </div>
        </label>

        <div class="control-group split">
          <button id="clear-selection-button" class="secondary-button" disabled>Clear</button>
          <button id="delete-selection-button" class="secondary-button" disabled>Delete</button>
        </div>

        <div class="archived-controls" hidden>
          <button id="smooth-selection-button" class="secondary-button" disabled>Smooth Selection</button>

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

          <button id="refine-selection-button" class="secondary-button" disabled>Refine</button>
          <button id="smooth-boundary-button" class="secondary-button" disabled>Smooth Boundary</button>
          <button id="remesh-selection-button" class="secondary-button" disabled>Remesh Selected</button>
        </div>
      </div>

      <div id="fill-controls" class="control-group" hidden>
        <p class="inline-note">
          Hover a bright-blue boundary loop until it turns purple, then left click to patch it.
        </p>
      </div>

      <div id="positive-controls" class="control-group" hidden>
        <p id="positive-boundary-target-status" class="inline-note">Select an open edge to build the Positive Limb.</p>
        <div class="stepper-header" hidden>
          <div>
            <p class="mini-heading">Positive Limb</p>
            <p id="positive-socket-step-label" class="inline-note">Step 1 of 9</p>
          </div>
          <div class="stepper-actions">
            <button id="positive-socket-prev-button" class="secondary-button">Previous</button>
            <button id="positive-socket-next-button" class="secondary-button">Next</button>
          </div>
        </div>
        <p id="positive-socket-step-title" class="inline-note" hidden>Full Mesh Remesh</p>

        <div id="positive-step-full-remesh" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Full Mesh Remesh</p>
          <label class="field range-field">
            <span>Target Edge Size <strong id="positive-full-remesh-edge-value">0.250 mm</strong></span>
            <input id="positive-full-remesh-edge-slider" type="range" min="0.05" max="1" step="0.01" value="0.25" />
          </label>
          <p class="inline-note">Run a full scan remesh before selecting the boundary for the positive limb.</p>
        </div>

        <div id="positive-step-target" class="socket-step-panel control-group" hidden>
          <p class="inline-note">
            Hover a bright-blue boundary loop until it turns purple, then left click to target it for the positive limb workflow.
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
          <p class="inline-note">This is the only slow step in Positive Limb, so it does not preview live. Press Next to run it.</p>
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
            <span>X Tilt <strong id="positive-directional-tilt-x-value">0.0Â°</strong></span>
            <input id="positive-directional-tilt-x-slider" type="range" min="-45" max="45" step="1" value="0" />
          </label>
          <label class="field range-field">
            <span>Y Tilt <strong id="positive-directional-tilt-y-value">0.0Â°</strong></span>
            <input id="positive-directional-tilt-y-slider" type="range" min="-45" max="45" step="1" value="0" />
          </label>
          <p class="inline-note">
            This final wall extrusion uses the current outer boundary, the unified boundary normals, and a plane test to pick the outward side. The wall length matches the largest edge of the mesh bounding box.
          </p>
        </div>

        <div id="positive-step-complete" class="socket-step-panel control-group" hidden>
          <p class="mini-heading">Complete</p>
          <p class="inline-note">
            The current positive limb pass is finished. Press Next to start over on a new boundary, or Previous to inspect the earlier stages.
          </p>
        </div>
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

    </aside>

    <main class="viewport-panel">      <div class="viewport-menu-bar">
        <div class="viewport-menu-group">
          <div class="option-button-wrap">
            <button id="import-button" class="primary-button menu-bar-button" type="button" aria-expanded="false" aria-controls="import-menu">Import</button>
            <div id="import-menu" class="option-menu" hidden>
              <button id="import-files-option" class="menu-button" type="button">Open STL / OBJ</button>
              <button id="import-folder-option" class="menu-button" type="button">Open OBJ Folder</button>
            </div>
          </div>
          <input id="file-input" type="file" accept=".stl,.obj,.mtl,.png,.jpg,.jpeg,.webp,.bmp" multiple hidden />
          <button id="export-button" class="secondary-button menu-bar-button" type="button" disabled>Export</button>
          <div class="option-button-wrap">
            <button id="view-button" class="secondary-button menu-bar-button" type="button" aria-expanded="false" aria-controls="view-menu">View</button>
            <div id="view-menu" class="option-menu" hidden>
              <button id="view-colored-option" class="menu-button" type="button" data-view-mode="colored">Colored</button>
              <button id="view-shaded-option" class="menu-button" type="button" data-view-mode="shaded">Shaded</button>
              <button id="view-wireframe-option" class="menu-button" type="button" data-view-mode="wireframe">Wireframe</button>
            </div>
          </div>
          <div class="option-button-wrap">
            <button id="rotate-button" class="secondary-button menu-bar-button" type="button" aria-expanded="false" aria-controls="rotate-menu" disabled>Rotate</button>
            <div id="rotate-menu" class="option-menu rotate-menu" hidden>
              <label class="menu-field rotation-axis-field">
                <span>X</span>
                <div class="angle-input-wrap">
                  <input id="rotate-x-input" type="number" step="0.001" value="0.000" />
                  <span>deg</span>
                </div>
              </label>
              <label class="menu-field rotation-axis-field">
                <span>Y</span>
                <div class="angle-input-wrap">
                  <input id="rotate-y-input" type="number" step="0.001" value="0.000" />
                  <span>deg</span>
                </div>
              </label>
              <label class="menu-field rotation-axis-field">
                <span>Z</span>
                <div class="angle-input-wrap">
                  <input id="rotate-z-input" type="number" step="0.001" value="0.000" />
                  <span>deg</span>
                </div>
              </label>
            </div>
          </div>
        </div>
        <div id="history-controls" class="viewport-menu-group">
          <button id="dark-theme-button" class="secondary-button menu-bar-button theme-button" type="button" aria-pressed="false">Dark Theme</button>
          <button id="undo-button" class="secondary-button menu-bar-button" type="button" disabled>Undo</button>
          <button id="redo-button" class="secondary-button menu-bar-button" type="button" disabled>Redo</button>
        </div>
      </div>      <div id="viewport" class="viewport-host">
        <div id="viewport-hints" class="viewport-hints" data-collapsed="false">
          <button id="toggle-controls-hint" class="controls-drawer-toggle" type="button" aria-label="Hide controls">
            <span class="drawer-label">Controls</span>
            <span id="controls-drawer-arrows" class="drawer-arrows">&lt;&lt;&lt;</span>
          </button>
          <div class="viewport-hints-header">
            <strong>Controls</strong>
          </div>
          <p id="mode-hint-primary">Left drag smooths and right drag rotates.</p>
          <p id="mode-hint-secondary">Mouse wheel zooms and middle drag pans.</p>
          <p id="mode-hint-tertiary">In Select mode, <kbd>Shift</kbd> adds and <kbd>Ctrl</kbd> subtracts.</p>
        </div>
        <div class="viewcube-panel" aria-label="ViewCube orientation controls">
          <button id="reset-view-cube" class="viewcube-home" type="button" aria-label="Home view">
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M3 10.8 12 3l9 7.8" />
              <path d="M5.5 9.5V21h13V9.5" />
              <path d="M9.5 21v-6h5v6" />
            </svg>
          </button>
          <div class="viewcube-stage">
            <div id="view-cube" class="viewcube">
              <button id="view-front" class="viewcube-face viewcube-face-front" data-direction="0,-1,0" type="button">Front</button>
              <button id="view-back" class="viewcube-face viewcube-face-back" data-direction="0,1,0" type="button">Back</button>
              <button id="view-left" class="viewcube-face viewcube-face-left" data-direction="1,0,0" type="button">Left</button>
              <button id="view-right" class="viewcube-face viewcube-face-right" data-direction="-1,0,0" type="button">Right</button>
              <button id="view-proximal" class="viewcube-face viewcube-face-top" data-direction="0,0,1" type="button">Proximal</button>
              <button id="view-distal" class="viewcube-face viewcube-face-bottom" data-direction="0,0,-1" type="button">Distal</button>
            </div>
          </div>
        </div>
        <aside id="measurement-panel" class="measurement-panel">
          <button id="toggle-measurement-panel" class="measurement-drawer-toggle" type="button" aria-label="Hide measurements">
            <span class="drawer-label">Measurements</span>
            <span id="measurement-drawer-arrows" class="drawer-arrows">&gt;&gt;&gt;</span>
          </button>
          <div class="measurement-header">
            <div>
              <p class="mini-heading">Measurements</p>
              <p class="inline-note">25 mm spacing along the Z proximal axis</p>
            </div>
          </div>
          <div class="measurement-actions">
            <button id="toggle-circumference-button" class="secondary-button" type="button" disabled>Show Measurements</button>
            <button id="set-circumference-start-button" class="secondary-button" type="button" disabled>Set Starting Circumference</button>
            <button id="take-measurement-button" class="secondary-button" type="button" disabled>Measure Height</button>
            <button id="clear-measurement-button" class="secondary-button" type="button" disabled>Point-to-Point</button>
          </div>
          <div class="measurement-summary">
            <span>Total height</span>
            <strong id="measurement-total-height">--</strong>
            <span>Height at click</span>
            <strong id="measurement-click-height">Click scan</strong>
          </div>
          <div class="measurement-table-wrap">
            <table class="measurement-table">
              <thead>
                <tr>
                  <th>Height</th>
                  <th>Circ.</th>
                </tr>
              </thead>
              <tbody id="measurement-table-body">
                <tr>
                  <td colspan="2">Load a scan.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </aside>
        <div id="positive-progress" class="positive-progress" hidden>
          <div class="positive-progress-box">
            <p class="mini-heading">Positive Limb</p>
            <p id="positive-progress-message">Preparing mesh.</p>
          </div>
        </div>
        <div id="operation-error-toast" class="operation-error-toast" role="alert" hidden>
          <button id="operation-error-close" class="operation-error-close" type="button" aria-label="Dismiss error">x</button>
          <p class="mini-heading">Action Failed</p>
          <p id="operation-error-message">The action could not be completed.</p>
        </div>
        <p id="status" class="status">Open a local STL or OBJ mesh to begin smoothing.</p>
      </div>
    </main>
  </div>
  <dialog id="export-dialog" class="export-dialog" aria-labelledby="export-dialog-title">
    <form id="export-form" class="export-form">
      <div class="export-dialog-heading">
        <p class="mini-heading">Mesh Export</p>
        <h2 id="export-dialog-title">Choose format and units</h2>
        <p>The imported geometry is always treated as millimetres. Exported coordinates are converted to the selected unit.</p>
      </div>
      <label class="export-field">
        <span>File format</span>
        <select id="export-format-select">
          <option value="obj">OBJ + MTL + PNG</option>
          <option value="stl">STL</option>
        </select>
      </label>
      <label class="export-field">
        <span>Coordinate unit</span>
        <select id="export-unit-select">
          <option value="mm">Millimetres (mm)</option>
          <option value="cm">Centimetres (cm)</option>
          <option value="m">Metres (m)</option>
          <option value="in">Inches (in)</option>
        </select>
      </label>
      <p class="export-unit-note">OBJ and STL are unitless formats. NouraSoft changes the coordinate values so one exported unit equals the unit selected above.</p>
      <div class="export-dialog-actions">
        <button id="export-cancel-button" class="secondary-button" type="button">Cancel</button>
        <button class="primary-button" type="submit">Export</button>
      </div>
    </form>
  </dialog>
`;

const importButton = requireElement<HTMLButtonElement>('import-button');
const brandLogo = requireElement<HTMLImageElement>('brand-logo');
const importMenu = requireElement<HTMLElement>('import-menu');
const importFilesOption = requireElement<HTMLButtonElement>('import-files-option');
const importFolderOption = requireElement<HTMLButtonElement>('import-folder-option');
const exportButton = requireElement<HTMLButtonElement>('export-button');
const exportDialog = requireElement<HTMLDialogElement>('export-dialog');
const exportForm = requireElement<HTMLFormElement>('export-form');
const exportFormatSelect = requireElement<HTMLSelectElement>('export-format-select');
const exportUnitSelect = requireElement<HTMLSelectElement>('export-unit-select');
const exportCancelButton = requireElement<HTMLButtonElement>('export-cancel-button');
const viewButton = requireElement<HTMLButtonElement>('view-button');
const viewMenu = requireElement<HTMLElement>('view-menu');
const viewModeButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-view-mode]'));
const rotateButton = requireElement<HTMLButtonElement>('rotate-button');
const rotateMenu = requireElement<HTMLElement>('rotate-menu');
const rotateXInput = requireElement<HTMLInputElement>('rotate-x-input');
const rotateYInput = requireElement<HTMLInputElement>('rotate-y-input');
const rotateZInput = requireElement<HTMLInputElement>('rotate-z-input');
const rotateInputs = {
  x: rotateXInput,
  y: rotateYInput,
  z: rotateZInput,
} satisfies Record<MeshRotationAxis, HTMLInputElement>;
const fileInput = requireElement<HTMLInputElement>('file-input');
const modeSelect = requireElement<HTMLSelectElement>('mode-select');
const workflowStepButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('.workflow-step'));
const radiusSlider = requireElement<HTMLInputElement>('radius-slider');
const radiusInput = requireElement<HTMLInputElement>('radius-input');
const strengthSlider = requireElement<HTMLInputElement>('strength-slider');
const strengthInput = requireElement<HTMLInputElement>('strength-input');
const trimlineOnlyButton = requireElement<HTMLButtonElement>('trimline-only-button');
const selectionToolSelect = requireElement<HTMLSelectElement>('selection-tool-select');
const selectVisibleToggle = requireElement<HTMLInputElement>('select-visible-toggle');
const selectionRadiusField = requireElement<HTMLElement>('selection-radius-field');
const selectionRadiusSlider = requireElement<HTMLInputElement>('selection-radius-slider');
const selectionRadiusInput = requireElement<HTMLInputElement>('selection-radius-input');
const selectionSmoothStrengthSlider = requireElement<HTMLInputElement>('selection-smooth-strength-slider');
const selectionSmoothStrengthValue = requireElement<HTMLElement>('selection-smooth-strength-value');
const selectionSmoothIterationsSlider = requireElement<HTMLInputElement>('selection-smooth-iterations-slider');
const selectionSmoothIterationsValue = requireElement<HTMLElement>('selection-smooth-iterations-value');
const selectionRemeshEdgeSlider = requireElement<HTMLInputElement>('selection-remesh-edge-slider');
const selectionRemeshEdgeValue = requireElement<HTMLElement>('selection-remesh-edge-value');
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
const positiveStepFullRemesh = requireElement<HTMLElement>('positive-step-full-remesh');
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
const positiveFullRemeshEdgeSlider = requireElement<HTMLInputElement>('positive-full-remesh-edge-slider');
const positiveFullRemeshEdgeValue = requireElement<HTMLElement>('positive-full-remesh-edge-value');
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
const historyControls = requireElement<HTMLElement>('history-controls');
const darkThemeButton = requireElement<HTMLButtonElement>('dark-theme-button');
const undoButton = requireElement<HTMLButtonElement>('undo-button');
const redoButton = requireElement<HTMLButtonElement>('redo-button');
const sculptControls = requireElement<HTMLElement>('sculpt-controls');
const selectionControls = requireElement<HTMLElement>('selection-controls');
const fileName = requireElement<HTMLElement>('file-name');
const status = requireElement<HTMLElement>('status');
const positiveProgress = requireElement<HTMLElement>('positive-progress');
const positiveProgressMessage = requireElement<HTMLElement>('positive-progress-message');
const operationErrorToast = requireElement<HTMLElement>('operation-error-toast');
const operationErrorClose = requireElement<HTMLButtonElement>('operation-error-close');
const operationErrorMessage = requireElement<HTMLElement>('operation-error-message');
const viewportHints = requireElement<HTMLElement>('viewport-hints');
const toggleControlsHintButton = requireElement<HTMLButtonElement>('toggle-controls-hint');
const controlsDrawerArrows = requireElement<HTMLElement>('controls-drawer-arrows');
const modeHintPrimary = requireElement<HTMLElement>('mode-hint-primary');
const modeHintSecondary = requireElement<HTMLElement>('mode-hint-secondary');
const modeHintTertiary = requireElement<HTMLElement>('mode-hint-tertiary');
const viewportHost = requireElement<HTMLElement>('viewport');
const measurementPanel = requireElement<HTMLElement>('measurement-panel');
const toggleMeasurementPanelButton = requireElement<HTMLButtonElement>('toggle-measurement-panel');
const measurementDrawerArrows = requireElement<HTMLElement>('measurement-drawer-arrows');
const toggleCircumferenceButton = requireElement<HTMLButtonElement>('toggle-circumference-button');
const setCircumferenceStartButton = requireElement<HTMLButtonElement>('set-circumference-start-button');
const takeMeasurementButton = requireElement<HTMLButtonElement>('take-measurement-button');
const clearMeasurementButton = requireElement<HTMLButtonElement>('clear-measurement-button');
const measurementTotalHeight = requireElement<HTMLElement>('measurement-total-height');
const measurementClickHeight = requireElement<HTMLElement>('measurement-click-height');
const measurementTableBody = requireElement<HTMLTableSectionElement>('measurement-table-body');
const viewCube = requireElement<HTMLElement>('view-cube');
const viewCubeStage = viewCube.closest<HTMLElement>('.viewcube-stage');
if (!viewCubeStage) {
  throw new Error('Expected ViewCube stage.');
}
const resetViewCubeButton = requireElement<HTMLButtonElement>('reset-view-cube');
const uiTooltip = document.createElement('div');
uiTooltip.id = 'ui-tooltip';
uiTooltip.className = 'ui-tooltip';
uiTooltip.hidden = true;
document.body.append(uiTooltip);

let currentFilename = 'No file loaded';
let currentTextureFile: File | null = null;
let meshLoaded = false;
let meshViewMode: MeshViewMode = 'colored';
let rotateToolActive = false;
let circumferenceOverlayVisible = false;
let measurementPickActive = false;
let pointToPointPickActive = false;
let measurementStartPickActive = false;
let measurementHoveredIndex: number | null = null;
let trimlineOnlySmooth = false;
const BRUSH_RADIUS_MAX_MM = 40;
const DEFAULT_BRUSH_RADIUS_PERCENT = 13;
const DEFAULT_BRUSH_STRENGTH_PERCENT = 35;
const DARK_THEME_STORAGE_KEY = 'nourasoft-dark-theme';
const EXPORT_FORMAT_STORAGE_KEY = 'nourasoft-export-format';
const EXPORT_UNIT_STORAGE_KEY = 'nourasoft-export-unit';
type MeshExportFormat = 'obj' | 'stl';
let darkThemeEnabled = readDarkThemePreference();
let lastExportFormat = readExportFormatPreference();
let lastExportUnit = readExportUnitPreference();
const SOCKET_MODEL_TARGET_STEP_INDEX = 0;
const SOCKET_MODEL_BOUNDARY_SMOOTH_STEP_INDEX = 1;

applyActionTooltips();
attachTooltipEvents();
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
const POSITIVE_SOCKET_FULL_REMESH_STEP_INDEX = 0;
const POSITIVE_SOCKET_TARGET_STEP_INDEX = 1;
const POSITIVE_SOCKET_BOUNDARY_SMOOTH_STEP_INDEX = 2;
const POSITIVE_SOCKET_REMESH_STEP_INDEX = 3;
const POSITIVE_SOCKET_EXTRUDE_STEP_INDEX = 4;
const POSITIVE_SOCKET_OFFSET_STEP_INDEX = 5;
const POSITIVE_SOCKET_FINAL_SMOOTH_STEP_INDEX = 6;
const POSITIVE_SOCKET_DIRECTIONAL_EXTRUDE_STEP_INDEX = 7;
const POSITIVE_SOCKET_COMPLETE_STEP_INDEX = 8;
const POSITIVE_SOCKET_STEP_TITLES = [
  'Full Mesh Remesh',
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
  positiveStepFullRemesh,
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

function applyActionTooltips(): void {
  const tooltipsById: Record<string, string> = {
    'import-button': 'Open scan import options.',
    'import-files-option': 'Choose an STL or OBJ scan. For textured OBJ files, select the OBJ, MTL, and image together.',
    'import-folder-option': 'Load an OBJ and its texture files from the same folder when browser support allows it.',
    'export-button': 'Choose units and export the current mesh as STL or textured OBJ.',
    'view-button': 'Switch how the scan is displayed in the viewport.',
    'view-colored-option': 'Show the scan color or baked vertex color when available.',
    'view-shaded-option': 'Show a neutral shaded surface for checking shape.',
    'view-wireframe-option': 'Show triangle edges for technical inspection.',
    'rotate-button': 'Rotate the mesh with live angle values or the on-model rotation rings.',
    'rotate-x-input': 'Rotate around the X axis in degrees. Updates live.',
    'rotate-y-input': 'Rotate around the Y axis in degrees. Updates live.',
    'rotate-z-input': 'Rotate around the Z axis in degrees. Updates live.',
    'dark-theme-button': 'Toggle the premium NOURA dark theme on or off.',
    'undo-button': 'Undo the last mesh edit.',
    'redo-button': 'Redo the last undone mesh edit.',
    'radius-slider': 'Adjust the Smooth Brush radius. 100% equals 40 mm.',
    'radius-input': 'Type the Smooth Brush radius as a percent of the 40 mm maximum.',
    'strength-slider': 'Adjust how strongly the Smooth Brush relaxes the surface.',
    'strength-input': 'Type Smooth Brush strength as a percent.',
    'trimline-only-button': 'Limits Smooth Brush strokes to open trimline boundary vertices.',
    'selection-tool-select': 'Choose the selection shape used to mark scan artifacts.',
    'select-visible-toggle': 'Turn on to select through the scan. Leave off to select only the closest visible surface.',
    'selection-radius-slider': 'Adjust the selection radius. 100% equals 40 mm.',
    'selection-radius-input': 'Type the selection radius as a percent of the 40 mm maximum.',
    'clear-selection-button': 'Clear the current selected faces without changing the mesh.',
    'delete-selection-button': 'Delete the selected artifact faces from the scan.',
    'smooth-selection-button': 'Archived tool: smooth only the selected faces.',
    'selection-smooth-strength-slider': 'Archived setting: controls how aggressively selected faces are smoothed.',
    'selection-smooth-iterations-slider': 'Archived setting: number of repeated smoothing passes.',
    'selection-remesh-edge-slider': 'Archived setting: target triangle size for remeshing selected faces.',
    'refine-selection-button': 'Archived tool: subdivide selected faces for denser editing.',
    'smooth-boundary-button': 'Archived tool: smooth the boundary around a selection.',
    'remesh-selection-button': 'Archived tool: rebuild selected faces with cleaner triangles.',
    'positive-boundary-target-status': 'Positive Limb starts by selecting the open boundary edge of the scan.',
    'positive-socket-prev-button': 'Go back to the previous archived Positive Limb step.',
    'positive-socket-next-button': 'Continue the archived Positive Limb step sequence.',
    'positive-full-remesh-edge-slider': 'Archived setting: full-scan remesh voxel size.',
    'positive-boundary-smooth-strength-slider': 'Archived setting: smooths the selected open edge before building the limb.',
    'positive-remesh-edge-slider': 'Archived setting: fixed-boundary remesh size.',
    'positive-extrude-distance-slider': 'Archived setting: extrusion distance from the boundary.',
    'positive-band-distance-slider': 'Archived setting: width of the offset band near the boundary.',
    'positive-selected-smooth-strength-slider': 'Archived setting: relaxes the final selected band.',
    'positive-directional-tilt-x-slider': 'Archived setting: tilt the final wall extrusion around X.',
    'positive-directional-tilt-y-slider': 'Archived setting: tilt the final wall extrusion around Y.',
    'boundary-target-status': 'Archived Socket Model status for the selected boundary.',
    'socket-model-prev-button': 'Go back to the previous archived Socket Model step.',
    'socket-model-next-button': 'Continue the archived Socket Model step sequence.',
    'boundary-smooth-strength-slider': 'Archived setting: smooths the selected boundary.',
    'boundary-remesh-edge-slider': 'Archived setting: fixed-boundary remesh size.',
    'boundary-thicken-slider': 'Archived setting: socket shell thickness.',
    'boundary-band-distance-slider': 'Archived setting: width of the offset band.',
    'boundary-selected-smooth-strength-slider': 'Archived setting: relaxes the selected band.',
    'remesh-edge-slider': 'Archived setting: target triangle size for whole-mesh remeshing.',
    'remesh-boundary-select': 'Archived setting: controls how remesh treats open boundaries.',
    'apply-remesh-button': 'Archived tool: rebuild the mesh triangle structure.',
    'thicken-slider': 'Archived setting: shell offset distance.',
    'apply-thicken-button': 'Archived tool: create shell thickness.',
    'reset-view-cube': 'Return the viewport to the default home view.',
    'view-front': 'Look at the front face of the scan.',
    'view-back': 'Look at the back face of the scan.',
    'view-left': 'Look at the left side of the scan.',
    'view-right': 'Look at the right side of the scan.',
    'view-proximal': 'Look from the proximal +Z direction.',
    'view-distal': 'Look from the distal -Z direction.',
    'toggle-controls-hint': 'Collapse or reopen the viewport controls guide.',
    'toggle-measurement-panel': 'Collapse or reopen the Measurements panel.',
    'toggle-circumference-button': 'Calculate and show circumference bands every 25 mm along +Z.',
    'set-circumference-start-button': 'Pick where the first circumference band starts. The bands update live while hovering.',
    'take-measurement-button': 'Preview and click to pin a vertical height measurement from the distal end to the point.',
    'clear-measurement-button': 'Click two surface points to measure the direct distance between them.',
  };
  for (const [id, text] of Object.entries(tooltipsById)) {
    setTooltipById(id, text);
  }

  const workflowTooltips: Partial<Record<InteractionMode, string>> = {
    select: 'Select artifact faces so they can be cleared or deleted.',
    fill: 'Patch open scan holes by clicking highlighted boundary loops.',
    sculpt: 'Use the Smooth Brush to relax rough areas on the scan.',
    positive: 'Select the open edge to build the Positive Limb automatically.',
    boundary: 'Archived paid workflow for socket model generation.',
    remesh: 'Archived tool for rebuilding mesh triangle layout.',
    thicken: 'Archived tool for adding shell thickness.',
  };
  workflowStepButtons.forEach((button) => {
    const mode = button.dataset.mode as InteractionMode | undefined;
    const tooltip = mode ? workflowTooltips[mode] : undefined;
    if (tooltip) {
      setTooltip(button, tooltip);
    }
  });

  [
    'radius-slider',
    'strength-slider',
    'selection-tool-select',
    'select-visible-toggle',
    'selection-radius-slider',
    'selection-smooth-strength-slider',
    'selection-smooth-iterations-slider',
    'selection-remesh-edge-slider',
    'rotate-x-input',
    'rotate-y-input',
    'rotate-z-input',
    'positive-full-remesh-edge-slider',
    'positive-boundary-smooth-strength-slider',
    'positive-remesh-edge-slider',
    'positive-extrude-distance-slider',
    'positive-band-distance-slider',
    'positive-selected-smooth-strength-slider',
    'positive-directional-tilt-x-slider',
    'positive-directional-tilt-y-slider',
    'boundary-smooth-strength-slider',
    'boundary-remesh-edge-slider',
    'boundary-thicken-slider',
    'boundary-band-distance-slider',
    'boundary-selected-smooth-strength-slider',
    'remesh-edge-slider',
    'remesh-boundary-select',
    'thicken-slider',
  ].forEach((id) => {
    const text = tooltipsById[id];
    if (text) {
      setTooltipForClosestLabel(id, text);
    }
  });

  document
    .querySelectorAll<HTMLElement>(
      '#fill-controls .inline-note, #positive-controls .inline-note, #boundary-controls .inline-note',
    )
    .forEach((element) => {
      if (!element.dataset.tooltip) {
        setTooltip(element, 'Instruction text for the current workflow step.');
      }
    });
}

function setTooltipById(id: string, text: string): void {
  const element = document.getElementById(id);
  if (element instanceof HTMLElement) {
    setTooltip(element, text);
  }
}

function setTooltipForClosestLabel(id: string, text: string): void {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLElement)) {
    return;
  }

  setTooltip(element, text);
  const label = element.closest<HTMLElement>('label');
  if (label) {
    setTooltip(label, text);
  }
}

function setTooltip(element: HTMLElement, text: string): void {
  element.dataset.tooltip = text;
}

const TOOLTIP_HOVER_DELAY_MS = 1000;
let activeTooltipTarget: HTMLElement | null = null;
let tooltipHoverTimer: number | null = null;
let pendingTooltipX = 0;
let pendingTooltipY = 0;

function attachTooltipEvents(): void {
  document.addEventListener(
    'pointerover',
    (event) => {
      const target = findTooltipTarget(event.target);
      if (!target) {
        return;
      }

      scheduleTooltip(target, event.clientX, event.clientY);
    },
    true,
  );
  document.addEventListener(
    'pointermove',
    (event) => {
      if (!activeTooltipTarget) {
        return;
      }

      pendingTooltipX = event.clientX;
      pendingTooltipY = event.clientY;
      if (!uiTooltip.hidden) {
        positionTooltip(event.clientX, event.clientY);
      }
    },
    true,
  );
  document.addEventListener(
    'pointerout',
    (event) => {
      if (!activeTooltipTarget) {
        return;
      }

      if (event.relatedTarget instanceof Node && activeTooltipTarget.contains(event.relatedTarget)) {
        return;
      }

      hideTooltip();
    },
    true,
  );
  document.addEventListener(
    'focusin',
    (event) => {
      const target = findTooltipTarget(event.target);
      if (!target) {
        return;
      }

      activeTooltipTarget = target;
      const rect = target.getBoundingClientRect();
      showTooltip(target, rect.left + rect.width * 0.5, rect.bottom);
    },
    true,
  );
  document.addEventListener('focusout', hideTooltip, true);
}

function findTooltipTarget(target: EventTarget | null): HTMLElement | null {
  if (!(target instanceof Element)) {
    return null;
  }

  return target.closest<HTMLElement>('[data-tooltip]');
}

function scheduleTooltip(target: HTMLElement, clientX: number, clientY: number): void {
  pendingTooltipX = clientX;
  pendingTooltipY = clientY;
  if (activeTooltipTarget === target) {
    return;
  }

  cancelTooltipTimer();
  activeTooltipTarget = target;
  uiTooltip.hidden = true;
  tooltipHoverTimer = window.setTimeout(() => {
    tooltipHoverTimer = null;
    if (activeTooltipTarget === target) {
      showTooltip(target, pendingTooltipX, pendingTooltipY);
    }
  }, TOOLTIP_HOVER_DELAY_MS);
}

function showTooltip(target: HTMLElement, clientX: number, clientY: number): void {
  const text = target.dataset.tooltip;
  if (!text) {
    hideTooltip();
    return;
  }

  uiTooltip.textContent = text;
  uiTooltip.hidden = false;
  positionTooltip(clientX, clientY);
}

function positionTooltip(clientX: number, clientY: number): void {
  const margin = 10;
  const offset = 14;
  const rect = uiTooltip.getBoundingClientRect();
  let x = clientX + offset;
  let y = clientY + offset;

  if (x + rect.width + margin > window.innerWidth) {
    x = clientX - rect.width - offset;
  }
  if (y + rect.height + margin > window.innerHeight) {
    y = clientY - rect.height - offset;
  }

  uiTooltip.style.left = `${Math.max(margin, x)}px`;
  uiTooltip.style.top = `${Math.max(margin, y)}px`;
}

function cancelTooltipTimer(): void {
  if (tooltipHoverTimer === null) {
    return;
  }

  window.clearTimeout(tooltipHoverTimer);
  tooltipHoverTimer = null;
}

function hideTooltip(): void {
  cancelTooltipTimer();
  activeTooltipTarget = null;
  uiTooltip.hidden = true;
}

const viewport = new ViewportController(viewportHost, {
  onHistoryChange: updateHistoryButtons,
  onSelectionChange: updateSelectionUi,
  onBoundaryWorkflowChange: updateBoundaryWorkflowUi,
  onBoundaryAction: ({ success, message, complete }) => {
    setStatus(message, !success);
    if (!success && modeSelect.value === 'positive') {
      showOperationError('Positive Limb', message);
    } else if (success && modeSelect.value === 'positive') {
      hideOperationError();
    }
    if (success && complete && modeSelect.value === 'positive') {
      positiveSocketStepIndex = POSITIVE_SOCKET_COMPLETE_STEP_INDEX;
      positiveBoundaryTargetStatus.textContent = 'Positive Limb complete.';
      syncPositiveSocketStepUi(false);
    }
  },
  onPositiveLimbProgress: ({ visible, message }) => {
    positiveProgress.hidden = !visible;
    positiveProgressMessage.textContent = message;
  },
  onMeshStatsChange: updateMeshStats,
  onHoleFill: ({ success, message }) => {
    setStatus(message, !success);
    if (!success) {
      showOperationError('Fill Hole', message);
    } else {
      hideOperationError();
    }
  },
  onViewCubeTransform: (transform) => {
    viewCube.style.transform = transform;
  },
  onMeasurementChange: updateMeasurementUi,
  onMeasurementCaptured: (heightMm) => {
    setStatus(`Measured ${formatMillimeters(heightMm, 1)} from the clicked point down to the distal end.`);
  },
  onPointToPointMeasurementCaptured: (distanceMm) => {
    setStatus(`Measured ${formatMillimeters(distanceMm, 1)} between the two clicked points.`);
  },
  onPointToPointPickStateChange: (active) => {
    pointToPointPickActive = active;
    syncMeasurementControls();
  },
  onMeasurementStartCaptured: (heightMm) => {
    setStatus(`Starting circumference set at ${formatMillimeters(heightMm, 1)} height.`);
  },
  onMeasurementPickStateChange: (active) => {
    measurementPickActive = active;
    syncMeasurementControls();
  },
  onMeasurementStartPickStateChange: (active) => {
    measurementStartPickActive = active;
    syncMeasurementControls();
  },
  onMeasurementHoverChange: (rowIndex) => {
    measurementHoveredIndex = rowIndex;
    syncMeasurementHoverRows();
  },
  onMeasurementVisibilityChange: (visible) => {
    circumferenceOverlayVisible = visible;
    syncMeasurementControls();
    updateMeasurementUi(viewport.getMeasurementState());
  },
  onRotationDraftChange: syncRotateInputs,
  onMeshViewModeChange: (mode) => {
    meshViewMode = mode;
    syncViewMenuUi();
  },
});

setDarkTheme(darkThemeEnabled, false);
viewport.setBrushType('smooth');
viewport.setBrushRadiusMm(radiusPercentToMillimeters(Number(radiusSlider.value)));
viewport.setBrushStrength(percentToUnit(Number(strengthSlider.value)));
viewport.setSmoothOnlyTrimline(trimlineOnlySmooth);
viewport.setInteractionMode(modeSelect.value as InteractionMode);
viewport.setSelectionTool(selectionToolSelect.value as SelectionTool);
viewport.setSelectOnlyVisible(!selectVisibleToggle.checked);
viewport.setSelectionRadiusMm(radiusPercentToMillimeters(Number(selectionRadiusSlider.value)));
viewport.setMeshViewMode(meshViewMode);
viewport.setMeasurementCircumferenceVisible(circumferenceOverlayVisible);
syncModeUi();
syncSelectionUi();
syncViewMenuUi();
syncMeasurementControls();
configureOperationSliders(0.5);
updateBoundaryWorkflowUi(viewport.getBoundaryWorkflowState());
syncSocketModelStepUi();
syncPositiveSocketStepUi();

brandLogo.addEventListener('error', () => {
  const fallbacks = brandLogo.dataset.fallbacks
    ?.split(',')
    .map((path) => path.trim())
    .filter(Boolean) ?? [];
  const [next, ...remaining] = fallbacks;
  if (!next) {
    brandLogo.dataset.missing = 'true';
    return;
  }

  brandLogo.dataset.fallbacks = remaining.join(',');
  brandLogo.src = next;
});

importButton.addEventListener('click', (event) => {
  event.stopPropagation();
  toggleOptionMenu(importMenu, importButton);
});
importFilesOption.addEventListener('click', () => {
  resetMeasurementActivity();
  closeOptionMenus();
  fileInput.click();
});
importFolderOption.addEventListener('click', async () => {
  resetMeasurementActivity();
  closeOptionMenus();
  await openObjFolder();
});
exportButton.addEventListener('click', () => {
  if (!meshLoaded) {
    return;
  }

  resetMeasurementActivity();
  closeOptionMenus();
  exportFormatSelect.value = lastExportFormat;
  exportUnitSelect.value = lastExportUnit;
  exportDialog.showModal();
});
viewButton.addEventListener('click', (event) => {
  event.stopPropagation();
  toggleOptionMenu(viewMenu, viewButton);
});
viewModeButtons.forEach((button) => {
  button.addEventListener('click', () => {
    resetMeasurementActivity();
    const nextMode = button.dataset.viewMode as MeshViewMode | undefined;
    if (!nextMode) {
      return;
    }

    meshViewMode = nextMode;
    viewport.setMeshViewMode(meshViewMode);
    syncViewMenuUi();
    closeOptionMenus();
    setStatus(`View set to ${formatViewMode(meshViewMode)}.`);
  });
});
rotateButton.addEventListener('click', (event) => {
  event.stopPropagation();
  if (!meshLoaded) {
    return;
  }

  const shouldOpen = rotateMenu.hidden === true;
  if (shouldOpen) {
    resetMeasurementActivity();
  }
  closeOptionMenus();
  rotateMenu.hidden = !shouldOpen;
  rotateButton.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
  setRotateToolActive(shouldOpen);
});
Object.values(rotateInputs).forEach((input) => {
  input.addEventListener('input', () => {
    updateRotationDraftFromInputs(false);
  });
  input.addEventListener('change', () => {
    updateRotationDraftFromInputs(true);
  });
});
darkThemeButton.addEventListener('click', () => {
  setDarkTheme(!darkThemeEnabled);
});
operationErrorClose.addEventListener('click', () => {
  hideOperationError();
});
exportCancelButton.addEventListener('click', () => {
  exportDialog.close();
});
exportDialog.addEventListener('click', (event) => {
  if (event.target === exportDialog) {
    exportDialog.close();
  }
});
exportForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const format = isMeshExportFormat(exportFormatSelect.value) ? exportFormatSelect.value : 'obj';
  const unit = isMeshExportUnit(exportUnitSelect.value) ? exportUnitSelect.value : 'mm';
  lastExportFormat = format;
  lastExportUnit = unit;
  persistExportPreferences();
  exportDialog.close();
  void exportMesh(format, unit);
});

async function exportMesh(format: MeshExportFormat, unit: MeshExportUnit): Promise<void> {
  resetMeasurementActivity();
  viewport.commitRotationDraft();
  closeOptionMenus();
  const exportBaseName = getNouraDesktopApi()?.saveExportedFiles ? getExportBaseName() : promptForExportBaseName();
  if (!exportBaseName) {
    setStatus(`${format.toUpperCase()} export canceled.`);
    return;
  }

  try {
    const exported =
      format === 'stl'
        ? viewport.exportStl(exportBaseName, unit)
        : viewport.exportObj(exportBaseName, unit);
    if (!exported) {
      setStatus(`Load a mesh before exporting ${format.toUpperCase()}.`, true);
      return;
    }

    const filesToSave = 'files' in exported ? exported.files : [exported];
    const saved = await saveExportedBlobFiles(filesToSave);
    if (!saved) {
      setStatus(`${format.toUpperCase()} export canceled.`);
      return;
    }

    const unitLabel = formatMeshExportUnit(unit);
    setStatus(
      format === 'obj'
        ? `Exported OBJ, MTL, and PNG in ${unitLabel}; the OBJ contains no vertex colors.`
        : `Exported STL coordinate values in ${unitLabel}.`,
    );
  } catch (error) {
    setStatus(getExportErrorMessage(error, `Failed to export ${format.toUpperCase()}.`), true);
  }
}
toggleCircumferenceButton.addEventListener('click', () => {
  setRotateToolActive(false);
  if (circumferenceOverlayVisible) {
    circumferenceOverlayVisible = false;
    viewport.setMeasurementCircumferenceVisible(false);
    syncMeasurementControls();
    updateMeasurementUi(viewport.getMeasurementState());
    setStatus('Measurements hidden.');
    return;
  }

  if (measurementStartPickActive) {
    viewport.cancelMeasurementStartPick();
  }
  if (pointToPointPickActive) {
    viewport.cancelPointToPointMeasurementPick();
  }
  circumferenceOverlayVisible = true;
  viewport.setMeasurementCircumferenceVisible(true);
  syncMeasurementControls();
  updateMeasurementUi(viewport.getMeasurementState());
  setStatus('Calculated circumference bands every 25 mm.');
});
setCircumferenceStartButton.addEventListener('click', () => {
  setRotateToolActive(false);
  if (measurementStartPickActive) {
    viewport.cancelMeasurementStartPick();
    setStatus('Starting circumference selection canceled.');
    return;
  }

  if (measurementPickActive) {
    viewport.cancelMeasurementPick();
  }
  if (pointToPointPickActive) {
    viewport.cancelPointToPointMeasurementPick();
  }

  if (!viewport.beginMeasurementStartPick()) {
    setStatus('Load a scan before setting the starting circumference.', true);
    return;
  }

  setStatus('Move over the scan to preview Z-axis circumference bands, then click to set the start.');
});
takeMeasurementButton.addEventListener('click', () => {
  setRotateToolActive(false);
  if (measurementPickActive) {
    viewport.cancelMeasurementPick();
    setStatus('Height measurement canceled.');
    return;
  }

  if (measurementStartPickActive) {
    viewport.cancelMeasurementStartPick();
  }
  if (pointToPointPickActive) {
    viewport.cancelPointToPointMeasurementPick();
  }

  if (!viewport.beginMeasurementPick()) {
    setStatus('Load a scan before taking a height measurement.', true);
    return;
  }

  circumferenceOverlayVisible = true;
  viewport.setMeasurementCircumferenceVisible(true);
  syncMeasurementControls();
  updateMeasurementUi(viewport.getMeasurementState());
  setStatus('Click the scan once to measure straight down to the distal end.');
});
clearMeasurementButton.addEventListener('click', () => {
  setRotateToolActive(false);
  if (pointToPointPickActive) {
    viewport.cancelPointToPointMeasurementPick();
    setStatus('Point-to-point measurement canceled.');
    return;
  }

  if (measurementPickActive) {
    viewport.cancelMeasurementPick();
  }
  if (measurementStartPickActive) {
    viewport.cancelMeasurementStartPick();
  }

  if (!viewport.beginPointToPointMeasurementPick()) {
    setStatus('Load a scan before taking a point-to-point measurement.', true);
    return;
  }

  circumferenceOverlayVisible = true;
  viewport.setMeasurementCircumferenceVisible(true);
  syncMeasurementControls();
  updateMeasurementUi(viewport.getMeasurementState());
  setStatus('Click the first point on the scan, then click the second point to measure direct length.');
});
toggleControlsHintButton.addEventListener('click', () => {
  const nextCollapsed = viewportHints.dataset.collapsed !== 'true';
  viewportHints.dataset.collapsed = nextCollapsed ? 'true' : 'false';
  controlsDrawerArrows.textContent = nextCollapsed ? '>>>' : '<<<';
  toggleControlsHintButton.setAttribute(
    'aria-label',
    nextCollapsed ? 'Show controls' : 'Hide controls',
  );
});
toggleMeasurementPanelButton.addEventListener('click', () => {
  const nextCollapsed = measurementPanel.dataset.collapsed !== 'true';
  measurementPanel.dataset.collapsed = nextCollapsed ? 'true' : 'false';
  measurementDrawerArrows.textContent = nextCollapsed ? '<<<' : '>>>';
  toggleMeasurementPanelButton.setAttribute(
    'aria-label',
    nextCollapsed ? 'Show measurements' : 'Hide measurements',
  );
});
undoButton.addEventListener('click', () => {
  resetMeasurementActivity();
  viewport.undo();
  if (modeSelect.value === 'positive') {
    positiveSocketStepIndex = POSITIVE_SOCKET_FULL_REMESH_STEP_INDEX;
    const summary = viewport.resetPositiveLimbPrompt();
    syncPositiveSocketStepUi(false);
    positiveBoundaryTargetStatus.textContent = 'Select an open edge to build the Positive Limb.';
    if (!summary) {
      setStatus('Positive Limb is active. Select an open edge to build it automatically.');
    } else if (summary.loopCount === 0) {
      setStatus('Positive Limb is active. No open edge loops were found.', true);
    } else {
      setStatus('Positive Limb is active. Select an open edge to build it automatically.');
    }
  }
});
redoButton.addEventListener('click', () => {
  resetMeasurementActivity();
  viewport.redo();
});
clearSelectionButton.addEventListener('click', () => {
  resetMeasurementActivity();
  if (viewport.clearSelection()) {
    setStatus('Selection cleared.');
  }
});
deleteSelectionButton.addEventListener('click', () => {
  resetMeasurementActivity();
  deleteSelectedFaces();
});
smoothSelectionButton.addEventListener('click', () => {
  resetMeasurementActivity();
  const result = viewport.smoothSelection(
    Number(selectionSmoothStrengthSlider.value),
    Number(selectionSmoothIterationsSlider.value),
  );
  setStatus(result.message, !result.success);
});
smoothBoundaryButton.addEventListener('click', () => {
  resetMeasurementActivity();
  const result = viewport.smoothSelectionBoundary(
    Number(selectionSmoothStrengthSlider.value),
    Number(selectionSmoothIterationsSlider.value),
  );
  setStatus(result.message, !result.success);
});
remeshSelectionButton.addEventListener('click', () => {
  resetMeasurementActivity();
  const result = viewport.remeshSelection(Number(selectionRemeshEdgeSlider.value));
  setStatus(result.message, !result.success);
});
refineSelectionButton.addEventListener('click', () => {
  resetMeasurementActivity();
  const result = viewport.refineSelection();
  setStatus(result.message, !result.success);
});
socketModelPrevButton.addEventListener('click', () => {
  resetMeasurementActivity();
  socketModelStepIndex = Math.max(0, socketModelStepIndex - 1);
  syncSocketModelStepUi(false);
});
socketModelNextButton.addEventListener('click', () => {
  resetMeasurementActivity();
  advanceSocketModelStep();
});
positiveSocketPrevButton.addEventListener('click', () => {
  resetMeasurementActivity();
  positiveSocketStepIndex = Math.max(0, positiveSocketStepIndex - 1);
  syncPositiveSocketStepUi(false);
});
positiveSocketNextButton.addEventListener('click', () => {
  resetMeasurementActivity();
  advancePositiveSocketStep();
});
resetViewCubeButton.addEventListener('click', () => {
  viewport.resetView();
});

workflowStepButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const mode = button.dataset.mode as InteractionMode | undefined;
    if (!mode || modeSelect.value === mode) {
      return;
    }

    viewport.clearSelection();
    modeSelect.value = mode;
    modeSelect.dispatchEvent(new Event('change', { bubbles: true }));
  });
});

let viewCubeDragActive = false;
let viewCubeDidDrag = false;
let viewCubeSuppressClick = false;
let viewCubePointerId = -1;
let viewCubeStartX = 0;
let viewCubeStartY = 0;
let viewCubeLastX = 0;
let viewCubeLastY = 0;
let viewCubePendingTarget: HTMLElement | null = null;

viewCubeStage.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) {
    return;
  }

  viewCubeDragActive = true;
  viewCubeDidDrag = false;
  viewCubePointerId = event.pointerId;
  viewCubeStartX = event.clientX;
  viewCubeStartY = event.clientY;
  viewCubeLastX = event.clientX;
  viewCubeLastY = event.clientY;
  viewCubePendingTarget =
    event.target instanceof HTMLElement
      ? event.target.closest<HTMLElement>('.viewcube-face')
      : null;
  viewCubeStage.setPointerCapture(event.pointerId);
  viewCubeStage.classList.add('is-dragging');
  event.preventDefault();
});

viewCubeStage.addEventListener('pointermove', (event) => {
  if (!viewCubeDragActive || event.pointerId !== viewCubePointerId) {
    return;
  }

  const totalDistance = Math.hypot(event.clientX - viewCubeStartX, event.clientY - viewCubeStartY);
  if (totalDistance > 3) {
    viewCubeDidDrag = true;
  }

  if (viewCubeDidDrag) {
    viewport.orbitFromViewCube(event.clientX - viewCubeLastX, event.clientY - viewCubeLastY);
  }

  viewCubeLastX = event.clientX;
  viewCubeLastY = event.clientY;
  event.preventDefault();
});

viewCubeStage.addEventListener('pointerup', (event) => {
  if (!viewCubeDragActive || event.pointerId !== viewCubePointerId) {
    return;
  }

  viewCubeDragActive = false;
  viewCubePointerId = -1;
  viewCubeStage.classList.remove('is-dragging');
  if (viewCubeStage.hasPointerCapture(event.pointerId)) {
    viewCubeStage.releasePointerCapture(event.pointerId);
  }

  if (viewCubeDidDrag) {
    viewCubeSuppressClick = true;
    window.setTimeout(() => {
      viewCubeSuppressClick = false;
    }, 0);
  } else {
    const elementAtPointer = document.elementFromPoint(event.clientX, event.clientY);
    const target =
      viewCubePendingTarget ??
      (elementAtPointer instanceof HTMLElement
        ? elementAtPointer.closest<HTMLElement>('.viewcube-face')
        : null);
    activateViewCubeTarget(target);
    viewCubeSuppressClick = true;
    window.setTimeout(() => {
      viewCubeSuppressClick = false;
    }, 0);
  }

  viewCubePendingTarget = null;
});

viewCubeStage.addEventListener('pointercancel', () => {
  viewCubeDragActive = false;
  viewCubePointerId = -1;
  viewCubePendingTarget = null;
  viewCubeStage.classList.remove('is-dragging');
});

viewCubeStage.addEventListener(
  'click',
  (event) => {
    if (!viewCubeSuppressClick) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
  },
  true,
);

function activateViewCubeTarget(target: HTMLElement | null): void {
  const direction = target?.dataset.direction;
  if (!direction) {
    return;
  }

  const [x, y, z] = direction.split(',').map(Number);
  if ([x, y, z].every(Number.isFinite)) {
    viewport.setOrientationVector(x, y, z);
  }
}

modeSelect.addEventListener('change', () => {
  resetMeasurementActivity();
  setRotateToolActive(false);
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
    positiveSocketStepIndex = POSITIVE_SOCKET_FULL_REMESH_STEP_INDEX;
    syncPositiveSocketStepUi(false);
    if (!meshLoaded) {
      setStatus('Load a mesh before using Positive Limb.');
    } else {
      setStatus('Positive Limb is active. Select an open edge to build it automatically.');
    }
  }
});

radiusSlider.addEventListener('input', () => {
  syncBrushRadiusFromPercent(Number(radiusSlider.value));
});

radiusInput.addEventListener('input', () => {
  if (radiusInput.value === '') {
    return;
  }

  syncBrushRadiusFromPercent(Number(radiusInput.value));
});
radiusInput.addEventListener('change', () => {
  syncBrushRadiusFromPercent(Number(radiusInput.value));
});

strengthSlider.addEventListener('input', () => {
  syncBrushStrengthFromPercent(Number(strengthSlider.value));
});

strengthInput.addEventListener('input', () => {
  if (strengthInput.value === '') {
    return;
  }

  syncBrushStrengthFromPercent(Number(strengthInput.value));
});
strengthInput.addEventListener('change', () => {
  syncBrushStrengthFromPercent(Number(strengthInput.value));
});

trimlineOnlyButton.addEventListener('click', () => {
  trimlineOnlySmooth = !trimlineOnlySmooth;
  viewport.setSmoothOnlyTrimline(trimlineOnlySmooth);
  trimlineOnlyButton.dataset.active = trimlineOnlySmooth ? 'true' : 'false';
  trimlineOnlyButton.setAttribute('aria-pressed', trimlineOnlySmooth ? 'true' : 'false');
  setStatus(trimlineOnlySmooth ? 'Smooth Brush will only relax open trimline edges.' : 'Smooth Brush will relax the local surface.');
});

selectionToolSelect.addEventListener('change', () => {
  viewport.setSelectionTool(selectionToolSelect.value as SelectionTool);
  syncSelectionUi();
});

selectVisibleToggle.addEventListener('change', () => {
  viewport.setSelectOnlyVisible(!selectVisibleToggle.checked);
});

selectionRadiusSlider.addEventListener('input', () => {
  syncSelectionRadiusFromPercent(Number(selectionRadiusSlider.value));
});

selectionRadiusInput.addEventListener('input', () => {
  if (selectionRadiusInput.value === '') {
    return;
  }

  syncSelectionRadiusFromPercent(Number(selectionRadiusInput.value));
});
selectionRadiusInput.addEventListener('change', () => {
  syncSelectionRadiusFromPercent(Number(selectionRadiusInput.value));
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

positiveFullRemeshEdgeSlider.addEventListener('input', () => {
  positiveFullRemeshEdgeValue.textContent = formatMillimeters(Number(positiveFullRemeshEdgeSlider.value), 3);
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
  const files = Array.from(fileInput.files ?? []);
  if (files.length === 0) {
    return;
  }
  setImportUiDisabled(true);

  try {
    await loadSelectedFiles(files);
  } catch (error) {
    console.error(error);
    const message = error instanceof Error ? error.message : 'Failed to load the selected mesh.';
    setStatus(message, true);
  } finally {
    setImportUiDisabled(false);
    fileInput.value = '';
  }
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    closeOptionMenus();
    setRotateToolActive(false);
    if (measurementPickActive) {
      viewport.cancelMeasurementPick();
      setStatus('Measurement canceled.');
    }
    return;
  }

  if (event.key !== 'Delete') {
    return;
  }

  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) {
    return;
  }

  deleteSelectedFaces();
});

document.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof Node)) {
    closeOptionMenus();
    return;
  }

  if (
    importButton.contains(target) ||
    importMenu.contains(target) ||
    viewButton.contains(target) ||
    viewMenu.contains(target) ||
    rotateButton.contains(target) ||
    rotateMenu.contains(target) ||
    (rotateToolActive && viewportHost.contains(target))
  ) {
    return;
  }

  closeOptionMenus();
});

window.addEventListener('beforeunload', () => viewport.dispose());

async function openObjFolder(): Promise<void> {
  const directoryPicker = (window as Window & { showDirectoryPicker?: () => Promise<unknown> }).showDirectoryPicker;
  if (!directoryPicker) {
    setStatus('Folder import is not supported in this browser. Select the OBJ, MTL, and image files together instead.', true);
    return;
  }

  setImportUiDisabled(true);
  try {
    const directoryHandle = await directoryPicker();
    const files = await collectDirectoryFiles(directoryHandle);
    await loadSelectedFiles(files);
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      return;
    }

    console.error(error);
    const message = error instanceof Error ? error.message : 'Failed to open the selected OBJ folder.';
    setStatus(message, true);
  } finally {
    setImportUiDisabled(false);
  }
}

function setImportUiDisabled(disabled: boolean): void {
  importButton.disabled = disabled;
  importFilesOption.disabled = disabled;
  importFolderOption.disabled = disabled;
  fileInput.disabled = disabled;
  if (disabled) {
    closeOptionMenus();
  }
}

function toggleOptionMenu(menu: HTMLElement, trigger: HTMLButtonElement): void {
  const shouldOpen = menu.hidden;
  closeOptionMenus();
  menu.hidden = !shouldOpen;
  trigger.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
}

function closeOptionMenus(): void {
  importMenu.hidden = true;
  viewMenu.hidden = true;
  rotateMenu.hidden = true;
  importButton.setAttribute('aria-expanded', 'false');
  viewButton.setAttribute('aria-expanded', 'false');
  rotateButton.setAttribute('aria-expanded', 'false');
  setRotateToolActive(false);
}

function setRotateToolActive(active: boolean): void {
  rotateToolActive = active && meshLoaded;
  rotateButton.dataset.active = rotateToolActive ? 'true' : 'false';
  viewport.setRotationOverlayVisible(rotateToolActive);
  if (!rotateToolActive) {
    rotateMenu.hidden = true;
    rotateButton.setAttribute('aria-expanded', 'false');
  }
}

function updateRotationDraftFromInputs(shouldFormat: boolean): void {
  if (!rotateToolActive) {
    return;
  }

  const angles = {
    x: Number(rotateXInput.value),
    y: Number(rotateYInput.value),
    z: Number(rotateZInput.value),
  };
  if (!Number.isFinite(angles.x) || !Number.isFinite(angles.y) || !Number.isFinite(angles.z)) {
    return;
  }

  const result = viewport.setMeshRotationDraft(angles, false);
  if (!result.success) {
    setStatus(result.message, true);
    return;
  }

  if (shouldFormat) {
    syncRotateInputs(angles);
  }
}

function syncRotateInputs(angles: Record<MeshRotationAxis, number>): void {
  rotateXInput.value = angles.x.toFixed(3);
  rotateYInput.value = angles.y.toFixed(3);
  rotateZInput.value = angles.z.toFixed(3);
}

function resetMeasurementActivity(): void {
  if (rotateToolActive) {
    setRotateToolActive(false);
  }

  if (
    !circumferenceOverlayVisible &&
    !measurementPickActive &&
    !measurementStartPickActive &&
    !pointToPointPickActive
  ) {
    return;
  }

  circumferenceOverlayVisible = false;
  viewport.setMeasurementCircumferenceVisible(false);
  viewport.cancelMeasurementPick();
  viewport.cancelMeasurementStartPick();
  viewport.cancelPointToPointMeasurementPick();
  viewport.clearPointToPointMeasurement();
  syncMeasurementControls();
  updateMeasurementUi(viewport.getMeasurementState());
}

function syncMeasurementControls(): void {
  toggleCircumferenceButton.disabled = !meshLoaded;
  setCircumferenceStartButton.disabled = !meshLoaded;
  takeMeasurementButton.disabled = !meshLoaded;
  clearMeasurementButton.disabled = !meshLoaded;
  toggleCircumferenceButton.textContent = circumferenceOverlayVisible ? 'Hide Measurements' : 'Show Measurements';
  toggleCircumferenceButton.dataset.active = circumferenceOverlayVisible ? 'true' : 'false';
  setCircumferenceStartButton.textContent = measurementStartPickActive ? 'Cancel Start' : 'Set Starting Circumference';
  setCircumferenceStartButton.dataset.active = measurementStartPickActive ? 'true' : 'false';
  takeMeasurementButton.textContent = measurementPickActive ? 'Cancel Height' : 'Measure Height';
  takeMeasurementButton.dataset.active = measurementPickActive ? 'true' : 'false';
  clearMeasurementButton.textContent = pointToPointPickActive ? 'Cancel Point-to-Point' : 'Point-to-Point';
  clearMeasurementButton.dataset.active = pointToPointPickActive ? 'true' : 'false';
}

function syncModeUi(): void {
  const mode = modeSelect.value as InteractionMode;
  workflowStepButtons.forEach((button) => {
    const isActive = button.dataset.mode === mode;
    button.dataset.active = isActive ? 'true' : 'false';
    button.setAttribute('aria-current', isActive ? 'step' : 'false');
  });
  sculptControls.hidden = mode !== 'sculpt';
  selectionControls.hidden = mode !== 'select';
  fillControls.hidden = mode !== 'fill';
  boundaryControls.hidden = mode !== 'boundary';
  positiveControls.hidden = mode !== 'positive';
  remeshControls.hidden = mode !== 'remesh';
  thickenControls.hidden = mode !== 'thicken';
  historyControls.hidden = false;

  if (mode === 'fill') {
    modeHintPrimary.textContent = 'Bright blue lines show open or non-manifold edge loops.';
    modeHintSecondary.textContent =
      'Move near a loop to preview it in purple, then left click to patch a clean boundary loop.';
    modeHintTertiary.textContent = 'Right drag rotates and middle drag pans while Fill Hole mode stays active.';
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
      'Set the thickness in millimeters, then apply it. Right drag rotates and middle drag pans.';
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
      'Positive Limb uses the same clean loop targeting, then extrudes the remeshed boundary outward with a live distance slider.';
    modeHintTertiary.textContent =
      'The offset stage auto-remeshes at extrude distance / 8, the band smooth keeps the selected patch boundary fixed, and the final wall extrusion uses X/Y tilt sliders with a bbox-sized length.';
    return;
  }

  if (mode === 'sculpt') {
    modeHintPrimary.innerHTML = 'Left drag smooths and right drag rotates.';
    modeHintSecondary.textContent = 'Mouse wheel zooms and middle drag pans.';
    modeHintTertiary.textContent = 'Brush radius is scaled from 0-100% with 100% equal to 40 mm.';
  } else {
    modeHintPrimary.innerHTML =
      'Sphere paints local face selection. Box and Snip drag a screen-space selection.';
    modeHintSecondary.textContent =
      'Select Through includes faces behind the visible front surface.';
    modeHintTertiary.textContent =
      'Right drag rotates, middle drag pans, Shift adds, Ctrl subtracts, and Delete removes selected faces.';
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

  if (modeSelect.value === 'positive' && positiveSocketStepIndex === POSITIVE_SOCKET_COMPLETE_STEP_INDEX) {
    positiveBoundaryTargetStatus.textContent = 'Positive Limb complete.';
  } else if (modeSelect.value === 'positive') {
    positiveBoundaryTargetStatus.textContent = 'Select an open edge to build the Positive Limb.';
  } else {
    positiveBoundaryTargetStatus.textContent = 'Positive Limb is inactive.';
  }

  // Boundary workflow updates are often emitted from the preview/commit methods
  // themselves. Re-entering the live preview here creates a feedback loop where
  // a commit triggers a preview, which mutates state again and can stall the wizard.
  syncSocketModelStepUi(false);
  syncPositiveSocketStepUi(false);
}

function updateMeshStats(stats: MeshStats): void {
  applyRemeshButton.disabled = stats.triangleCount === 0;
  applyThickenButton.disabled = stats.triangleCount === 0;
  meshLoaded = stats.triangleCount > 0;
  exportButton.disabled = !meshLoaded;
  rotateButton.disabled = !meshLoaded;
  if (stats.triangleCount === 0) {
    fileName.textContent = 'No file loaded';
    currentFilename = 'No file loaded';
    currentTextureFile = null;
    setRotateToolActive(false);
    closeOptionMenus();
  } else {
    fileName.textContent = currentFilename;
  }

  syncMeasurementControls();
  syncViewMenuUi();
  configureOperationSliders(stats.boundsRadius);
}

function readDarkThemePreference(): boolean {
  try {
    const storedPreference = window.localStorage.getItem(DARK_THEME_STORAGE_KEY);
    return storedPreference === null ? true : storedPreference === 'true';
  } catch {
    return true;
  }
}

function readExportFormatPreference(): MeshExportFormat {
  try {
    const storedPreference = window.localStorage.getItem(EXPORT_FORMAT_STORAGE_KEY);
    return isMeshExportFormat(storedPreference) ? storedPreference : 'obj';
  } catch {
    return 'obj';
  }
}

function readExportUnitPreference(): MeshExportUnit {
  try {
    const storedPreference = window.localStorage.getItem(EXPORT_UNIT_STORAGE_KEY);
    return isMeshExportUnit(storedPreference) ? storedPreference : 'mm';
  } catch {
    return 'mm';
  }
}

function persistExportPreferences(): void {
  try {
    window.localStorage.setItem(EXPORT_FORMAT_STORAGE_KEY, lastExportFormat);
    window.localStorage.setItem(EXPORT_UNIT_STORAGE_KEY, lastExportUnit);
  } catch {
    // Export preference persistence is optional; exporting still works without storage.
  }
}

function isMeshExportFormat(value: string | null): value is MeshExportFormat {
  return value === 'obj' || value === 'stl';
}

function isMeshExportUnit(value: string | null): value is MeshExportUnit {
  return value === 'mm' || value === 'cm' || value === 'm' || value === 'in';
}

function formatMeshExportUnit(unit: MeshExportUnit): string {
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

function setDarkTheme(enabled: boolean, shouldPersist = true): void {
  darkThemeEnabled = enabled;
  document.documentElement.dataset.theme = enabled ? 'dark' : 'light';
  darkThemeButton.dataset.active = enabled ? 'true' : 'false';
  darkThemeButton.setAttribute('aria-pressed', enabled ? 'true' : 'false');
  viewport.setUiTheme(enabled ? 'dark' : 'light');

  if (!shouldPersist) {
    return;
  }

  try {
    window.localStorage.setItem(DARK_THEME_STORAGE_KEY, enabled ? 'true' : 'false');
  } catch {
    // Theme persistence is a convenience only; the toggle still works without storage.
  }
}

function syncViewMenuUi(): void {
  viewModeButtons.forEach((button) => {
    const isActive = button.dataset.viewMode === meshViewMode;
    button.dataset.active = isActive ? 'true' : 'false';
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
}

function formatViewMode(mode: MeshViewMode): string {
  return mode.charAt(0).toUpperCase() + mode.slice(1);
}

function updateMeasurementUi(state: MeasurementState): void {
  const measurementVisible =
    circumferenceOverlayVisible ||
    measurementPickActive ||
    pointToPointPickActive ||
    state.clickedHeightMm !== null ||
    state.pointToPointDistanceMm !== null;
  measurementTotalHeight.textContent =
    measurementVisible && state.totalHeightMm > 0 ? formatMillimeters(state.totalHeightMm, 1) : '--';
  measurementClickHeight.textContent =
    measurementVisible
      ? state.clickedHeightMm === null
        ? 'Click scan'
        : formatMillimeters(state.clickedHeightMm, 1)
      : 'Take measurement';

  measurementTableBody.replaceChildren();
  if (!circumferenceOverlayVisible) {
    measurementHoveredIndex = null;
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 2;
    cell.textContent = meshLoaded ? 'Calculate circumferences.' : 'Load a scan.';
    row.append(cell);
    measurementTableBody.append(row);
    return;
  }

  if (state.rows.length === 0) {
    measurementHoveredIndex = null;
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 2;
    cell.textContent = state.totalHeightMm > 0 ? 'No 25 mm sections found.' : 'Load a scan.';
    row.append(cell);
    measurementTableBody.append(row);
    return;
  }

  if (measurementHoveredIndex !== null && measurementHoveredIndex >= state.rows.length) {
    measurementHoveredIndex = null;
  }

  for (const measurement of state.rows) {
    const index = state.rows.indexOf(measurement);
    const row = document.createElement('tr');
    row.dataset.measurementIndex = String(index);
    row.classList.toggle('is-hovered', measurementHoveredIndex === index);
    row.addEventListener('mouseenter', () => {
      measurementHoveredIndex = index;
      viewport.setHoveredMeasurementIndex(index);
      syncMeasurementHoverRows();
    });
    row.addEventListener('mouseleave', () => {
      measurementHoveredIndex = null;
      viewport.setHoveredMeasurementIndex(null);
      syncMeasurementHoverRows();
    });
    const distanceCell = document.createElement('td');
    const circumferenceCell = document.createElement('td');
    distanceCell.textContent = formatMillimeters(measurement.distanceFromDistalMm, 0);
    circumferenceCell.textContent = formatMillimeters(measurement.circumferenceMm, 1);
    row.append(distanceCell, circumferenceCell);
    measurementTableBody.append(row);
  }
}

function syncMeasurementHoverRows(): void {
  measurementTableBody
    .querySelectorAll<HTMLTableRowElement>('tr[data-measurement-index]')
    .forEach((row) => {
      row.classList.toggle('is-hovered', Number(row.dataset.measurementIndex) === measurementHoveredIndex);
    });
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

let operationErrorTimer: number | null = null;

function showOperationError(action: string, detail: string): void {
  const cleanDetail = detail.trim() || 'The action failed before it returned a detailed message.';
  operationErrorMessage.textContent = `${action} failed: ${cleanDetail}`;
  operationErrorToast.hidden = false;

  if (operationErrorTimer !== null) {
    window.clearTimeout(operationErrorTimer);
  }
  operationErrorTimer = window.setTimeout(() => {
    hideOperationError();
  }, 10000);
}

function hideOperationError(): void {
  operationErrorToast.hidden = true;
  if (operationErrorTimer !== null) {
    window.clearTimeout(operationErrorTimer);
    operationErrorTimer = null;
  }
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
    POSITIVE_SOCKET_STEP_PANELS[i].hidden = true;
  }

  positiveSocketStepLabel.hidden = true;
  positiveSocketStepTitle.hidden = true;
  positiveSocketPrevButton.hidden = true;
  positiveSocketNextButton.hidden = true;
  positiveSocketStepLabel.textContent = '';
  positiveSocketStepTitle.textContent = POSITIVE_SOCKET_STEP_TITLES[positiveSocketStepIndex];
  positiveSocketPrevButton.disabled = positiveSocketStepIndex === POSITIVE_SOCKET_FULL_REMESH_STEP_INDEX;
  positiveSocketNextButton.textContent =
    positiveSocketStepIndex === POSITIVE_SOCKET_COMPLETE_STEP_INDEX
      ? 'Start Over'
      : positiveSocketStepIndex === POSITIVE_SOCKET_DIRECTIONAL_EXTRUDE_STEP_INDEX
        ? 'Finish'
        : 'Next';

  const state = viewport.getBoundaryWorkflowState();
  positiveSocketNextButton.disabled =
    modeSelect.value === 'positive' &&
    positiveSocketStepIndex === POSITIVE_SOCKET_TARGET_STEP_INDEX &&
    !state.hasSelectedBoundary;
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
    case POSITIVE_SOCKET_FULL_REMESH_STEP_INDEX:
      result = viewport.applySurfaceRemesh(Number(positiveFullRemeshEdgeSlider.value), 'refined');
      break;
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
      positiveSocketStepIndex = POSITIVE_SOCKET_FULL_REMESH_STEP_INDEX;
      syncPositiveSocketStepUi(false);
      setStatus('Positive Limb reset. Run the full mesh remesh to begin again.');
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
      setStatus('Positive Limb finished. Press Next to start over on a new boundary, or Previous to inspect the earlier steps.');
    }
  }
}

async function loadSelectedFiles(files: File[]): Promise<void> {
  const file = files.find((candidate) => ['obj', 'stl'].includes(getExtension(candidate.name)));
  if (!file) {
    setStatus('Choose an STL or OBJ mesh.', true);
    return;
  }

  setStatus(`Loading ${file.name} as millimeters...`);
  const loaded = await loadMeshFile(files, 'mm');
  currentFilename = loaded.filename;
  currentTextureFile = loaded.textureFile;
  loaded.geometry.computeBoundsTree({
    maxLeafSize: 20,
    setBoundingBox: false,
    indirect: true,
  });

  const editable = createEditableMeshData(loaded.geometry);
  const engine = new SculptEngine(editable, 12);
  viewport.setSession(editable, engine, loaded.texture);
  viewport.setBrushType('smooth');
  viewport.setBrushRadiusMm(radiusPercentToMillimeters(Number(radiusSlider.value)));
  viewport.setBrushStrength(percentToUnit(Number(strengthSlider.value)));
  viewport.setSmoothOnlyTrimline(trimlineOnlySmooth);
  viewport.setInteractionMode(modeSelect.value as InteractionMode);
  viewport.setSelectionTool(selectionToolSelect.value as SelectionTool);
  viewport.setSelectOnlyVisible(!selectVisibleToggle.checked);
  viewport.setSelectionRadiusMm(radiusPercentToMillimeters(Number(selectionRadiusSlider.value)));
  viewport.setMeshViewMode(meshViewMode);

  fileName.textContent = currentFilename;
  setStatus(
    `Loaded ${loaded.filename} (${loaded.extension.toUpperCase()})${loaded.texture ? ' with texture' : ''} using ${loaded.importUnit} -> mm with ${formatCount(
      loaded.triangleCount,
    )} triangles.`,
  );
}

async function collectDirectoryFiles(directoryHandle: unknown): Promise<File[]> {
  const files: File[] = [];
  await collectDirectoryFilesRecursive(directoryHandle, files);
  return files;
}

async function collectDirectoryFilesRecursive(directoryHandle: unknown, files: File[]): Promise<void> {
  const entries = (directoryHandle as { entries?: () => AsyncIterable<[string, unknown]> }).entries;
  if (!entries) {
    throw new Error('This browser did not return a readable folder handle.');
  }

  for await (const [, handle] of entries.call(directoryHandle)) {
    const kind = (handle as { kind?: string }).kind;
    if (kind === 'file') {
      const file = await (handle as { getFile: () => Promise<File> }).getFile();
      if (isImportRelevantFile(file.name)) {
        files.push(file);
      }
    } else if (kind === 'directory') {
      await collectDirectoryFilesRecursive(handle, files);
    }
  }
}

function isImportRelevantFile(filename: string): boolean {
  return ['obj', 'stl', 'mtl', 'png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(getExtension(filename));
}

function getExtension(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() ?? '';
}

function getExportBaseName(): string {
  const withoutExtension = currentFilename.replace(/\.[^.]+$/, '');
  const sanitized = withoutExtension.replace(/[<>:"/\\|?*\u0000-\u001f]+/g, '_').trim();
  return sanitized || 'NouraSoft_export';
}

function promptForExportBaseName(): string | null {
  const requestedName = window.prompt('Export file name', getExportBaseName());
  if (requestedName === null) {
    return null;
  }

  const sanitized = requestedName
    .replace(/\.[^.]+$/, '')
    .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, '_')
    .trim();
  return sanitized || 'NouraSoft_export';
}

async function saveExportedBlobFiles(files: Array<{ filename: string; blob: Blob }>): Promise<boolean> {
  const desktopApi = getNouraDesktopApi();
  if (desktopApi?.saveExportedFiles) {
    const payloadFiles = await Promise.all(
      files.map(async (file) => ({
        filename: file.filename,
        mimeType: file.blob.type,
        data: await file.blob.arrayBuffer(),
      })),
    );
    const result = await desktopApi.saveExportedFiles({ files: payloadFiles });
    return !result.canceled;
  }

  for (const file of files) {
    downloadBlob(file.filename, file.blob);
  }
  return true;
}

function getNouraDesktopApi(): NouraDesktopApi | null {
  return (window as Window & { nouraDesktop?: NouraDesktopApi }).nouraDesktop ?? null;
}

function getExportErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.rel = 'noopener';
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 5000);
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

  configureBrushControls();
  configureSelectionRadiusControls();

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
    positiveFullRemeshEdgeSlider,
    positiveFullRemeshEdgeValue,
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

function configureBrushControls(): void {
  radiusSlider.min = '0';
  radiusSlider.max = '100';
  radiusSlider.step = '1';
  radiusInput.min = '0';
  radiusInput.max = '100';
  radiusInput.step = '1';

  strengthSlider.min = '0';
  strengthSlider.max = '100';
  strengthSlider.step = '1';
  strengthInput.min = '0';
  strengthInput.max = '100';
  strengthInput.step = '1';

  syncBrushRadiusFromPercent(clampPercent(Number(radiusSlider.value), DEFAULT_BRUSH_RADIUS_PERCENT));
  syncBrushStrengthFromPercent(clampPercent(Number(strengthSlider.value), DEFAULT_BRUSH_STRENGTH_PERCENT));
}

function configureSelectionRadiusControls(): void {
  selectionRadiusSlider.min = '0';
  selectionRadiusSlider.max = '100';
  selectionRadiusSlider.step = '1';
  selectionRadiusInput.min = '0';
  selectionRadiusInput.max = '100';
  selectionRadiusInput.step = '1';
  syncSelectionRadiusFromPercent(clampPercent(Number(selectionRadiusSlider.value), 15));
}

function syncBrushRadiusFromPercent(value: number): void {
  const percent = clampWholePercent(value, DEFAULT_BRUSH_RADIUS_PERCENT);
  const displayValue = percent.toFixed(0);
  radiusSlider.value = displayValue;
  radiusInput.value = displayValue;
  viewport.setBrushRadiusMm(radiusPercentToMillimeters(percent));
}

function syncSelectionRadiusFromPercent(value: number): void {
  const percent = clampWholePercent(value, 15);
  const displayValue = percent.toFixed(0);
  selectionRadiusSlider.value = displayValue;
  selectionRadiusInput.value = displayValue;
  viewport.setSelectionRadiusMm(radiusPercentToMillimeters(percent));
}

function syncBrushStrengthFromPercent(value: number): void {
  const percent = clampPercent(value, DEFAULT_BRUSH_STRENGTH_PERCENT);
  const displayValue = percent.toFixed(0);
  strengthSlider.value = displayValue;
  strengthInput.value = displayValue;
  viewport.setBrushStrength(percentToUnit(percent));
}

function radiusPercentToMillimeters(percent: number): number {
  return clampPercent(percent, DEFAULT_BRUSH_RADIUS_PERCENT) / 100 * BRUSH_RADIUS_MAX_MM;
}

function percentToUnit(percent: number): number {
  return clampPercent(percent, DEFAULT_BRUSH_STRENGTH_PERCENT) / 100;
}

function clampPercent(value: number, fallback: number): number {
  const safeValue = Number.isFinite(value) ? value : fallback;
  return Math.min(100, Math.max(0, safeValue));
}

function clampWholePercent(value: number, fallback: number): number {
  return Math.round(clampPercent(value, fallback));
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

