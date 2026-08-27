import { spawn } from 'node:child_process';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const electronCommand = process.platform === 'win32'
  ? path.join(rootDir, 'node_modules', '.bin', 'electron.cmd')
  : path.join(rootDir, 'node_modules', '.bin', 'electron');
const rendererUrl = 'http://127.0.0.1:5173';
const childProcesses = [];

function spawnChecked(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: rootDir,
    stdio: 'inherit',
    shell: false,
    ...options,
  });
  childProcesses.push(child);
  return child;
}

function runToCompletion(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawnChecked(command, args);
    child.on('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command} ${args.join(' ')} exited with code ${code ?? 'unknown'}`));
      }
    });
    child.on('error', reject);
  });
}

function waitForServer(url, timeoutMs = 30000) {
  const startedAt = Date.now();

  return new Promise((resolve, reject) => {
    const tick = () => {
      const request = http.get(url, (response) => {
        response.resume();
        resolve();
      });

      request.on('error', () => {
        if (Date.now() - startedAt >= timeoutMs) {
          reject(new Error(`Timed out waiting for ${url}`));
          return;
        }
        setTimeout(tick, 300);
      });
      request.setTimeout(1000, () => {
        request.destroy();
      });
    };

    tick();
  });
}

function stopChildren() {
  for (const child of childProcesses) {
    if (!child.killed) {
      child.kill();
    }
  }
}

process.on('SIGINT', () => {
  stopChildren();
  process.exit(130);
});
process.on('SIGTERM', () => {
  stopChildren();
  process.exit(143);
});

await runToCompletion(npmCommand, ['run', 'build:electron']);

const vite = spawnChecked(npmCommand, ['run', 'dev', '--', '--host', '127.0.0.1', '--port', '5173', '--strictPort']);
vite.on('exit', (code) => {
  if (code !== 0) {
    stopChildren();
    process.exit(code ?? 1);
  }
});

await waitForServer(rendererUrl);

const electronEnv = {
  ...process.env,
  ELECTRON_RENDERER_URL: rendererUrl,
};
delete electronEnv.ELECTRON_RUN_AS_NODE;

const electron = spawnChecked(electronCommand, ['.'], {
  env: electronEnv,
});

electron.on('exit', (code) => {
  stopChildren();
  process.exit(code ?? 0);
});
