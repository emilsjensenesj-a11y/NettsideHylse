const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

function sleep(milliseconds) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
}

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== 'win32') {
    return;
  }

  const iconPath = path.join(context.packager.projectDir, 'build', 'icon.ico');
  const rceditPath = path.join(context.packager.projectDir, 'node_modules', 'electron-winstaller', 'vendor', 'rcedit.exe');
  const executablePath = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.exe`);

  if (!fs.existsSync(iconPath) || !fs.existsSync(rceditPath) || !fs.existsSync(executablePath)) {
    return;
  }

  let lastError = null;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      execFileSync(rceditPath, [executablePath, '--set-icon', iconPath], { stdio: 'inherit' });
      return;
    } catch (error) {
      lastError = error;
      sleep(750 * attempt);
    }
  }

  console.warn(`Unable to embed the unpacked exe icon after retries: ${lastError?.message ?? lastError}`);
};
