export function captureBoundaryGuide(
  positions: ArrayLike<number>,
  orderedVertexIds: Uint32Array,
): Float32Array | null {
  if (orderedVertexIds.length < 3) {
    return null;
  }

  const guide = new Float32Array(orderedVertexIds.length * 3);
  for (let i = 0; i < orderedVertexIds.length; i += 1) {
    const sourceOffset = orderedVertexIds[i] * 3;
    const targetOffset = i * 3;
    guide[targetOffset] = positions[sourceOffset];
    guide[targetOffset + 1] = positions[sourceOffset + 1];
    guide[targetOffset + 2] = positions[sourceOffset + 2];
  }

  return guide;
}

export function smoothBoundaryLoopVertices(
  positions: ArrayLike<number>,
  normals: ArrayLike<number>,
  orderedVertexIds: Uint32Array,
  intensity: number,
  iterations: number,
): Float32Array | null {
  if (orderedVertexIds.length < 3) {
    return null;
  }

  const resolvedIntensity = clamp(intensity, 0, 1);
  const resolvedIterations = Math.max(1, Math.round(iterations));
  let current = new Float32Array(positions);
  let next = new Float32Array(current.length);

  for (let iteration = 0; iteration < resolvedIterations; iteration += 1) {
    next.set(current);

    for (let i = 0; i < orderedVertexIds.length; i += 1) {
      const vertex = orderedVertexIds[i];
      const previousVertex = orderedVertexIds[(i - 1 + orderedVertexIds.length) % orderedVertexIds.length];
      const nextVertex = orderedVertexIds[(i + 1) % orderedVertexIds.length];

      const vertexOffset = vertex * 3;
      const previousOffset = previousVertex * 3;
      const nextOffset = nextVertex * 3;

      const targetX = (current[previousOffset] + current[nextOffset]) * 0.5;
      const targetY = (current[previousOffset + 1] + current[nextOffset + 1]) * 0.5;
      const targetZ = (current[previousOffset + 2] + current[nextOffset + 2]) * 0.5;

      let deltaX = targetX - current[vertexOffset];
      let deltaY = targetY - current[vertexOffset + 1];
      let deltaZ = targetZ - current[vertexOffset + 2];

      const normalX = normals[vertexOffset];
      const normalY = normals[vertexOffset + 1];
      const normalZ = normals[vertexOffset + 2];
      const normalLength = Math.hypot(normalX, normalY, normalZ);
      if (normalLength > 1e-8) {
        const invNormalLength = 1 / normalLength;
        const normalizedX = normalX * invNormalLength;
        const normalizedY = normalY * invNormalLength;
        const normalizedZ = normalZ * invNormalLength;
        const normalComponent =
          deltaX * normalizedX + deltaY * normalizedY + deltaZ * normalizedZ;
        deltaX -= normalComponent * normalizedX;
        deltaY -= normalComponent * normalizedY;
        deltaZ -= normalComponent * normalizedZ;
      }

      next[vertexOffset] = current[vertexOffset] + deltaX * resolvedIntensity;
      next[vertexOffset + 1] = current[vertexOffset + 1] + deltaY * resolvedIntensity;
      next[vertexOffset + 2] = current[vertexOffset + 2] + deltaZ * resolvedIntensity;
    }

    const swap = current;
    current = next;
    next = swap;
  }

  return current;
}

export function selectTrianglesNearBoundaryGuide(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  guide: Float32Array,
  distanceMm: number,
): Uint8Array {
  const triangleCount = Math.floor(indices.length / 3);
  const selection = new Uint8Array(triangleCount);
  if (guide.length < 6 || distanceMm <= 0) {
    return selection;
  }

  const distanceSq = distanceMm * distanceMm;
  const centroidPaddingSq = (distanceMm * 1.2) * (distanceMm * 1.2);

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const triOffset = triangle * 3;
    const aOffset = indices[triOffset] * 3;
    const bOffset = indices[triOffset + 1] * 3;
    const cOffset = indices[triOffset + 2] * 3;

    if (
      pointToGuideDistanceSq(positions[aOffset], positions[aOffset + 1], positions[aOffset + 2], guide) <= distanceSq ||
      pointToGuideDistanceSq(positions[bOffset], positions[bOffset + 1], positions[bOffset + 2], guide) <= distanceSq ||
      pointToGuideDistanceSq(positions[cOffset], positions[cOffset + 1], positions[cOffset + 2], guide) <= distanceSq
    ) {
      selection[triangle] = 1;
      continue;
    }

    const centroidX = (positions[aOffset] + positions[bOffset] + positions[cOffset]) / 3;
    const centroidY = (positions[aOffset + 1] + positions[bOffset + 1] + positions[cOffset + 1]) / 3;
    const centroidZ = (positions[aOffset + 2] + positions[bOffset + 2] + positions[cOffset + 2]) / 3;
    if (pointToGuideDistanceSq(centroidX, centroidY, centroidZ, guide) <= centroidPaddingSq) {
      selection[triangle] = 1;
    }
  }

  return selection;
}

function pointToGuideDistanceSq(x: number, y: number, z: number, guide: Float32Array): number {
  let bestDistanceSq = Infinity;

  for (let i = 0; i < guide.length; i += 3) {
    const next = (i + 3) % guide.length;
    const ax = guide[i];
    const ay = guide[i + 1];
    const az = guide[i + 2];
    const bx = guide[next];
    const by = guide[next + 1];
    const bz = guide[next + 2];
    const distanceSq = pointToSegmentDistanceSq(x, y, z, ax, ay, az, bx, by, bz);
    if (distanceSq < bestDistanceSq) {
      bestDistanceSq = distanceSq;
    }
  }

  return bestDistanceSq;
}

function pointToSegmentDistanceSq(
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

  const t = clamp((apx * abx + apy * aby + apz * abz) / abLengthSq, 0, 1);
  const closestX = ax + abx * t;
  const closestY = ay + aby * t;
  const closestZ = az + abz * t;
  const dx = px - closestX;
  const dy = py - closestY;
  const dz = pz - closestZ;
  return dx * dx + dy * dy + dz * dz;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
