
// --- Live Devoxelization Extension (Unified Server) ---
let devoxEnabled = false;
let currentDevoxMesh = null;
let lastDevoxState = null;
let lastDevoxView = 'solid';
let devoxGroup = null;
let devoxMeshObj = null;
let devoxLinesObj = null;
let devoxBuiltMesh = null;
let devoxBuiltRevision = -1;

function rebuildDevoxSceneObjects() {
  if (!mass || !mass.parent) return;
  if (!devoxGroup) {
    devoxGroup = new THREE.Group();
    devoxGroup.name = 'devoxGroup';
    mass.parent.add(devoxGroup);
  }
  const active = Boolean(devoxEnabled && currentDevoxMesh?.faces?.length && lastDevoxView === 'solid' && lastDevoxState && currentDevoxMesh.revision === lastDevoxState.revision);
  mass.visible = !active;
  if (typeof coreMass !== 'undefined' && coreMass) coreMass.visible = !active;
  if (outline) outline.visible = !active && lastDevoxView === 'solid';
  devoxGroup.visible = active;
  if (!active || (devoxBuiltMesh === currentDevoxMesh && devoxBuiltRevision === lastDevoxState.revision)) return;
  for (const object of [devoxMeshObj,devoxLinesObj]) {
    if (!object) continue;
    devoxGroup.remove(object);object.geometry.dispose();object.material.dispose();
  }
  devoxMeshObj = devoxLinesObj = null;
  devoxBuiltMesh = currentDevoxMesh;devoxBuiltRevision = lastDevoxState.revision;

  const verts = currentDevoxMesh.vertices;
  const faces = currentDevoxMesh.faces;
  const creases = currentDevoxMesh.creases || [];
  const offset = sceneOffset(lastDevoxState);
  const angle = lastDevoxState.context?.gridAngle || 0;
  const displayLift = Math.max(0.2, ...(lastDevoxState.context?.site?.vertices || []).map(v => v[2])) + 0.03;

  const worldVerts = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    const [gx, gy, gz] = verts[i];
    const [wx, wy] = gridToWorld(gx, gy, offset, angle);
    worldVerts[i * 3] = wx;
    worldVerts[i * 3 + 1] = gz * 3.0 + displayLift;
    worldVerts[i * 3 + 2] = -wy;
  }

  let geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(worldVerts, 3));
  const idx = [];
  for (let i = 0; i < faces.length; i++) {
    idx.push(faces[i][0], faces[i][1], faces[i][2]);
  }
  geo.setIndex(idx);

  if (currentDevoxMesh.nonFlat) {
    geo.computeVertexNormals();
  } else {
    geo = geo.toNonIndexed();
    geo.computeVertexNormals();
  }

  // Match VoxelForm signature lightBlue ink shading when using vertexColors or uniform
  const useVC = Boolean(mass.material && mass.material.vertexColors);
  const posAttr = geo.getAttribute('position');
  if (useVC && posAttr) {
    const cols = new Float32Array(posAttr.count * 3);
    for (let i = 0; i < posAttr.count; i++) {
      cols[i * 3] = 1;
      cols[i * 3 + 1] = 1;
      cols[i * 3 + 2] = 1;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  }

  const mat = mass.material ? mass.material.clone() : new THREE.MeshLambertMaterial({ color: solid });
  mat.onBeforeCompile = mass.material.onBeforeCompile;
  devoxMeshObj = new THREE.Mesh(geo, mat);
  devoxMeshObj.castShadow = true;
  devoxMeshObj.receiveShadow = true;
  devoxGroup.add(devoxMeshObj);

  if (creases.length) {
    const linePos = new Float32Array(creases.length * 6);
    for (let i = 0; i < creases.length; i++) {
      const [u, v] = creases[i];
      linePos[i * 6 + 0] = worldVerts[u * 3 + 0];
      linePos[i * 6 + 1] = worldVerts[u * 3 + 1] + 0.015;
      linePos[i * 6 + 2] = worldVerts[u * 3 + 2];
      linePos[i * 6 + 3] = worldVerts[v * 3 + 0];
      linePos[i * 6 + 4] = worldVerts[v * 3 + 1] + 0.015;
      linePos[i * 6 + 5] = worldVerts[v * 3 + 2];
    }
    const lineGeo = new LineSegmentsGeometry();lineGeo.setPositions(linePos);
    const lineMat = new LineMaterial({color:outlineDark,worldUnits:true,linewidth:outlineWorldWidth || .15,depthWrite:false,toneMapped:false});
    devoxLinesObj = new LineSegments2(lineGeo,lineMat);
    devoxLinesObj.onBeforeRender = renderer => {renderer.getSize(lineMat.resolution);lineMat.linewidth=outlineWorldWidth || .15;};
    devoxGroup.add(devoxLinesObj);
  }
  if (typeof render === 'function') render();
  else if (typeof requestRender === 'function') requestRender();
  else window.dispatchEvent(new Event('resize'));
}

window.addEventListener('message', event => {
  if (event.source !== window.parent || event.origin !== location.origin) return;
  const msg = event.data;
  if (!msg) return;
  if (['voxel_initialize', 'voxel_update', 'voxel_display'].includes(msg.type)) {
    if (msg.state) lastDevoxState = msg.state;
    if (msg.view) lastDevoxView = msg.view;
    rebuildDevoxSceneObjects();
  } else if (msg.type === 'voxel_devox') {
    devoxEnabled = Boolean(msg.enabled);
    currentDevoxMesh = msg.mesh || null;
    rebuildDevoxSceneObjects();
  }
});
