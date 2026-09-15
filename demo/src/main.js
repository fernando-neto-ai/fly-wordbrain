import './styles.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { attachFlyMotion } from './fly-motion.js';

const $ = (id) => document.getElementById(id);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const state = { manifest: null, layout: null, trace: null, frame: 0, playing: false, speed: 1, narration: false, motion: 'idle', view: 'fly', ready: false };
let lastStep = 0, transitionStart = 0, loadSequence = 0;
const frameDuration = () => 360 / state.speed;
const byId = new Map();
let actor, flyModel, motion, brainMain, brainInset, neuronGeometry, brainUniforms, flyMaterials = [];
let renderMain, renderInset, controls, scene, camera, insetScene, insetCamera;
let cameraGoal = null, targetGoal = null, hovering = 0, movementTime = 0, walkPhase = 0;
const clock = new THREE.Clock();

function toast(message) { $('error-toast').textContent = message; $('error-toast').hidden = false; setTimeout(() => { $('error-toast').hidden = true; }, 7000); }
function createRenderer(element, transparent = true) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: transparent, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.setClearColor(0x000000, 0);
  element.appendChild(renderer.domElement);
  return renderer;
}
function setupScene() {
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x10221d, 0.03);
  camera = new THREE.PerspectiveCamera(33, 1, 0.02, 100);
  camera.position.set(5.6, 4.0, 8.7);
  renderMain = createRenderer($('fly-stage'));
  renderMain.shadowMap.enabled = true;
  renderMain.shadowMap.type = THREE.PCFSoftShadowMap;
  const pmrem = new THREE.PMREMGenerator(renderMain);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  controls = new OrbitControls(camera, renderMain.domElement);
  controls.target.set(0, 0.5, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 2.3; controls.maxDistance = 16;
  controls.maxPolarAngle = Math.PI * .48;
  controls.enablePan = false;
  controls.addEventListener('start', () => { cameraGoal = null; targetGoal = null; });
  scene.add(new THREE.HemisphereLight(0xd4f0de, 0x192f20, 1.5));
  const key = new THREE.DirectionalLight(0xffdda9, 2.8);
  key.position.set(1.5, 8, 5); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -6, right: 6, top: 5, bottom: -5, near: 0.1, far: 25 });
  key.shadow.bias = -.0005; key.shadow.normalBias = .03;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xa3dbcf, 3.3); rim.position.set(-5, 4, -4); scene.add(rim);
  const warm = new THREE.PointLight(0xe2a75e, 8, 12); warm.position.set(4, 1.5, 1); scene.add(warm);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.ShadowMaterial({ opacity: .35 }));
  ground.rotation.x = -Math.PI / 2; ground.position.y = -.9; ground.receiveShadow = true; scene.add(ground);
  const disk = new THREE.Mesh(new THREE.CircleGeometry(3.7, 128), new THREE.MeshStandardMaterial({ color: 0x13241b, roughness: .94, transparent: true, opacity: .18, depthWrite: false }));
  disk.rotation.x = -Math.PI / 2; disk.position.set(0, -.915, 0); disk.receiveShadow = true; scene.add(disk);
  for (const radius of [2.9, 3.55]) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(radius, radius + .006, 128), new THREE.MeshBasicMaterial({ color: 0x5d7760, transparent: true, opacity: .3, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = -.90; scene.add(ring);
  }
  actor = new THREE.Group(); actor.position.set(.2, -.92, 0); actor.scale.setScalar(1.45); scene.add(actor);
  insetScene = new THREE.Scene();
  insetCamera = new THREE.PerspectiveCamera(36, 1, .01, 20); insetCamera.position.set(0, .04, 2.1);
  renderInset = createRenderer($('brain-stage')); renderInset.toneMappingExposure = 1.3;
  const resize = () => {
    for (const [renderer, cam, element] of [[renderMain, camera, $('fly-stage')], [renderInset, insetCamera, $('brain-stage')]]) {
      const { width, height } = element.getBoundingClientRect();
      renderer.setSize(width, height, false); cam.aspect = width / Math.max(height, 1); cam.updateProjectionMatrix();
    }
  };
  new ResizeObserver(resize).observe($('fly-stage')); new ResizeObserver(resize).observe($('brain-stage')); resize();
  const motions = document.createElement('div'); motions.className = 'motion-controls';
  motions.innerHTML = '<span>ILLUSTRATED MOVEMENT</span><button data-motion="idle" class="active" aria-pressed="true">Listen</button><button data-motion="walk" aria-pressed="false">Wander</button><button data-motion="fly" aria-pressed="false">Take flight ↗</button>';
  document.querySelector('.specimen').appendChild(motions);
  motions.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
    state.motion = button.dataset.motion; motions.querySelectorAll('button').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', b === button); });
  }));
  const label = document.createElement('span'); label.className = 'brain-close-label'; label.textContent = 'ANATOMICAL CLOSE-UP · ENLARGED'; document.querySelector('.specimen').appendChild(label);
  loadFly();
}

async function loadFly() {
  try {
    const [gltf, meta, rig] = await Promise.all([new GLTFLoader().loadAsync('/assets/fly.glb'), fetchJSON('/assets/fly-meta.json'), fetchJSON('/assets/fly-rig.json')]);
    flyModel = gltf.scene;
    flyModel.traverse(node => {
      if (!node.isMesh) return;
      node.castShadow = true; node.receiveShadow = true;
      node.material = node.material.clone();
      const name = node.name.toLowerCase();
      if (name.includes('wing')) {
        node.material.transparent = true; node.material.opacity = .43; node.material.side = THREE.DoubleSide;
        node.material.depthWrite = false; node.material.roughness = .28; node.material.metalness = .15;
        node.castShadow = false;
      } else if (name.includes('eye')) {
        node.material.color.set(0x7e352b); node.material.roughness = .34; node.material.metalness = .22;
      } else {
        node.material.color.multiply(new THREE.Color(0xbfa57f));
        node.material.roughness = .56; node.material.metalness = .17;
      }
      flyMaterials.push({ material: node.material, opacity: node.material.opacity, transparent: node.material.transparent, name });
    });
    actor.add(flyModel); motion = attachFlyMotion(flyModel, rig);
    actor.userData.brainAnchor = new THREE.Vector3(...meta.brain_anchor);
    attachBrainToFly();
  } catch (error) {
    console.error('Fly specimen could not load', error);
    toast('The fly body could not load. Story playback and recorded brain activity remain available.');
  }
}

function buildNeurons(layout) {
  const points = layout.neurons.map(n => new THREE.Vector3(...n.position));
  const bbox = new THREE.Box3().setFromPoints(points), center = bbox.getCenter(new THREE.Vector3()), size = bbox.getSize(new THREE.Vector3());
  const scale = 1.6 / Math.max(size.x, size.y, size.z);
  const positions = new Float32Array(points.length * 3);
  points.forEach((p, i) => { p.sub(center).multiplyScalar(scale); positions.set([p.x, -p.y, p.z], i * 3); });
  neuronGeometry = new THREE.BufferGeometry();
  neuronGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  neuronGeometry.setAttribute('aPrevious', new THREE.BufferAttribute(new Float32Array(points.length), 1));
  neuronGeometry.setAttribute('aCurrent', new THREE.BufferAttribute(new Float32Array(points.length), 1));
  brainUniforms = { uBlend: { value: 1 }, uPointScale: { value: 1 }, uOpacity: { value: .8 } };
  const material = new THREE.ShaderMaterial({
    uniforms: brainUniforms, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
    vertexShader: `attribute float aPrevious; attribute float aCurrent; uniform float uBlend; uniform float uPointScale; varying float vActivity; void main(){ vActivity=mix(aPrevious,aCurrent,uBlend); vec4 p=modelViewMatrix*vec4(position,1.); gl_Position=projectionMatrix*p; gl_PointSize=(.95+3.6*sqrt(vActivity))*uPointScale; }`,
    fragmentShader: `precision highp float; varying float vActivity; uniform float uOpacity; void main(){ float r=length(gl_PointCoord-vec2(.5))*2.; if(r>1.)discard; float light=sqrt(vActivity); vec3 cool=vec3(.35,.76,.69); vec3 warm=vec3(1.,.74,.35); vec3 color=mix(cool,warm,smoothstep(.05,.58,light)); float alpha=exp(-r*r*3.)*(.018+light*.72)*uOpacity; gl_FragColor=vec4(color*(.65+light*.75),alpha); }`
  });
  brainInset = new THREE.Points(neuronGeometry, material); brainInset.rotation.set(.08, -.08, .02); insetScene.add(brainInset);
  brainMain = new THREE.Points(neuronGeometry, material.clone());
  brainMain.material.uniforms.uBlend = brainUniforms.uBlend;
  brainMain.material.uniforms.uPointScale.value = .72;
  brainMain.material.uniforms.uOpacity.value = .12;
  brainMain.scale.setScalar(.36); brainMain.rotation.set(0, Math.PI / 2, 0);
  attachBrainToFly();
  $('neuron-label').textContent = `${points.length.toLocaleString()} MEASURED POSITIONS`;
}
function attachBrainToFly() {
  if (!brainMain || !actor.userData.brainAnchor) return;
  brainMain.position.copy(actor.userData.brainAnchor); actor.add(brainMain);
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`${response.status}: ${url}`);
  return response.json();
}
async function loadManifest() {
  const manifestURL = new URL('/data/manifest.json', location.href);
  try {
    const manifest = await fetchJSON(manifestURL);
    if (manifest.mode !== 'recorded_replay' || !manifest.stories?.length) throw new Error('No recorded stories available');
    const layout = await fetchJSON(new URL(manifest.neuron_layout_url, manifestURL));
    if (!layout.neurons?.length) throw new Error('Neuron coordinates unavailable');
    state.manifest = manifest; state.layout = layout;
    buildNeurons(layout);
    $('story-select').replaceChildren(...manifest.stories.map((s, index) => {
      byId.set(s.id, { ...s, index, url: new URL(s.trace_url, manifestURL) });
      const option = new Option(s.title, s.id); return option;
    }));
    $('story-select').disabled = false;
    $('model-status').textContent = 'RECORDED MODEL REPLAY';
    fillProvenance(manifest, layout);
    await loadStory(manifest.stories[0].id);
  } catch (error) {
    console.info('Waiting for complete recorded data:', error.message);
    $('loading-detail').textContent = 'Waiting for the preserved model recordings. This page will load them automatically.';
    setTimeout(loadManifest, 4000);
  }
}
function fillProvenance(manifest, layout) {
  const m = manifest.model;
  $('about-model').textContent = `${m.label}. ${m.parameter_counts.total.toLocaleString()} parameters; ${m.neurons.toLocaleString()} neurons and ${m.edges.toLocaleString()} canonical connections. Encoder width ${m.encoder_width}; readout rank ${m.readout_rank}.`;
  $('about-checkpoint').textContent = `${m.selector.replaceAll('_', ' ')} checkpoint at ${m.checkpoint_updates.toLocaleString()} updates. Recorded on CPU; the full graph participates in inference. SHA-256: ${m.checkpoint_sha256}.`;
  $('about-activity').textContent = `${manifest.activity.note} We display ${layout.neurons.length.toLocaleString()} sampled neuron states, quantized as round(255 × |h|). The color and brightness use a fixed square-root curve, with interpolation between recorded frames. Each frame is the state after consuming its input token and before predicting the highlighted token. The small graph shows the sample’s mean |h|. Colors do not indicate excitation or inhibition.`;
  $('about-geometry').textContent = `${layout.neurons.length.toLocaleString()} real MaleCNS soma positions, sampled from the 44,279 neurons with measured coordinates in this 49,393-neuron model. The remaining 5,114 have no supplied soma position. The display uses translation, rotation and a uniform scale, preserving the relative geometry. The inset is an anatomical view; placement in the fly’s head is illustrative.`;
}
async function loadStory(id) {
  const request = ++loadSequence;
  pause(); state.ready = false;
  $('play-button').disabled = true; $('timeline').disabled = true; $('replay-button').disabled = true;
  $('waiting-state').hidden = false; $('loading-detail').textContent = 'Loading the recorded thought sequence.';
  $('story-text').replaceChildren(); $('story-end').hidden = true;
  const descriptor = byId.get(id);
  $('prompt-text').textContent = descriptor.prompt; $('chapter-number').textContent = String(descriptor.index + 1).padStart(2, '0');
  try {
    const trace = await fetchJSON(descriptor.url);
    if (request !== loadSequence) return;
    const count = state.layout.neurons.length;
    if (!trace.frames?.length || trace.frames.some(f => f.state_u8?.length !== count || f.state_u8.some(v => !Number.isInteger(v) || v < 0 || v > 255))) throw new Error('The neural trace does not match the measured neuron layout');
    const generation = trace.frames.filter(f => f.phase === 'generation');
    if (generation.some(f => typeof f.continuation_so_far !== 'string')) throw new Error('Missing exact decoded text alignment');
    if (generation.at(-1)?.continuation_so_far !== trace.continuation) throw new Error('The final story and decoded trace do not match');
    state.trace = trace; state.ready = true;
    trace.means = trace.frames.map(f => f.state_u8.reduce((sum, v) => sum + v, 0) / (count * 255));
    $('timeline').max = trace.frames.length - 1; $('timeline').disabled = false;
    $('play-button').disabled = false; $('replay-button').disabled = false;
    $('waiting-state').hidden = true;
    seek(0, true); $('story-scroll').scrollTop = 0;
  } catch (error) {
    console.error(error); $('loading-detail').textContent = `The recording could not be loaded: ${error.message}.`;
    toast('Unable to load this story’s trace. Choose another story or reload.');
  }
}
function setActivity(frame, immediate = false) {
  if (!neuronGeometry || !frame) return;
  const previous = neuronGeometry.getAttribute('aPrevious'), current = neuronGeometry.getAttribute('aCurrent');
  for (let i = 0; i < current.count; i++) {
    previous.array[i] = immediate ? frame.state_u8[i] / 255 : current.array[i];
    current.array[i] = frame.state_u8[i] / 255;
  }
  previous.needsUpdate = true; current.needsUpdate = true;
  transitionStart = performance.now(); brainUniforms.uBlend.value = immediate ? 1 : 0;
}
function seek(index, immediate = false) {
  if (!state.ready) return;
  state.frame = Math.max(0, Math.min(index, state.trace.frames.length - 1));
  const frame = state.trace.frames[state.frame];
  setActivity(frame, immediate);
  const previousText = state.trace.frames[Math.max(0, state.frame - 1)].continuation_so_far || '';
  const text = frame.continuation_so_far || '';
  const shared = text.startsWith(previousText) ? previousText.length : 0;
  const fragment = document.createDocumentFragment();
  fragment.append(document.createTextNode(text.slice(0, shared)));
  const highlight = document.createElement('span'); highlight.className = 'current'; highlight.textContent = text.slice(shared); fragment.append(highlight);
  if (state.frame < state.trace.frames.length - 1) { const cursor = document.createElement('span'); cursor.className = 'cursor'; fragment.append(cursor); }
  $('story-text').replaceChildren(fragment);
  $('waiting-state').hidden = true;
  $('story-end').hidden = state.frame !== state.trace.frames.length - 1;
  $('story-end').querySelector('span').textContent = state.trace.generation.stopped_on_eos ? 'THE END OF THIS STORY' : 'RECORDING PAUSED AT THE TOKEN LIMIT';
  document.querySelector('.token-lens .micro-label').textContent = frame.phase === 'prompt' ? 'READING THE PROMPT' : 'THE NEXT TOKEN';
  $('current-token').textContent = frame.phase === 'prompt' ? (frame.input_text.trim() || '〈start〉') : (frame.predicted_text.trim() || '〈end〉');
  $('token-probability').textContent = frame.phase === 'generation' ? `${(frame.probability * 100).toFixed(1)}%` : '—';
  $('token-counter').textContent = `${frame.phase === 'generation' ? frame.generation_index + 1 : 0} TOKENS`;
  $('activity-value').textContent = `${state.trace.means[state.frame].toFixed(3)} |h|`;
  $('timeline').value = state.frame;
  $('timeline').style.setProperty('--progress', `${state.frame / (state.trace.frames.length - 1) * 100}%`);
  $('frame-counter').textContent = `${String(state.frame + 1).padStart(2, '0')} / ${state.trace.frames.length}`;
  updatePlaybackLabel(); drawSpark();
  if (state.playing && frame.phase === 'generation') $('story-scroll').scrollTop = $('story-scroll').scrollHeight;
}
function drawSpark() {
  const canvas = $('activity-spark'), ctx = canvas.getContext('2d'), values = state.trace.means;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const draw = (end, color, width) => {
    ctx.beginPath(); ctx.lineWidth = width; ctx.strokeStyle = color;
    for (let i = 0; i <= end; i++) { const x = i / (values.length - 1) * 140, y = 25 - (values[i] - lo) / span * 20; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
  };
  draw(values.length - 1, '#abc79920', 1); draw(state.frame, '#c1d7a8', 1.3);
}
function updatePlaybackLabel() {
  const frame = state.trace?.frames[state.frame];
  $('play-label').textContent = state.playing ? 'Pause story' : state.ready && state.frame === state.trace.frames.length - 1 ? 'Play again' : 'Play story';
  $('play-button').querySelector('.play-symbol').textContent = state.playing ? 'Ⅱ' : '▶';
  $('play-button').setAttribute('aria-label', state.playing ? 'Pause story' : 'Play story');
  $('playback-status').textContent = !state.ready ? 'A tiny storyteller is getting ready' : frame.phase === 'prompt' ? 'Reading the beginning…' : state.frame === state.trace.frames.length - 1 ? 'A little story, told.' : state.playing ? 'One thought becomes another' : 'A thought, held for a moment';
}
function play() {
  if (!state.ready) return;
  if (state.frame === state.trace.frames.length - 1) seek(0, true);
  state.playing = true; lastStep = performance.now(); updatePlaybackLabel();
  if (state.narration) startNarration();
}
function pause() {
  state.playing = false; updatePlaybackLabel();
  if ('speechSynthesis' in window) speechSynthesis.cancel();
}
function startNarration() {
  if (!('speechSynthesis' in window) || !state.trace) return;
  speechSynthesis.cancel();
  const prefix = state.trace.frames[state.frame].continuation_so_far || '';
  const words = (state.frame === 0 ? state.trace.prompt : '') + state.trace.continuation.slice(prefix.length);
  const utterance = new SpeechSynthesisUtterance(words);
  const voices = speechSynthesis.getVoices();
  const voice = voices.find(v => /^en/.test(v.lang) && /Samantha|Karen|Daniel/.test(v.name)) || voices.find(v => /^en/.test(v.lang));
  if (voice) utterance.voice = voice;
  utterance.rate = .78 * state.speed; utterance.pitch = 1.04;
  speechSynthesis.speak(utterance);
}
function setView(view) {
  state.view = view; document.body.classList.toggle('brain-view', view === 'brain');
  $('view-fly').classList.toggle('active', view === 'fly'); $('view-brain').classList.toggle('active', view === 'brain');
  $('view-fly').setAttribute('aria-pressed', view === 'fly'); $('view-brain').setAttribute('aria-pressed', view === 'brain');
  if (view === 'brain' && actor.userData.brainAnchor) {
    const target = actor.localToWorld(actor.userData.brainAnchor.clone());
    targetGoal = target; cameraGoal = target.clone().add(new THREE.Vector3(2.7, 1.1, 3.7));
  } else { targetGoal = new THREE.Vector3(0, .5, 0); cameraGoal = new THREE.Vector3(5.6, 4, 8.7); }
  if (brainMain) {
    // A uniform enlargement preserves the measured geometry in the close-up.
    brainMain.scale.setScalar(view === 'brain' ? 1.05 : .36);
    brainMain.material.uniforms.uOpacity.value = view === 'brain' ? .8 : .12;
    brainMain.material.uniforms.uPointScale.value = view === 'brain' ? 1.35 : .72;
  }
  for (const { material, opacity, transparent, name } of flyMaterials) {
    material.transparent = view === 'brain' || transparent;
    material.opacity = view === 'brain' ? (name.includes('wing') ? .012 : .025) : opacity;
    material.depthWrite = view !== 'brain' && opacity > .8;
    material.needsUpdate = true;
  }
  flyModel?.traverse(node => { if (node.isMesh) node.castShadow = view !== 'brain' && !node.name.toLowerCase().includes('wing'); });
}
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), .05), t = clock.elapsedTime, now = performance.now();
  if (state.playing && now - lastStep >= frameDuration()) {
    lastStep = now;
    if (state.frame < state.trace.frames.length - 1) seek(state.frame + 1);
    else pause();
  }
  if (brainUniforms && brainUniforms.uBlend.value < 1) brainUniforms.uBlend.value = Math.min(1, (now - transitionStart) / Math.min(160, frameDuration() * .55));
  if (motion) {
    movementTime += reducedMotion ? dt * .2 : dt;
    if (state.motion === 'walk') walkPhase += dt * (reducedMotion ? .18 : .5);
    // The displayed path and gait use the same instantaneous physical speed.
    const pathRate = reducedMotion ? .18 : .5;
    const vx = .6 * pathRate * Math.cos(walkPhase), vz = .5 * pathRate * Math.cos(2 * walkPhase);
    const ax = -.6 * pathRate ** 2 * Math.sin(walkPhase), az = -(pathRate ** 2) * Math.sin(2 * walkPhase);
    const yawRate = -(vx * az - vz * ax) / (vx * vx + vz * vz);
    const movement = motion.update(movementTime, dt, state.motion, { walkSpeed: Math.hypot(vx, vz) / actor.scale.x, yawRate });
    hovering = THREE.MathUtils.damp(hovering, movement.flightBlend, 3.5, dt);
    actor.position.y = -.92 + hovering * .7 + (state.motion === 'fly' ? Math.sin(t * 2) * .04 * hovering : 0);
    if (state.motion === 'walk') {
      actor.position.x = .2 + Math.sin(walkPhase) * .6; actor.position.z = Math.sin(2 * walkPhase) * .25;
      const targetYaw = -Math.atan2(vz, vx), deltaYaw = Math.atan2(Math.sin(targetYaw - actor.rotation.y), Math.cos(targetYaw - actor.rotation.y));
      actor.rotation.y += deltaYaw * (1 - Math.exp(-dt * 8));
    }
    actor.rotation.z = hovering * Math.sin(t * 1.3) * .04;
  }
  if (cameraGoal && targetGoal) { camera.position.lerp(cameraGoal, .065); controls.target.lerp(targetGoal, .065); if (camera.position.distanceTo(cameraGoal) < .005) { cameraGoal = null; targetGoal = null; } }
  controls?.update();
  if (renderMain) renderMain.render(scene, camera);
  if (renderInset) renderInset.render(insetScene, insetCamera);
}

$('play-button').addEventListener('click', () => state.playing ? pause() : play());
$('replay-button').addEventListener('click', () => { pause(); seek(0, true); $('story-scroll').scrollTop = 0; play(); });
$('timeline').addEventListener('input', event => { pause(); seek(Number(event.target.value), true); $('story-scroll').scrollTop = $('story-scroll').scrollHeight; });
$('speed-select').addEventListener('change', event => { state.speed = Number(event.target.value); if (state.playing && state.narration) startNarration(); });
$('story-select').addEventListener('change', event => loadStory(event.target.value));
$('narration-button').addEventListener('click', () => {
  if (!('speechSynthesis' in window)) return toast('This browser does not support speech narration.');
  state.narration = !state.narration; $('narration-button').setAttribute('aria-pressed', state.narration); $('narration-button').setAttribute('aria-label', state.narration ? 'Disable browser narration' : 'Enable browser narration');
  if (state.playing && state.narration) startNarration(); else speechSynthesis.cancel();
});
$('about-button').addEventListener('click', () => $('about-dialog').showModal());
$('close-about').addEventListener('click', () => $('about-dialog').close());
$('about-dialog').addEventListener('click', event => { if (event.target === $('about-dialog')) $('about-dialog').close(); });
$('view-fly').addEventListener('click', () => setView('fly')); $('view-brain').addEventListener('click', () => setView('brain')); $('reset-view').addEventListener('click', () => setView(state.view));
document.addEventListener('keydown', event => { if (event.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(event.target.tagName) && !$('about-dialog').open) { event.preventDefault(); state.playing ? pause() : play(); } });
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
try { setupScene(); animate(); loadManifest(); } catch (error) { console.error(error); toast('This demo needs WebGL to show its fly and neural activity.'); }
window.__flyDemo = { state, seek, play, pause, setView, get scene() { return scene; }, get actor() { return actor; }, get motion() { return motion; } };
