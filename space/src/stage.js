// The performance stage: one fruit fly, one microphone, one slow camera orbit.
//
// The body is a measured NeuroMechFly rig driven by the same forward kinematics the
// recorded demo used. Its movement is stagecraft — the language model does not control
// it, and the demo says so on the page.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { attachFlyMotion } from './fly-motion.js';

const BODY_TINT = 0xbfa57f;

export class Stage {
  constructor(element) {
    this.element = element;
    this.orbiting = true;
    this.speaking = 0;
    this.emphasis = 0;
    this.clock = new THREE.Clock();
    this.orbitAngle = 0.62;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(0x050507, 1);
    element.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x05050a, 0.045);
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.02, 120);
    this.target = new THREE.Vector3(0.55, 1.0, 0);

    this.#lights();
    this.#floor();

    this.actor = new THREE.Group();
    this.actor.position.set(-0.55, 0, 0);
    this.actor.scale.setScalar(1.35);
    this.scene.add(this.actor);

    this.microphone = this.#microphone();
    this.scene.add(this.microphone);

    this.#resize();
    window.addEventListener('resize', () => this.#resize());
  }

  #lights() {
    this.scene.add(new THREE.HemisphereLight(0x33506a, 0x0a0a10, 0.55));

    const key = new THREE.SpotLight(0xffe6c4, 42, 22, Math.PI / 7, 0.45, 1.6);
    key.position.set(3.4, 6.2, 3.1);
    key.target.position.set(0.2, 0.4, 0);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.bias = -0.0022;
    this.scene.add(key, key.target);

    const rim = new THREE.SpotLight(0x67d8ff, 26, 20, Math.PI / 6, 0.6, 1.5);
    rim.position.set(-4.6, 3.4, -3.8);
    rim.target.position.set(0, 0.5, 0);
    this.scene.add(rim, rim.target);

    const blush = new THREE.PointLight(0xe2589c, 14, 11);
    blush.position.set(-1.4, 2.3, -3.6);
    this.scene.add(blush);

    this.practical = new THREE.PointLight(0xff7a5a, 2.2, 4.5);
    this.practical.position.set(1.95, 1.2, 0);
    this.scene.add(this.practical);
  }

  #floor() {
    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(9, 96),
      new THREE.MeshStandardMaterial({ color: 0x0c0c12, roughness: 0.42, metalness: 0.32 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    for (const radius of [1.5, 2.4, 3.6]) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(radius, radius + 0.008, 128),
        new THREE.MeshBasicMaterial({ color: 0x1d3b44, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.002;
      this.scene.add(ring);
    }
  }

  #microphone() {
    const group = new THREE.Group();
    const metal = new THREE.MeshStandardMaterial({ color: 0x2b2d33, roughness: 0.32, metalness: 0.95 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x111117, roughness: 0.55, metalness: 0.7 });

    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.5, 0.09, 40), dark);
    base.position.y = 0.045;
    base.castShadow = true;
    group.add(base);

    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.07, 1.15, 24), metal);
    post.position.y = 0.66;
    post.castShadow = true;
    group.add(post);

    const yoke = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.028, 12, 40, Math.PI), metal);
    yoke.position.y = 1.32;
    yoke.rotation.y = Math.PI / 2;
    group.add(yoke);

    const head = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.235, 0.16, 12, 28),
      new THREE.MeshStandardMaterial({ color: 0x3a3d45, roughness: 0.28, metalness: 1, envMapIntensity: 1.3 }),
    );
    head.position.y = 1.36;
    head.rotation.z = Math.PI / 2;
    head.castShadow = true;
    group.add(head);

    // A suggestion of a grille, so the head reads as a microphone rather than a pill.
    const grille = new THREE.Mesh(
      new THREE.TorusGeometry(0.245, 0.006, 8, 34),
      new THREE.MeshStandardMaterial({ color: 0x8b8f99, roughness: 0.3, metalness: 1 }),
    );
    grille.position.y = 1.36;
    grille.rotation.y = Math.PI / 2;
    group.add(grille);
    for (let i = 0; i < 4; i += 1) {
      const band = grille.clone();
      band.position.x = -0.09 + i * 0.06;
      group.add(band);
    }

    this.tally = new THREE.Mesh(
      new THREE.TorusGeometry(0.1, 0.018, 10, 28),
      new THREE.MeshBasicMaterial({ color: 0xff4b39 }),
    );
    this.tally.position.set(0, 1.03, 0);
    this.tally.rotation.x = Math.PI / 2;
    this.tally.material.opacity = 0.25;
    this.tally.material.transparent = true;
    group.add(this.tally);

    group.position.set(1.72, 0, 0);
    group.scale.setScalar(1.34);
    return group;
  }

  async load() {
    const [gltf, meta, rig] = await Promise.all([
      new GLTFLoader().loadAsync('assets/fly.glb'),
      fetch('assets/fly-meta.json').then((r) => r.json()),
      fetch('assets/fly-rig.json').then((r) => r.json()),
    ]);
    const root = gltf.scene;
    root.traverse((node) => {
      if (!node.isMesh) return;
      node.castShadow = true;
      node.receiveShadow = true;
      node.material = node.material.clone();
      node.material.color.multiply(new THREE.Color(BODY_TINT));
      node.material.roughness = Math.min(1, (node.material.roughness ?? 0.7) * 0.85);
    });
    this.motion = attachFlyMotion(root, rig);
    this.actor.add(root);
    this.actor.position.y = -this.motion.groundY * this.actor.scale.y;
    this.meta = meta;
    // Face the microphone: the rig's forward axis is +X and the mic stands at +X.
    this.actor.rotation.y = 0;
    return this;
  }

  setOrbiting(on) { this.orbiting = on; }

  /** Called on every produced token so the body reacts to the actual recital. */
  pulse(strength = 1) { this.emphasis = Math.min(1.4, this.emphasis + 0.8 * strength); }

  setSpeaking(on) { this.speakingTarget = on ? 1 : 0; }

  #resize() {
    const width = this.element.clientWidth;
    const height = this.element.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  render() {
    const dt = Math.min(0.05, this.clock.getDelta());
    const time = this.clock.elapsedTime;
    this.#resize();

    this.speaking += ((this.speakingTarget ?? 0) - this.speaking) * (1 - Math.exp(-dt / 0.18));
    this.emphasis *= Math.exp(-dt / 0.22);

    if (this.motion) this.motion.update(time, dt, 'idle');

    // Recital body language: a lean toward the microphone, a breath-rate bob, and a
    // kick on every token the model actually produced.
    const lean = this.speaking * 0.18;
    // Each word boundary lands as a visible dip-and-rise, the way a speaker punches a
    // syllable, rather than a subliminal nudge.
    const bob = Math.sin(time * 2.1) * 0.02 + this.emphasis * 0.13;
    const sway = Math.sin(time * 0.8) * 0.035 * (0.3 + this.speaking);
    this.actor.position.y = -this.motion?.groundY * this.actor.scale.y + bob;
    this.actor.rotation.z = -lean * 0.4 - this.emphasis * 0.14;
    this.actor.rotation.y = sway + this.emphasis * 0.04;
    this.actor.position.x = -0.55 + this.speaking * 0.12 + this.emphasis * 0.05;

    const glow = 0.2 + this.speaking * 0.6 + this.emphasis * 0.5;
    this.tally.material.opacity = Math.min(1, glow);
    this.tally.scale.setScalar(1 + this.emphasis * 0.3);
    this.practical.intensity = 1.4 + this.speaking * 2.6 + this.emphasis * 3.2;

    if (this.orbiting) this.orbitAngle += dt * 0.12;
    // Keep the fly and the microphone both in shot however wide the stage is: a short,
    // wide panel needs more distance than a tall one to fit the same subject.
    const vertical = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const fitHeight = 3.2 / Math.tan(vertical);
    const fitWidth = 5.6 / (Math.tan(vertical) * Math.max(0.6, this.camera.aspect));
    const radius = Math.max(fitHeight, fitWidth) * 0.92 - this.speaking * 0.35;
    const height = 1.9 + Math.sin(this.orbitAngle * 0.6) * 0.45;
    this.camera.position.set(
      Math.cos(this.orbitAngle) * radius + 0.5,
      height,
      Math.sin(this.orbitAngle) * radius,
    );
    this.camera.lookAt(this.target);
    this.renderer.render(this.scene, this.camera);
  }
}
