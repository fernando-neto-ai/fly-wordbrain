/**
 * Anatomical fly motion adapted from Xenova Neural Canvas at pinned revision
 * 776d115ee5aa934578a87fd6d260d138084f59c1.
 * Original kinematic code MIT, Copyright (c)2024; application MIT Copyright(c)2026.
 * Full notices: public/assets/Body-MIT.txt and public/assets/Xenova-LICENSE.txt.
 * Body rig/mesh data Apache-2.0, NeuroMechFly v2 / NeLy-EPFL.
 * Source URLs and SHA256: public/assets/fly-rig.json.provenance.
 * This is an engineered illustrative gait, not a behavioral readout of the LLM.
 */
import * as THREE from 'three';

// NeuroMechFly forward kinematics. Matrices are row-major arrays,
// matching the argument order of THREE.Matrix4.set().

const DEG2RAD = Math.PI / 180;

// MuJoCo (w, x, y, z) quaternion -> 3x3 rotation (row-major, length 9).
export function quatToMatrix(q) {
  let [w, x, y, z] = q;
  const n = Math.hypot(w, x, y, z);
  if (n === 0) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  w /= n;
  x /= n;
  y /= n;
  z /= n;
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  ];
}

// Rotation of `angle` (rad) about a unit `axis` (Rodrigues), 3x3 row-major.
export function axisAngleMatrix(axis, angle) {
  const [ax, ay, az] = axis;
  const c = Math.cos(angle),
    s = Math.sin(angle),
    t = 1 - c;
  return [
    c + t * ax * ax,
    t * ax * ay - s * az,
    t * ax * az + s * ay,
    t * ax * ay + s * az,
    c + t * ay * ay,
    t * ay * az - s * ax,
    t * ax * az - s * ay,
    t * ay * az + s * ax,
    c + t * az * az,
  ];
}

// 3x3 * 3x3 (row-major).
function mat3Mul(A, B) {
  const C = new Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return C;
}

// 4x4 * 4x4 (row-major).
export function mat4Mul(A, B) {
  const C = new Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      C[i * 4 + j] =
        A[i * 4] * B[j] +
        A[i * 4 + 1] * B[4 + j] +
        A[i * 4 + 2] * B[8 + j] +
        A[i * 4 + 3] * B[12 + j];
  return C;
}

// Homogeneous 4x4 from a 3x3 rotation (length 9) and a translation (length 3).
function homogeneous(R, p) {
  return [R[0], R[1], R[2], p[0], R[3], R[4], R[5], p[1], R[6], R[7], R[8], p[2], 0, 0, 0, 1];
}

// Apply a 4x4 to a 3-vector (treated as a point, w=1) -> length-3 array.
export function mat4ApplyPoint(M, v) {
  return [
    M[0] * v[0] + M[1] * v[1] + M[2] * v[2] + M[3],
    M[4] * v[0] + M[5] * v[1] + M[6] * v[2] + M[7],
    M[8] * v[0] + M[9] * v[1] + M[10] * v[2] + M[11],
  ];
}

// Cache each joint’s axes in model.dofs order; rotation order matters.
function jointDofMap(model) {
  if (model.__jointDofs) return model.__jointDofs;
  const map = {};
  for (const d of model.dofs) {
    const key = `${d.parent}>${d.child}`;
    const axis = Array.isArray(d.axis) ? d.axis : model.axisVector[d.axis];
    (map[key] || (map[key] = [])).push({ name: d.name, axis });
  }
  Object.defineProperty(model, '__jointDofs', { value: map, enumerable: false });
  return map;
}

// Compose hinge rotations in joint order; angles are in radians.
function jointRotation(jointDofs, anglesRad) {
  let rot = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (const { name, axis } of jointDofs) {
    const angle = anglesRad[name] || 0;
    if (angle) rot = mat3Mul(rot, axisAngleMatrix(axis, angle));
  }
  return rot;
}

// Forward kinematics: world (body-frame) 4x4 transform for every segment.
// `anglesRad` maps DOF name -> radians; missing DOFs are 0.
export function segmentTransforms(model, anglesRad) {
  const T = {};
  const dofMap = jointDofMap(model);
  const root = model.rest[model.root];
  T[model.root] = homogeneous(quatToMatrix(root.quat), root.pos);
  for (const [parent, child] of model.joints) {
    const cfg = model.rest[child];
    const rest = homogeneous(quatToMatrix(cfg.quat), cfg.pos);
    const jdofs = dofMap[`${parent}>${child}`] || [];
    const joint = homogeneous(jointRotation(jdofs, anglesRad), [0, 0, 0]);
    T[child] = mat4Mul(mat4Mul(T[parent], rest), joint);
  }
  return T;
}

// World-frame origin (translation) of each segment's joint, {segment: [x,y,z]}.
export function jointPositions(model, anglesRad) {
  const T = segmentTransforms(model, anglesRad);
  const out = {};
  for (const s of model.segments) out[s] = [T[s][3], T[s][7], T[s][11]];
  return out;
}

// A {dof: radians} pose from the model's neutral pose (degrees).
export function neutralPoseRad(model) {
  const a = {};
  for (const d of model.dofs) a[d.name] = (model.neutralDeg[d.name] || 0) * DEG2RAD;
  return a;
}

export function zeroPoseRad(model) {
  const a = {};
  for (const d of model.dofs) a[d.name] = 0;
  return a;
}

// Maximum coordinate error against the bundled neutral-pose reference.
export function validateFK(model) {
  const got = jointPositions(model, neutralPoseRad(model));
  const ref = model.reference.neutralJointPositions;
  let maxErr = 0;
  for (const s of model.segments) {
    for (let i = 0; i < 3; i++) maxErr = Math.max(maxErr, Math.abs(got[s][i] - ref[s][i]));
  }
  return maxErr;
}

// Joint-space envelope measured against the supplied thorax/abdomen meshes.
// Original hinges and meshes are preserved. Angles are in model joint axes.
export const WINGBEAT_HZ = 277;
const clamp = (x) => Math.max(0, Math.min(1, x));
export function wingPose(state, phaseOffset = 0) {
  const extension = clamp(state.wingOpen || 0);
  const amplitude = clamp((extension - 0.7) / 0.3);
  const phase = (state.time || 0) * Math.PI * 2 * WINGBEAT_HZ + phaseOffset;
  const pose = {};
  for (const side of ['l', 'r']) {
    const sign = side === 'l' ? 1 : -1,
      prefix = `c_thorax-${side}_wing-`;
    pose[prefix + 'pitch'] = ((30 - 20 * extension) * Math.PI) / 180;
    pose[prefix + 'roll'] = (-75 * extension * sign * Math.PI) / 180;
    pose[prefix + 'yaw'] =
      ((5 + 15 * extension + 35 * amplitude * Math.sin(phase)) * sign * Math.PI) / 180;
  }
  return pose;
}

export const LEGS = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh'];
/** Damped Jacobian IK with a standing-posture constraint. */
export class Gait {
  constructor(model) {
    this.model = model;
    this.pose = neutralPoseRad(model);
    this.wingModel = {
      ...model,
      segments: [model.root, 'l_wing', 'r_wing'],
      joints: model.joints.filter(([, s]) => s.endsWith('_wing')),
      dofs: model.dofs.filter((d) => d.child.endsWith('_wing')),
    };
    this.legs = LEGS.map((name, i) => {
      const segments = model.segments.filter((s) => s === model.root || s.startsWith(name + '_'));
      const local = {
        ...model,
        segments,
        joints: model.joints.filter(([, s]) => s.startsWith(name + '_')),
        dofs: model.dofs.filter((d) => d.child.startsWith(name + '_')),
      };
      const dofs = local.dofs.filter(
        (d) => d.limitDeg && (!d.child.includes('tarsus') || d.child.endsWith('tarsus1')),
      );
      return {
        name,
        i,
        local,
        dofs,
        tip: name + '_tarsus5',
        neutral: model.reference.neutralJointPositions[name + '_tarsus5'],
      };
    });
    this.update({ phase: 0, velocity: 0, yawRate: 0, y: 0 });
    // Calibrate the standing posture once from the measured skeleton and foot height.
    this.referencePose = { ...this.pose };
  }
  point(leg) {
    const t = segmentTransforms(leg.local, this.pose)[leg.tip];
    return [t[3], t[7], t[11]];
  }
  solve(leg, target, iterations = 10) {
    for (let it = 0; it < iterations; it++) {
      const p = this.point(leg),
        error = target.map((v, k) => v - p[k]);
      // Stop only when both foot position and joint posture have converged.
      if (
        Math.hypot(...error) < 0.001 &&
        (it > 0 ||
          !this.referencePose ||
          leg.dofs.every((d) => Math.abs(this.pose[d.name] - this.referencePose[d.name]) < 0.001))
      )
        break;
      const eps = 0.001,
        J = leg.dofs.map((d) => {
          this.pose[d.name] += eps;
          const q = this.point(leg);
          this.pose[d.name] -= eps;
          return q.map((v, k) => (v - p[k]) / eps);
        });
      // A foot constrains three of seven DOFs. Project the standing-posture
      // correction into the null space once per update to prevent joint drift.
      const preferred = leg.dofs.map((d) =>
        it === 0 && this.referencePose
          ? this.postureGain * (this.referencePose[d.name] - this.pose[d.name])
          : 0,
      );
      for (let k = 0; k < 3; k++)
        error[k] -= J.reduce((sum, col, j) => sum + col[k] * preferred[j], 0);
      const A = [
        [0.001, 0, 0],
        [0, 0.001, 0],
        [0, 0, 0.001],
      ];
      for (const col of J)
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) A[r][c] += col[r] * col[c];
      // Pivoted 3×3 solve, A u = error − J preferred.
      // Joint update = Jᵀ u + preferred, including the projected posture task.
      const mat = A.map((row, r) => [...row, error[r]]);
      for (let k = 0; k < 3; k++) {
        let p = k;
        for (let r = k + 1; r < 3; r++) if (Math.abs(mat[r][k]) > Math.abs(mat[p][k])) p = r;
        [mat[k], mat[p]] = [mat[p], mat[k]];
        const divisor = mat[k][k];
        for (let c = k; c < 4; c++) mat[k][c] /= divisor;
        for (let r = 0; r < 3; r++)
          if (r !== k) {
            const f = mat[r][k];
            for (let c = k; c < 4; c++) mat[r][c] -= f * mat[k][c];
          }
      }
      const u = mat.map((r) => r[3]);
      leg.dofs.forEach((d, j) => {
        const delta = preferred[j] + J[j].reduce((s, v, k) => s + v * u[k], 0);
        const [lo, hi] = d.limitDeg.map((v) => (v * Math.PI) / 180);
        this.pose[d.name] = Math.max(
          lo,
          Math.min(hi, this.pose[d.name] + Math.max(-0.15, Math.min(0.15, delta))),
        );
      });
    }
    return Math.hypot(...this.point(leg).map((v, k) => v - target[k]));
  }
  update(state) {
    const elapsed =
      Number.isFinite(state.time) && Number.isFinite(this.lastTime)
        ? Math.max(0, Math.min(0.05, state.time - this.lastTime))
        : 0.01;
    this.postureGain = 1 - Math.exp(-elapsed / 0.12);
    this.lastTime = state.time;
    const v = state.velocity || 0,
      turn = state.yawRate || 0,
      speed = Math.abs(v) + Math.abs(turn) * 0.5;
    const frequency = Math.max(0.01, Math.min(14, speed / 1.4));
    this.errors = [];
    this.targets = [];
    for (const leg of this.legs) {
      const side = leg.i < 3 ? 1 : -1;
      const phase = (state.phase / (Math.PI * 2) + ([0, 2, 4].includes(leg.i) ? 0 : 0.5)) % 1;
      const duty = 0.65,
        stride = Math.max(
          -1,
          Math.min(1, ((v - side * turn * Math.abs(leg.neutral[1])) / frequency) * duty),
        );
      const swing = Math.max(0, (phase - duty) / (1 - duty));
      const x =
        phase < duty
          ? stride * (0.5 - phase / duty)
          : stride * (-0.5 + swing - Math.sin(swing * 2 * Math.PI) / (2 * Math.PI));
      const lift = Math.sin(swing * Math.PI) * 0.35 * Math.min(1, speed / 0.2);
      const flight = Math.max(0, Math.min(1, state.flightBlend || 0));
      const tuck = flight * (1 - 0.8 * (state.landing || 0));
      const target = [
        leg.neutral[0] + x * (1 - flight) - 0.12 * tuck,
        leg.neutral[1] * (1 - 0.18 * tuck),
        0.11 + lift * (1 - flight) + 0.48 * tuck,
      ];
      this.errors.push(this.solve(leg, target));
      this.targets.push(target);
    }
    Object.assign(this.pose, wingPose(state));
    return segmentTransforms(this.model, this.pose);
  }
  wingTransforms(state, phaseOffset = 0) {
    return segmentTransforms(this.wingModel, wingPose(state, phaseOffset));
  }
}

/** Animate the neutral-world-baked fly GLB, leaving its parent actor untouched.
 * Call update(timeSeconds,dtSeconds,'idle'|'walk'|'fly', options) per frame.
 * options.walkSpeed is nonnegative mm/s; options.yawRate is signed radians/s.
 * groundY is the standing floor height in original GLB units, for outer staging.
 */
export function attachFlyMotion(root, rig) {
  if (rig?.format_version !== 1 || !rig.model || !rig.neutralTransforms) {
    throw new Error('Missing measured fly rig');
  }
  const rotation = new THREE.Matrix4().set(...rig.bodyToGltf);
  const inverseRotation = rotation.clone().invert();
  const neutral = {};
  for (const [name, elements] of Object.entries(rig.neutralTransforms)) {
    neutral[name] = new THREE.Matrix4().set(...elements).invert();
  }
  const meshes = [];
  root.traverse(mesh => {
    if (mesh.isMesh && neutral[mesh.name]) {
      mesh.matrixAutoUpdate = false;
      mesh.geometry.computeBoundingBox();
      meshes.push(mesh);
    }
  });
  if (meshes.length !== Object.keys(rig.model.meshes).length) {
    throw new Error(`Fly rig/GLB mismatch: found ${meshes.length} measured meshes`);
  }
  const gait = new Gait(rig.model);
  let velocity=0, phase=0, flightBlend=0, wingOpen=0;
  let lastTime=null;
  const state = {time:0,velocity:0,yawRate:0,phase:0,flightBlend:0,wingOpen:0,landing:0};
  const current = new THREE.Matrix4();
  const box = new THREE.Box3();
  const footNames = new Set(LEGS.map(leg => leg + '_tarsus5'));
  function apply(pose) {
    const transforms = gait.update(pose);
    let floor = Infinity;
    for (const mesh of meshes) {
      // Mesh vertices already contain neutral world placement. Undo neutral
      // body placement, apply current measured joints, and rotate back to glTF.
      current.set(...transforms[mesh.name]);
      mesh.matrix.copy(rotation).multiply(current).multiply(neutral[mesh.name]).multiply(inverseRotation);
      mesh.matrixWorldNeedsUpdate = true;
      if (footNames.has(mesh.name)) {
        box.copy(mesh.geometry.boundingBox).applyMatrix4(mesh.matrix);
        floor = Math.min(floor, box.min.y);
      }
    }
    root.updateMatrixWorld(true);
    return floor;
  }
  const groundY = apply(state);
  return {
    groundY,
    mode:'idle',
    update(time, dt, mode='idle', options={}) {
      if (!['idle','walk','fly'].includes(mode)) throw new Error('Unknown fly motion mode: '+mode);
      if (!Number.isFinite(time) || !Number.isFinite(dt) || dt<0) throw new Error('Invalid fly animation time');
      const walkSpeed = options.walkSpeed ?? 4.2;
      const requestedYawRate = options.yawRate ?? 0;
      if (!Number.isFinite(walkSpeed) || walkSpeed < 0 || !Number.isFinite(requestedYawRate)) {
        throw new Error('Fly motion requires finite nonnegative walkSpeed and finite yawRate');
      }
      // Duplicate timestamps freeze the pose, matching a paused visual clock.
      const elapsed = lastTime===time ? 0 : Math.min(0.05,dt);
      lastTime=time;
      const smooth=(value,target,tau)=>value+(target-value)*(1-Math.exp(-elapsed/tau));
      const flying=mode==='fly';
      velocity=smooth(velocity,mode==='walk'?walkSpeed:0,0.12);
      flightBlend=smooth(flightBlend,flying?1:0,0.16);
      wingOpen=smooth(wingOpen,flying?1:0,flying?0.045:0.09);
      if(Math.abs(velocity)<0.001)velocity=0;
      if(flightBlend<0.001&&!flying)flightBlend=0;
      if(wingOpen<0.001&&!flying)wingOpen=0;
      const yawRate = mode==='walk' ? requestedYawRate : 0;
      const gaitSpeed = Math.abs(velocity) + Math.abs(yawRate)*0.5;
      phase+=Math.min(14,gaitSpeed/1.4)*Math.PI*2*elapsed;
      Object.assign(state,{time,velocity,yawRate,phase,flightBlend,wingOpen,landing:!flying&&flightBlend>0?1:0});
      const currentGroundY=apply(state);
      this.mode=mode;
      return {...state,currentGroundY,groundY,maxFootTargetError:Math.max(...gait.errors)};
    },
  };
}
