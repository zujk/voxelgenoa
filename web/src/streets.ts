import * as THREE from 'three';
import type { Rect } from './hud';
import { el, fetchJson } from './util';
import type { World } from './world';

// [name, x, y (grid, metres), ground z, direction (radians, in the grid)] from build.py
type Anchor = [string, number, number, number, number];
interface Street { name: string; p: THREE.Vector3; d: THREE.Vector3; el: HTMLElement | null; w: number; occluded: boolean; on: boolean }

const NEAR = 700, MAX = 36, SAME = 260; // metres from the camera; names on screen; px between two of one street
const overlap = (r: Rect, q: Rect) => r.x0 < q.x1 + 4 && r.x1 + 4 > q.x0 && r.y0 < q.y1 + 2 && r.y1 + 2 > q.y0;

// Street names written along the street, on the ground, for the streets near the camera. A name hidden by a
// building (a voxel ray) or crowding another one stays off.
export class Streets {
  private readonly host = el('div', '', '');
  private all: Street[] = [];
  private near: Street[] = [];
  private shown = new Set<Street>();
  private last = 0;
  private frame = 0;

  constructor() { this.host.id = 'streets'; }

  async load(url: string) {
    const { streets } = await fetchJson<{ streets: Anchor[] }>(url);
    this.all = streets.map(([name, x, y, z, a]) => ({
      name, p: new THREE.Vector3(x, z + 1.2, y), d: new THREE.Vector3(Math.cos(a), 0, Math.sin(a)), el: null, w: 0, occluded: false, on: false,
    }));
  }

  // one search entry per street: the anchor nearest the middle of its anchors, and how far they spread
  places(): [string, number, number, number][] {
    const by = new Map<string, Street[]>();
    for (const s of this.all) { const a = by.get(s.name); if (a) a.push(s); else by.set(s.name, [s]); }
    return [...by].map(([name, ss]) => {
      const cx = ss.reduce((a, s) => a + s.p.x, 0) / ss.length, cz = ss.reduce((a, s) => a + s.p.z, 0) / ss.length;
      const m = ss.reduce((a, s) => (Math.hypot(s.p.x - cx, s.p.z - cz) < Math.hypot(a.p.x - cx, a.p.z - cz) ? s : a));
      const size = Math.max(40, ...ss.map((s) => Math.hypot(s.p.x - m.p.x, s.p.z - m.p.z)));
      return [name, m.p.x, m.p.z, Math.min(size, 400)];
    });
  }

  update(camera: THREE.Camera, world: World | null, obstacles: Rect[]) {
    const now = performance.now(), c = camera.position;
    if (!world) return;
    if (now - this.last > 250) { // the candidates: the nearest anchors, nearest first
      this.last = now;
      this.near = this.all.filter((s) => Math.abs(s.p.x - c.x) < NEAR && Math.abs(s.p.z - c.z) < NEAR && s.p.distanceTo(c) < NEAR)
        .sort((a, b) => a.p.distanceToSquared(c) - b.p.distanceToSquared(c)).slice(0, 160);
    }
    this.checkOcclusion(camera, world);
    const placed: Rect[] = [...obstacles], names: [string, number, number][] = [];
    const v = new THREE.Vector3(), u = new THREE.Vector3(), next = new Set<Street>();
    for (const s of this.near) {
      if (next.size >= MAX || s.occluded) continue;
      v.copy(s.p).project(camera);
      if (v.z >= 1) continue;
      const x = (v.x * 0.5 + 0.5) * innerWidth, y = (-v.y * 0.5 + 0.5) * innerHeight;
      if (x < 20 || x > innerWidth - 20 || y < 20 || y > innerHeight - 20) continue;
      if (names.some(([n, nx, ny]) => n === s.name && Math.hypot(nx - x, ny - y) < SAME)) continue;
      u.copy(s.p).addScaledVector(s.d, 8).project(camera);
      let ang = Math.atan2((-u.y * 0.5 + 0.5) * innerHeight - y, (u.x * 0.5 + 0.5) * innerWidth - x);
      if (ang > Math.PI / 2) ang -= Math.PI; else if (ang < -Math.PI / 2) ang += Math.PI; // never upside down
      if (!s.el) {
        s.el = el('div', 'st off', '', this.host);
        s.el.textContent = s.name;
        s.w = s.el.offsetWidth || s.name.length * 6;
      }
      const hw = (s.w / 2) * Math.abs(Math.cos(ang)) + 7 * Math.abs(Math.sin(ang)), hh = (s.w / 2) * Math.abs(Math.sin(ang)) + 7 * Math.abs(Math.cos(ang));
      const r = { x0: x - hw, x1: x + hw, y0: y - hh, y1: y + hh };
      if (placed.some((q) => overlap(r, q))) continue;
      placed.push(r); names.push([s.name, x, y]); next.add(s);
      s.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -50%) rotate(${ang.toFixed(3)}rad)`;
      s.el.style.setProperty('--fade', String(Math.min(1, (NEAR - s.p.distanceTo(c)) / 200).toFixed(2)));
    }
    for (const s of this.shown) if (!next.has(s)) s.el?.classList.add('off');
    for (const s of next) s.el!.classList.remove('off');
    this.shown = next;
  }

  // a few anchors per frame: a building well in front of the street hides its name; the facades lining a narrow
  // street (the caruggi) do not, or the old town would have none
  private checkOcclusion(camera: THREE.Camera, world: World) {
    const dir = new THREE.Vector3();
    for (let k = 0; k < 8 && this.near.length; k++) {
      const s = this.near[this.frame++ % this.near.length];
      dir.subVectors(s.p, camera.position);
      const dist = dir.length();
      dir.divideScalar(dist);
      s.occluded = world.raycast(camera.position, dir, dist) < dist - 20;
    }
  }
}
