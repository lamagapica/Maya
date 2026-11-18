"""Path tracer CPU (NumPy, vectorizado por rayos). Sin dependencias de Qt.

Soporta: luces puntuales y direccionales con sombras duras, reflejos (metálico /
rugosidad GGX aproximada), oclusión ambiental (rayos de hemisferio coseno),
iluminación de entorno (HDRI/cielo), texturas de albedo y normales interpoladas.
Intersección: Möller–Trumbore por bloques, con descarte por AABB de cada objeto.
Pensado para escenas de baja/media resolución (<~5k triángulos); el siguiente
paso natural es un BVH o mover el bucle a un compute shader.
"""
import numpy as np
from PIL import Image
from ..core.mathutils import normalize


def aces(x):
    x = x * 0.6
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def to_srgb8(lin):
    return (aces(lin) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


class PathTracer:
    RAY_CHUNK, TRI_BLOCK = 4096, 128

    def __init__(self, scene, camera, env, settings):
        self.s, self.env, self.cam = settings, env, camera
        self._build(scene)

    # ------------------------------------------------------------ escena
    def _build(self, scene):
        v0, e1, e2, nrm, uvs, tmat, objs = [], [], [], [], [], [], []
        mids = {mid: i for i, mid in enumerate(scene.materials)}
        n0 = 0
        for node in scene.nodes.values():
            if node.kind != "mesh" or not node.mesh.f or not scene.is_visible(node):
                continue
            M = scene.world_matrix(node)
            a = node.mesh.render_arrays()
            p = (a["pos"].astype(float) @ M[:3, :3].T + M[:3, 3]).reshape(-1, 3, 3)
            nm = normalize(a["nrm"].astype(float) @ np.linalg.inv(M[:3, :3])).reshape(-1, 3, 3)
            v0.append(p[:, 0]); e1.append(p[:, 1] - p[:, 0]); e2.append(p[:, 2] - p[:, 0])
            nrm.append(nm); uvs.append(a["uv"].astype(float).reshape(-1, 3, 2))
            tmat.append(np.full(len(p), mids.get(node.material, 0)))
            objs.append((n0, n0 + len(p), p.reshape(-1, 3).min(0) - 1e-4, p.reshape(-1, 3).max(0) + 1e-4))
            n0 += len(p)
        z = lambda *s: np.zeros(s)
        self.v0 = np.vstack(v0) if v0 else z(0, 3)
        self.e1 = np.vstack(e1) if e1 else z(0, 3)
        self.e2 = np.vstack(e2) if e2 else z(0, 3)
        self.nrm = np.concatenate(nrm) if nrm else z(0, 3, 3)
        self.uv = np.concatenate(uvs) if uvs else z(0, 3, 2)
        self.tmat = np.concatenate(tmat) if tmat else np.zeros(0, int)
        self.objs = objs
        mats = list(scene.materials.values())
        self.alb = np.array([np.array(m.albedo) ** 2.2 for m in mats])
        self.rough = np.array([max(m.roughness, 0.04) for m in mats])
        self.metal = np.array([m.metallic for m in mats])
        self.tex = {}
        for i, m in enumerate(mats):
            if m.albedo_tex:
                try:
                    self.tex[i] = (np.asarray(Image.open(m.albedo_tex).convert("RGB"), np.float32) / 255.0) ** 2.2
                except OSError:
                    pass
        self.lights = []
        for node in scene.nodes.values():
            if node.kind == "light" and scene.is_visible(node):
                M = scene.world_matrix(node)
                col = np.array(node.light["color"]) * node.light["intensity"]
                self.lights.append(dict(dir=node.light["type"] == "directional", pos=M[:3, 3].copy(),
                                        d=node.light_dir(M), col=col))
        if not self.lights:
            self.lights.append(dict(dir=True, pos=np.zeros(3), d=normalize([-0.4, -1.0, -0.6]), col=np.array([3.0, 3.0, 3.0])))

    # ------------------------------------------------------------ intersección
    def _block(self, o, d, s, e):
        ox, oy, oz = o[:, 0:1], o[:, 1:2], o[:, 2:3]
        dx, dy, dz = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        v0, e1, e2 = self.v0[s:e], self.e1[s:e], self.e2[s:e]
        a = [e1[:, k][None] for k in range(3)]; b = [e2[:, k][None] for k in range(3)]; c = [v0[:, k][None] for k in range(3)]
        px, py, pz = dy * b[2] - dz * b[1], dz * b[0] - dx * b[2], dx * b[1] - dy * b[0]
        det = a[0] * px + a[1] * py + a[2] * pz
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        tx, ty, tz = ox - c[0], oy - c[1], oz - c[2]
        u = (tx * px + ty * py + tz * pz) * inv
        qx, qy, qz = ty * a[2] - tz * a[1], tz * a[0] - tx * a[2], tx * a[1] - ty * a[0]
        v = (dx * qx + dy * qy + dz * qz) * inv
        t = (b[0] * qx + b[1] * qy + b[2] * qz) * inv
        good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
        return np.where(good, t, np.inf), u, v

    def intersect(self, O, D, tmax=np.inf):
        R = len(O)
        best = np.full(R, np.inf) if np.isscalar(tmax) else np.asarray(tmax, float).copy()
        tri = -np.ones(R, int); bu = np.zeros(R); bv = np.zeros(R)
        if R == 0:
            return best, tri, bu, bv
        inv = 1.0 / np.where(np.abs(D) < 1e-12, 1e-12, D)
        for (s0, s1, lo, hi) in self.objs:
            t0, t1 = (lo - O) * inv, (hi - O) * inv
            tn, tf = np.minimum(t0, t1).max(1), np.maximum(t0, t1).min(1)
            cand = np.nonzero((tf >= np.maximum(tn, 0)) & (tn < best))[0]
            for c in range(0, len(cand), self.RAY_CHUNK):
                ids = cand[c:c + self.RAY_CHUNK]
                for s in range(s0, s1, self.TRI_BLOCK):
                    t, u, v = self._block(O[ids], D[ids], s, min(s + self.TRI_BLOCK, s1))
                    k = t.argmin(1); r = np.arange(len(ids)); tm = t[r, k]
                    upd = tm < best[ids]
                    w = ids[upd]
                    best[w], tri[w], bu[w], bv[w] = tm[upd], s + k[upd], u[r, k][upd], v[r, k][upd]
        return best, tri, bu, bv

    def occluded(self, O, D, tmax):
        return self.intersect(O, D, tmax)[1] >= 0

    # ------------------------------------------------------------ muestreo
    @staticmethod
    def _cos_hemisphere(N, rng):
        r1, r2 = rng.random(len(N)), rng.random(len(N))
        x, y, z = np.sqrt(r1) * np.cos(2 * np.pi * r2), np.sqrt(r1) * np.sin(2 * np.pi * r2), np.sqrt(1 - r1)
        a = np.where((np.abs(N[:, 0]) > 0.9)[:, None], [[0, 1, 0]], [[1, 0, 0]])
        T = normalize(np.cross(a, N)); B = np.cross(N, T)
        return normalize(T * x[:, None] + B * y[:, None] + N * z[:, None])

    def _primary(self, rng):
        W, H = self.s.width, self.s.height
        right, up, fwd = self.cam.basis()
        tf = np.tan(np.radians(self.cam.fov) / 2)
        xs = (np.arange(W)[None, :] + rng.random((H, W))) / W
        ys = (np.arange(H)[:, None] + rng.random((H, W))) / H
        px, py = (xs * 2 - 1) * tf * W / H, (1 - ys * 2) * tf
        d = fwd + right * px[..., None] + up * py[..., None]
        return normalize(d.reshape(-1, 3))

    def trace(self, rng):
        s = self.s
        D = self._primary(rng)
        N0 = len(D)
        O = np.repeat(self.cam.eye[None], N0, 0)
        thr, col = np.ones((N0, 3)), np.zeros((N0, 3))
        idx = np.arange(N0)
        for bounce in range(s.bounces + 1):
            if len(idx) == 0:
                break
            t, tri, u, v = self.intersect(O[idx], D[idx])
            hit = tri >= 0
            mi = idx[~hit]
            if len(mi):
                col[mi] += thr[mi] * self.env.sample(D[mi]) * s.env_intensity
            hi, tri, u, v, t = idx[hit], tri[hit], u[hit], v[hit], t[hit]
            if len(hi) == 0:
                break
            Dh = D[hi]
            P = O[hi] + Dh * t[:, None]
            w = 1 - u - v
            N = normalize(w[:, None] * self.nrm[tri, 0] + u[:, None] * self.nrm[tri, 1] + v[:, None] * self.nrm[tri, 2])
            N = np.where(((N * Dh).sum(1) > 0)[:, None], -N, N)
            uv = w[:, None] * self.uv[tri, 0] + u[:, None] * self.uv[tri, 1] + v[:, None] * self.uv[tri, 2]
            mat = self.tmat[tri]
            alb = self.alb[mat].copy()
            for mi_, img in self.tex.items():
                sel = mat == mi_
                if sel.any():
                    h, wd, _ = img.shape
                    xi = (np.floor(uv[sel, 0] * wd).astype(int)) % wd
                    yi = (h - 1 - np.floor(uv[sel, 1] * h).astype(int) % h)
                    alb[sel] *= img[yi, xi]
            rough, metal = self.rough[mat], self.metal[mat]
            F0 = 0.04 * (1 - metal)[:, None] + alb * metal[:, None]
            V = -Dh
            NdV = np.clip((N * V).sum(1), 1e-4, 1)
            Pb = P + N * 2e-3
            rad = np.zeros((len(hi), 3))
            a2 = (rough ** 2) ** 2
            for L in self.lights:
                if L["dir"]:
                    Ld = np.repeat(-L["d"][None], len(hi), 0); dist = np.full(len(hi), np.inf); att = np.ones(len(hi))
                else:
                    dv = L["pos"] - P; dist = np.linalg.norm(dv, axis=1); Ld = dv / dist[:, None]; att = 1 / (dist ** 2 + 1e-4)
                NdL = np.clip((N * Ld).sum(1), 0, 1)
                m = np.nonzero(NdL > 0)[0]
                if len(m) == 0:
                    continue
                m = m[~self.occluded(Pb[m], Ld[m], dist[m])]
                if len(m) == 0:
                    continue
                H = normalize(V[m] + Ld[m]); NdH = np.clip((N[m] * H).sum(1), 0, 1); VdH = np.clip((V[m] * H).sum(1), 0, 1)
                D_ = a2[m] / (np.pi * (NdH ** 2 * (a2[m] - 1) + 1) ** 2)
                k = (rough[m] + 1) ** 2 / 8
                G = (NdV[m] / (NdV[m] * (1 - k) + k)) * (NdL[m] / (NdL[m] * (1 - k) + k))
                F = F0[m] + (1 - F0[m]) * ((1 - VdH) ** 5)[:, None]
                spec = (D_ * G / (4 * NdV[m] * NdL[m] + 1e-4))[:, None] * F
                diff = alb[m] * (1 - metal[m])[:, None] / np.pi
                rad[m] += (diff + spec) * L["col"] * (att[m] * NdL[m])[:, None]
            # ambiente + oclusión ambiental (1 rayo por muestra; converge al acumular)
            ao_d = self._cos_hemisphere(N, rng)
            ao = 1.0 - s.ao_strength * self.occluded(Pb, ao_d, s.ao_radius).astype(float)
            amb = self.env.irradiance(N) * s.env_intensity * alb * (1 - metal)[:, None] * ao[:, None]
            col[hi] += thr[hi] * (rad + amb)
            # reflexión
            Fr = F0 + (1 - F0) * ((1 - NdV) ** 5)[:, None]
            Rr = Dh - 2 * (Dh * N).sum(1, keepdims=True) * N
            Rr = normalize(Rr + (rough ** 2)[:, None] * normalize(rng.normal(size=Rr.shape)))
            below = (Rr * N).sum(1) < 0
            Rr[below] -= 2 * (Rr[below] * N[below]).sum(1, keepdims=True) * N[below]
            nt = thr[hi] * Fr
            keep = nt.max(1) > 0.03
            idx = hi[keep]
            O[idx], D[idx], thr[idx] = Pb[keep], normalize(Rr[keep]), nt[keep]
        return col

    def render(self, progress=None, cancelled=None):
        rng = np.random.default_rng(7)
        acc = np.zeros((self.s.width * self.s.height, 3))
        img = None
        for i in range(self.s.spp):
            if cancelled and cancelled():
                break
            acc += self.trace(rng)
            img = to_srgb8(acc / (i + 1)).reshape(self.s.height, self.s.width, 3)
            if progress:
                progress(i + 1, img)
        return img
