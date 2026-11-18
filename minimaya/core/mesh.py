"""Malla poligonal editable.

Representación: vértices (N,3) + caras como listas de índices (n-gonos, orden CCW).
La conectividad tipo half-edge se deriva bajo demanda: cada cara aporta las
semiaristas dirigidas (a->b); su gemela es (b->a). `half_edges()` y
`edge_faces()` dan esa adyacencia en O(F).
"""
import itertools
from collections import defaultdict
import numpy as np
from .mathutils import normalize

_VERSION = itertools.count(1)


class Mesh:
    def __init__(self, vertices, faces):
        self.v = np.array(vertices, dtype=float).reshape(-1, 3)
        self.f = [[int(i) for i in f] for f in faces]
        self.smooth = False
        self.uv_mode = "box"          # 'box' (cúbica) | 'planar'
        self.uv_axis = "y"            # eje de proyección planar
        self.uv_scale = [1.0, 1.0]
        self.uv_offset = [0.0, 0.0]
        self.version = next(_VERSION)

    # ---- utilidades ----
    def touch(self):
        self.version = next(_VERSION)

    def copy(self):
        m = Mesh(self.v.copy(), [list(f) for f in self.f])
        m.smooth, m.uv_mode, m.uv_axis = self.smooth, self.uv_mode, self.uv_axis
        m.uv_scale, m.uv_offset = list(self.uv_scale), list(self.uv_offset)
        return m

    def compact(self):
        """Elimina vértices huérfanos. Devuelve el mapa viejo->nuevo (-1 si se borró)."""
        used = np.zeros(len(self.v), bool)
        for f in self.f:
            used[f] = True
        remap = -np.ones(len(self.v), int)
        remap[used] = np.arange(int(used.sum()))
        self.v = self.v[used]
        self.f = [[int(remap[i]) for i in f] for f in self.f]
        return remap

    def bounds(self):
        if len(self.v) == 0:
            return np.zeros(3), np.zeros(3)
        return self.v.min(0), self.v.max(0)

    # ---- topología ----
    def half_edges(self):
        he = {}
        for fi, f in enumerate(self.f):
            for i, a in enumerate(f):
                he[(a, f[(i + 1) % len(f)])] = fi
        return he

    def edge_faces(self):
        ef = defaultdict(list)
        for fi, f in enumerate(self.f):
            for i, a in enumerate(f):
                b = f[(i + 1) % len(f)]
                ef[(a, b) if a < b else (b, a)].append(fi)
        return dict(ef)

    def edges(self):
        return sorted(self.edge_faces().keys())

    # ---- geometría ----
    def face_normals(self, unit=True):
        out = np.zeros((len(self.f), 3))
        for fi, f in enumerate(self.f):
            p = self.v[f]; q = np.roll(p, -1, axis=0)       # método de Newell
            out[fi] = [((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
                       ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
                       ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum()]
        return normalize(out) if unit else out

    def vertex_normals(self):
        fn = self.face_normals(unit=False)          # ponderadas por área
        vn = np.zeros_like(self.v)
        for fi, f in enumerate(self.f):
            vn[f] += fn[fi]
        return normalize(vn)

    def face_center(self, fi):
        return self.v[self.f[fi]].mean(0)

    # ---- render ----
    def _corners(self):
        ci = np.fromiter(itertools.chain.from_iterable(self.f), dtype=int)
        cf = np.repeat(np.arange(len(self.f)), [len(f) for f in self.f])
        return ci, cf

    def triangles(self):
        """Triangulación en abanico. Devuelve (tri_corner (T,3), tri_face (T,))."""
        tc, tf, s = [], [], 0
        for fi, f in enumerate(self.f):
            k = len(f)
            for j in range(1, k - 1):
                tc.append((s, s + j, s + j + 1)); tf.append(fi)
            s += k
        return np.array(tc, int).reshape(-1, 3), np.array(tf, int)

    def corner_uv(self):
        """UV por esquina de cara (proyección planar o cúbica + escala/offset)."""
        ci, cf = self._corners()
        if len(ci) == 0:
            return np.zeros((0, 2))
        pos = self.v[ci]
        if self.uv_mode == "planar":
            a, b = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[self.uv_axis]
            u, w = pos[:, a], pos[:, b]
        else:
            dom = np.argmax(np.abs(self.face_normals()[cf]), axis=1)
            u = np.where(dom == 0, pos[:, 2], pos[:, 0])
            w = np.where(dom == 1, pos[:, 2], pos[:, 1])
        return np.stack([u * self.uv_scale[0] + self.uv_offset[0],
                         w * self.uv_scale[1] + self.uv_offset[1]], axis=1)

    def render_arrays(self):
        """Buffers por vértice-de-triángulo: pos, nrm, uv (+ tri_face, tri_verts)."""
        tc, tf = self.triangles()
        if len(tc) == 0:
            z3, z2 = np.zeros((0, 3), "f4"), np.zeros((0, 2), "f4")
            return dict(pos=z3, nrm=z3, uv=z2, tri_face=tf, tri_verts=np.zeros((0, 3), int))
        ci, _ = self._corners()
        flat = tc.ravel()
        vi = ci[flat]
        nrm = self.vertex_normals()[vi] if self.smooth else self.face_normals()[np.repeat(tf, 3)]
        return dict(pos=self.v[vi].astype("f4"), nrm=nrm.astype("f4"),
                    uv=self.corner_uv()[flat].astype("f4"),
                    tri_face=tf, tri_verts=vi.reshape(-1, 3))

    def edge_lines(self):
        es = np.array(self.edges(), int).reshape(-1, 2)
        return self.v[es.ravel()].astype("f4") if len(es) else np.zeros((0, 3), "f4")
