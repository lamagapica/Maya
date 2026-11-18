"""Gizmos de transformación (mover/rotar/escalar): geometría, dibujo y cálculo de arrastre."""
import numpy as np
import moderngl
from ..core import primitives as P
from ..core.mathutils import rot_y, rot_z, normalize, gl

COLORS = [(0.92, 0.22, 0.22, 1.0), (0.30, 0.85, 0.30, 1.0), (0.28, 0.48, 1.0, 1.0)]
HOVER = (1.0, 0.92, 0.2, 1.0)
AXES = np.eye(3)


def _part(mesh, R=None, t=(0, 0, 0)):
    a = mesh.render_arrays()
    pos, nrm = a["pos"].astype(float), a["nrm"].astype(float)
    if R is not None:
        pos, nrm = pos @ R.T, nrm @ R.T
    return np.hstack([pos + np.asarray(t), nrm]).astype("f4")


def _axis_frames():
    return [np.eye(3), rot_z(90), rot_y(-90)]       # lleva +X a X, Y, Z


def _build_tool(tool):
    parts = []
    along_x = rot_z(-90)                              # primitivas con eje Y -> eje X
    for Ra in _axis_frames():
        if tool == "move":
            geo = [_part(P.cylinder(0.025, 0.7, 8), along_x, (0.45, 0, 0)), _part(P.cone(0.075, 0.26, 12), along_x, (1.0, 0, 0))]
        elif tool == "scale":
            geo = [_part(P.cylinder(0.025, 0.85, 8), along_x, (0.5, 0, 0)), _part(P.cube(0.14), None, (1.0, 0, 0))]
        else:
            geo = [_part(P.torus(1.0, 0.018, 64, 6), along_x)]
        arr = np.vstack(geo)
        pos = arr[:, :3] @ Ra.T; nrm = arr[:, 3:] @ Ra.T
        parts.append(np.hstack([pos, nrm]).astype("f4"))
    return parts


class Gizmo:
    def __init__(self, ctx, prog):
        self.ctx, self.prog = ctx, prog
        self.vaos = {}
        center = _part(P.cube(0.16))
        self.vaos["center"] = self._vao(center)
        for tool in ("move", "rotate", "scale"):
            self.vaos[tool] = [self._vao(p) for p in _build_tool(tool)]

    def _vao(self, arr):
        vbo = self.ctx.buffer(arr.tobytes())
        return self.ctx.vertex_array(self.prog, [(vbo, "3f 3f", "in_pos", "in_nrm")])

    def draw(self, tool, origin, basis, size, VP, V, hover):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        M = np.eye(4); M[:3, :3] = basis * size; M[:3, 3] = origin
        self.prog["uMVP"].write(gl(VP @ M))
        self.prog["uNV"].write(np.ascontiguousarray(V[:3, :3].T, "f4").tobytes())
        for i, vao in enumerate(self.vaos[tool]):
            self.prog["uColor"].value = HOVER if hover == i else COLORS[i]
            vao.render(moderngl.TRIANGLES)
        if tool == "scale":
            self.prog["uColor"].value = HOVER if hover == 3 else (0.9, 0.9, 0.9, 1.0)
            self.vaos["center"].render(moderngl.TRIANGLES)
        ctx.enable(moderngl.DEPTH_TEST)


# ------------------------------------------------------------------ matemáticas de interacción
def closest_param(origin, axis, ro, rd):
    """Parámetro t sobre la recta origin + t*axis más cercano al rayo (ro, rd)."""
    w0 = origin - ro
    b = axis @ rd; d = axis @ w0; e = rd @ w0
    den = 1.0 - b * b
    if abs(den) < 1e-6:
        return 0.0
    return (b * e - d) / den


def ray_plane(ro, rd, p0, n):
    den = rd @ n
    if abs(den) < 1e-8:
        return None
    t = ((p0 - ro) @ n) / den
    return ro + rd * t if t > 0 else None


def plane_basis(axis):
    a = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = normalize(np.cross(a, axis))
    return u, np.cross(axis, u)


def ring_angle(origin, axis, ro, rd):
    q = ray_plane(ro, rd, origin, axis)
    if q is None:
        return None
    u, v = plane_basis(axis)
    return np.arctan2((q - origin) @ v, (q - origin) @ u)


def pick(tool, origin, basis, size, mouse, ro, rd, project):
    """Devuelve 0/1/2 (eje), 3 (centro, solo escalar) o None. `project` -> (N,2) px."""
    if tool == "rotate":
        best, bd = None, 0.12 * size
        for i in range(3):
            q = ray_plane(ro, rd, origin, basis[:, i])
            if q is None:
                continue
            d = abs(np.linalg.norm(q - origin) - size)
            if d < bd:
                best, bd = i, d
        return best
    pts = np.array([origin] + [origin + basis[:, i] * size * 1.05 for i in range(3)])
    sp, ok = project(pts)
    if not ok.all():
        return None
    if tool == "scale" and np.linalg.norm(sp[0] - mouse) < 12:
        return 3
    best, bd = None, 11.0
    for i in range(3):
        a, b = sp[0], sp[i + 1]
        ab = b - a
        t = np.clip(((mouse - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        d = np.linalg.norm(a + ab * t - mouse)
        if d < bd:
            best, bd = i, d
    return best
