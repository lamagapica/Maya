"""Matemáticas 3D con NumPy. Convención: vectores columna, matrices 4x4 (M @ v)."""
import numpy as np


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def translation(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def scaling(s):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def rot_x(d):
    r = np.radians(d); c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(d):
    r = np.radians(d); c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(d):
    r = np.radians(d); c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def euler_to_matrix(e):
    """Orden XYZ de Maya: R = Rz @ Ry @ Rx (grados)."""
    return rot_z(e[2]) @ rot_y(e[1]) @ rot_x(e[0])


def matrix_to_euler(R):
    ry = np.arcsin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(R[2, 0]) < 0.99999:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return np.degrees([rx, ry, rz])


def axis_rotation(axis, deg):
    a = normalize(axis); r = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(r) * K + (1 - np.cos(r)) * (K @ K)


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4)
    m[0, 0] = 2 / (r - l); m[0, 3] = -(r + l) / (r - l)
    m[1, 1] = 2 / (t - b); m[1, 3] = -(t + b) / (t - b)
    m[2, 2] = -2 / (f - n); m[2, 3] = -(f + n) / (f - n)
    return m


def look_at(eye, target, up):
    eye = np.asarray(eye, float)
    f = normalize(np.asarray(target, float) - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def gl(m):
    """Matriz NumPy (M@v) -> bytes column-major para GLSL."""
    return np.ascontiguousarray(np.asarray(m).T, dtype="f4").tobytes()


def ray_triangles(o, d, tri):
    """Möller–Trumbore vectorizado. tri: (T,3,3). Devuelve t (T,) con inf si falla."""
    if len(tri) == 0:
        return np.zeros(0)
    v0 = tri[:, 0]; e1 = tri[:, 1] - v0; e2 = tri[:, 2] - v0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = 1.0 / np.where(ok, det, 1.0)
    tv = o - v0
    u = np.einsum("ij,ij->i", tv, p) * inv
    q = np.cross(tv, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    return np.where(good, t, np.inf)
