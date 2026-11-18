"""Pruebas sin GUI: python -m tests.test_core"""
import numpy as np
from minimaya.core import primitives as P
from minimaya.core.modeling import extrude_faces, bevel_edges, catmull_clark


def closed_manifold(m):
    he = m.half_edges()
    n = sum(len(f) for f in m.f)
    return len(he) == n and all((b, a) in he for (a, b) in he)


def volume(m):
    t = 0.0
    for f in m.f:
        for k in range(1, len(f) - 1):
            t += np.dot(m.v[f[0]], np.cross(m.v[f[k]], m.v[f[k + 1]])) / 6
    return t


def test_primitives():
    for name, m in dict(cube=P.cube(), sphere=P.sphere(), cyl=P.cylinder(), cone=P.cone(), torus=P.torus()).items():
        assert closed_manifold(m), name
        assert volume(m) > 0, name + " normales hacia dentro"


def test_extrude():
    m = P.cube(2.0)
    top = [i for i, n in enumerate(m.face_normals()) if n[1] > 0.9]
    extrude_faces(m, top, 1.0)
    assert closed_manifold(m) and len(m.f) == 10 and len(m.v) == 12
    assert abs(volume(m) - 12.0) < 1e-9


def test_extrude_all_region():
    m = P.cube(2.0)
    extrude_faces(m, [0, 1, 2, 3, 4, 5], 0.5)
    assert closed_manifold(m)


def test_bevel_one_edge():
    m = P.cube(2.0)
    v0 = volume(m)
    assert bevel_edges(m, [m.edges()[0]], 0.3)
    assert closed_manifold(m) and 0 < volume(m) < v0


def test_bevel_all_edges():
    m = P.cube(2.0)
    assert bevel_edges(m, m.edges(), 0.2)
    assert closed_manifold(m) and 0 < volume(m) < 8.0


def test_bevel_loop_on_cylinder():
    m = P.cylinder(1, 2, 16)
    ef = m.edge_faces()
    top = [e for e, fs in ef.items() if all(v >= 16 for v in e)]
    assert bevel_edges(m, top, 0.15) and closed_manifold(m)


def test_subdivide():
    m = P.cube(2.0)
    catmull_clark(m, 2)
    assert closed_manifold(m) and len(m.f) == 96
    assert 0 < volume(m) < 8.0


def test_uv():
    m = P.cube(2.0)
    assert m.corner_uv().shape == (24, 2)
    a = m.render_arrays()
    assert a["pos"].shape == (36, 3) and a["uv"].shape == (36, 2)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("OK", k)
