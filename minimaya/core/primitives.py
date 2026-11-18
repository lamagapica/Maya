"""Primitivas poligonales (todas con normales hacia afuera)."""
import numpy as np
from .mesh import Mesh


def _orient_outward(m):
    c = m.v.mean(0)
    fn = m.face_normals()
    for fi, f in enumerate(m.f):
        if np.dot(fn[fi], m.v[f].mean(0) - c) < 0:
            m.f[fi] = f[::-1]
    return m


def cube(size=2.0):
    h = size / 2
    v = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h), (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 4, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [3, 7, 6, 2]]
    return Mesh(v, f)


def plane(size=4.0, div=1):
    n = div + 1
    v = [(-size / 2 + size * i / div, 0, -size / 2 + size * j / div) for i in range(n) for j in range(n)]
    f = [[i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j]
         for i in range(div) for j in range(div)]
    return Mesh(v, f)


def sphere(radius=1.0, rings=12, segs=24):
    v = [(0, radius, 0)]
    for i in range(1, rings):
        th = np.pi * i / rings
        for j in range(segs):
            ph = 2 * np.pi * j / segs
            v.append((radius * np.sin(th) * np.cos(ph), radius * np.cos(th), radius * np.sin(th) * np.sin(ph)))
    v.append((0, -radius, 0))
    ring = lambda i, j: 1 + (i - 1) * segs + j % segs
    f = [[0, ring(1, j), ring(1, j + 1)] for j in range(segs)]
    for i in range(1, rings - 1):
        for j in range(segs):
            f.append([ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j + 1)])
    last = len(v) - 1
    f += [[last, ring(rings - 1, j + 1), ring(rings - 1, j)] for j in range(segs)]
    return _orient_outward(Mesh(v, f))


def cylinder(radius=1.0, height=2.0, segs=24):
    v = []
    for y in (-height / 2, height / 2):
        v += [(radius * np.cos(2 * np.pi * j / segs), y, radius * np.sin(2 * np.pi * j / segs)) for j in range(segs)]
    f = [[j, (j + 1) % segs, segs + (j + 1) % segs, segs + j] for j in range(segs)]
    f += [list(range(segs)), list(range(segs, 2 * segs))]
    return _orient_outward(Mesh(v, f))


def cone(radius=1.0, height=2.0, segs=24):
    v = [(radius * np.cos(2 * np.pi * j / segs), -height / 2, radius * np.sin(2 * np.pi * j / segs)) for j in range(segs)]
    v.append((0, height / 2, 0))
    f = [[j, (j + 1) % segs, segs] for j in range(segs)] + [list(range(segs))]
    return _orient_outward(Mesh(v, f))


def torus(R=1.0, r=0.35, segs=32, sides=16):
    v = []
    for i in range(segs):
        u = 2 * np.pi * i / segs
        for j in range(sides):
            w = 2 * np.pi * j / sides
            v.append(((R + r * np.cos(w)) * np.cos(u), r * np.sin(w), (R + r * np.cos(w)) * np.sin(u)))
    f = [[i * sides + j, ((i + 1) % segs) * sides + j, ((i + 1) % segs) * sides + (j + 1) % sides,
          i * sides + (j + 1) % sides] for i in range(segs) for j in range(sides)]
    m = Mesh(v, f)
    fn = m.face_normals()
    for fi, face in enumerate(m.f):                      # hacia afuera respecto al centro del tubo
        p = m.v[face].mean(0)
        tube_c = np.array([p[0], 0, p[2]]); tube_c = tube_c / max(np.linalg.norm(tube_c), 1e-9) * R
        if np.dot(fn[fi], p - tube_c) < 0:
            m.f[fi] = face[::-1]
    return m
