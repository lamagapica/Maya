"""Operaciones de modelado: Extrude, Bevel y subdivisión Catmull-Clark."""
from collections import defaultdict
import numpy as np
from .mathutils import normalize


def extrude_faces(mesh, face_ids, distance=0.5):
    """Extruye una región de caras a lo largo de sus normales.

    - Los vértices de la región se duplican y se desplazan (normal promedio de
      las caras seleccionadas que comparten el vértice).
    - Cada semiarista de contorno (a->b) de la región genera un quad lateral
      [a, b, b', a'] con orientación coherente con la malla original.
    Devuelve los índices de las caras extruidas (siguen siendo las mismas).
    """
    sel = set(face_ids)
    if not sel:
        return []
    fn = mesh.face_normals()
    acc = defaultdict(lambda: np.zeros(3))
    for fi in sel:
        for v in mesh.f[fi]:
            acc[v] += fn[fi]
    base = len(mesh.v)
    new_id, new_pos = {}, []
    for k, v in enumerate(sorted(acc)):
        new_id[v] = base + k
        new_pos.append(mesh.v[v] + normalize(acc[v]) * distance)
    he = mesh.half_edges()
    sides = []
    for fi in sel:
        f = mesh.f[fi]
        for i, a in enumerate(f):
            b = f[(i + 1) % len(f)]
            twin = he.get((b, a))
            if twin is None or twin not in sel:          # arista de contorno
                sides.append([a, b, new_id[b], new_id[a]])
    for fi in sel:
        mesh.f[fi] = [new_id[v] for v in mesh.f[fi]]
    mesh.v = np.vstack([mesh.v, np.array(new_pos)])
    mesh.f.extend(sides)
    mesh.compact()               # elimina vértices interiores huérfanos (no cambia índices de caras)
    mesh.touch()
    return sorted(sel)


def bevel_edges(mesh, edges, width=0.1):
    """Chaflán de 1 segmento sobre aristas seleccionadas (variedad cerrada recomendada).

    Cada esquina de cara junto a una arista biselada se desplaza `width` a lo largo
    de la arista vecina; entre las dos caras se crea una tira (quad) y los huecos
    que quedan en los vértices se rellenan con tapas detectadas por semiaristas
    sin gemela. Devuelve True si cambió la malla.
    """
    E = {tuple(sorted(e)) for e in edges}
    ef = mesh.edge_faces()
    E = {e for e in E if len(ef.get(e, [])) == 2}
    if not E:
        return False
    V = mesh.v
    nv = len(V)
    new_pos = []

    def add(p):
        new_pos.append(p)
        return nv + len(new_pos) - 1

    def isb(a, b):
        return ((a, b) if a < b else (b, a)) in E

    def slide(v, u):
        d = V[u] - V[v]; L = np.linalg.norm(d)
        return d / L * min(width, 0.45 * L)

    need = set()
    for f in mesh.f:
        m = len(f)
        for i, v in enumerate(f):
            p, n = f[i - 1], f[(i + 1) % m]
            bp, bn = isb(p, v), isb(v, n)
            if bp and not bn:
                need.add((v, n))
            elif bn and not bp:
                need.add((v, p))
    S = {k: add(V[k[0]] + slide(*k)) for k in sorted(need)}

    new_faces, he_new = [], {}
    for f in mesh.f:
        m = len(f)
        R = []
        for i, v in enumerate(f):
            p, n = f[i - 1], f[(i + 1) % m]
            bp, bn = isb(p, v), isb(v, n)
            if not bp and not bn:
                sp, sn = S.get((v, p)), S.get((v, n))
                if sp is not None and sn is not None:
                    r = [sp, sn]                       # el vértice queda cortado por el chaflán
                else:
                    r = ([sp] if sp is not None else []) + [v] + ([sn] if sn is not None else [])
            elif bn and not bp:
                r = [S[(v, p)]]
            elif bp and not bn:
                r = [S[(v, n)]]
            else:
                r = [add(V[v] + slide(v, p) + slide(v, n))]
            R.append(r)
        for i, v in enumerate(f):
            n = f[(i + 1) % m]
            if isb(v, n):
                he_new[(v, n)] = (R[i][-1], R[(i + 1) % m][0])
        new_faces.append([x for r in R for x in r])

    for (a, b), (s, e) in he_new.items():                # tiras entre caras
        if a < b and (b, a) in he_new:
            s2, e2 = he_new[(b, a)]
            new_faces.append([e, s, e2, s2])

    # tapas en los vértices: bucles de semiaristas de contorno invertidas
    aff = set(range(nv, nv + len(new_pos))) | {v for e in E for v in e}
    hes = {(f[i], f[(i + 1) % len(f)]) for f in new_faces for i in range(len(f))}
    nxt = defaultdict(list)
    for a, b in hes:
        if (b, a) not in hes and a in aff and b in aff:
            nxt[b].append(a)
    for sb in list(nxt):
        while nxt[sb]:
            loop, cur = [sb], sb
            while True:
                if not nxt[cur]:
                    loop = None
                    break
                nx = nxt[cur].pop()
                if nx == sb:
                    break
                loop.append(nx); cur = nx
            if loop and len(loop) >= 3:
                new_faces.append(loop)

    mesh.v = np.vstack([mesh.v, np.array(new_pos)]) if new_pos else mesh.v
    mesh.f = new_faces
    mesh.compact()
    mesh.touch()
    return True


def catmull_clark(mesh, levels=1):
    """Subdivisión Catmull-Clark (caras n-gon -> quads, bordes con regla de contorno)."""
    for _ in range(levels):
        V, F = mesh.v, mesh.f
        fp = np.array([V[f].mean(0) for f in F])
        ef = mesh.edge_faces()
        edges = list(ef.keys())
        eidx = {e: i for i, e in enumerate(edges)}
        ep = np.zeros((len(edges), 3))
        for e, i in eidx.items():
            fs = ef[e]
            mid = (V[e[0]] + V[e[1]]) / 2
            ep[i] = (V[e[0]] + V[e[1]] + fp[fs[0]] + fp[fs[1]]) / 4 if len(fs) == 2 else mid
        vfaces, vedges = defaultdict(list), defaultdict(list)
        for fi, f in enumerate(F):
            for v in f:
                vfaces[v].append(fi)
        for e in edges:
            for v in e:
                vedges[v].append(e)
        nV = np.array(V)
        for v in range(len(V)):
            es = vedges[v]; n = len(es); P = V[v]
            if n == 0:
                continue
            bd = [e for e in es if len(ef[e]) == 1]
            if bd:
                if len(bd) == 2:
                    a = [x for x in bd[0] if x != v][0]; b = [x for x in bd[1] if x != v][0]
                    nV[v] = (6 * P + V[a] + V[b]) / 8
            else:
                Fa = fp[vfaces[v]].mean(0)
                R = np.mean([(V[e[0]] + V[e[1]]) / 2 for e in es], axis=0)
                nV[v] = (Fa + 2 * R + (n - 3) * P) / n
        nVn, nE = len(V), len(edges)
        def E(a, b):
            return nVn + eidx[(a, b) if a < b else (b, a)]
        faces = []
        for fi, f in enumerate(F):
            k = len(f)
            for i, v in enumerate(f):
                faces.append([v, E(v, f[(i + 1) % k]), nVn + nE + fi, E(f[i - 1], v)])
        mesh.v = np.vstack([nV, ep, fp])
        mesh.f = faces
    mesh.touch()
    return mesh
