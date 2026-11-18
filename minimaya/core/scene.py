"""Modelo de escena: nodos (malla/luz/grupo), jerarquía, materiales PBR y undo por snapshot."""
import copy
import itertools
import numpy as np
from .mathutils import euler_to_matrix, normalize

_NODE_IDS = itertools.count(1)
_MAT_IDS = itertools.count(2)


class Material:
    def __init__(self, name="lambert", albedo=(0.7, 0.7, 0.7), roughness=0.5, metallic=0.0, mid=None):
        self.id = mid if mid is not None else next(_MAT_IDS)
        self.name = name
        self.albedo = list(albedo)
        self.roughness = roughness
        self.metallic = metallic
        self.albedo_tex = None
        self.normal_tex = None
        self.normal_strength = 1.0


class Node:
    def __init__(self, name, kind, mesh=None):
        self.id = next(_NODE_IDS)
        self.name = name
        self.kind = kind                 # 'mesh' | 'light' | 'group'
        self.parent = None
        self.t = np.zeros(3)
        self.r = np.zeros(3)             # grados, orden XYZ
        self.s = np.ones(3)
        self.visible = True
        self.mesh = mesh
        self.material = 1
        self.light = {"type": "point", "color": [1.0, 1.0, 1.0], "intensity": 30.0}

    def local_matrix(self):
        m = np.eye(4)
        m[:3, :3] = euler_to_matrix(self.r) @ np.diag(self.s)
        m[:3, 3] = self.t
        return m

    def light_dir(self, world):
        return normalize(world[:3, :3] @ np.array([0.0, 0.0, -1.0]))


class RenderSettings:
    def __init__(self):
        self.width, self.height = 480, 270
        self.spp = 16
        self.bounces = 3
        self.ao_radius = 1.0
        self.ao_strength = 1.0
        self.env_intensity = 1.0


class Scene:
    def __init__(self):
        self.nodes = {}
        self.materials = {1: Material("lambert1", mid=1)}
        self.selection = []
        self.render = RenderSettings()

    # ---- creación ----
    def unique_name(self, base):
        names = {n.name for n in self.nodes.values()}
        i = 1
        while f"{base}{i}" in names:
            i += 1
        return f"{base}{i}"

    def add_mesh(self, mesh, base="polyMesh"):
        n = Node(self.unique_name(base), "mesh", mesh)
        self.nodes[n.id] = n
        return n

    def add_light(self, kind="point"):
        n = Node(self.unique_name("pointLight" if kind == "point" else "directionalLight"), "light")
        n.light["type"] = kind
        if kind == "directional":
            n.r = np.array([-50.0, 30.0, 0.0]); n.t = np.array([0.0, 6.0, 0.0]); n.light["intensity"] = 3.0
        else:
            n.t = np.array([3.0, 5.0, 3.0])
        self.nodes[n.id] = n
        return n

    def add_group(self, ids):
        g = Node(self.unique_name("group"), "group")
        self.nodes[g.id] = g
        for i in ids:
            if i in self.nodes:
                self.nodes[i].parent = g.id
        return g

    def add_material(self, name="material"):
        m = Material(self.unique_name_mat(name))
        self.materials[m.id] = m
        return m

    def unique_name_mat(self, base):
        names = {m.name for m in self.materials.values()}
        i = 1
        while f"{base}{i}" in names:
            i += 1
        return f"{base}{i}"

    # ---- jerarquía ----
    def children(self, pid):
        return [n for n in self.nodes.values() if n.parent == pid]

    def descendants(self, nid):
        out = []
        for c in self.children(nid):
            out += [c.id] + self.descendants(c.id)
        return out

    def world_matrix(self, node):
        m = node.local_matrix()
        p = self.nodes.get(node.parent)
        return self.world_matrix(p) @ m if p is not None else m

    def is_visible(self, node):
        while node is not None:
            if not node.visible:
                return False
            node = self.nodes.get(node.parent)
        return True

    def delete(self, ids):
        kill = set(ids)
        for i in list(ids):
            kill |= set(self.descendants(i))
        for i in kill:
            self.nodes.pop(i, None)
        self.selection = [i for i in self.selection if i in self.nodes]

    def duplicate(self, ids):
        out = []
        for i in ids:
            n = self.nodes.get(i)
            if n is None or n.kind == "group":
                continue
            c = copy.deepcopy(n)
            c.id = next(_NODE_IDS)
            c.name = self.unique_name(n.name.rstrip("0123456789") or n.name)
            c.t = c.t + np.array([1.0, 0.0, 0.0])
            self.nodes[c.id] = c
            out.append(c.id)
        return out

    # ---- snapshot (undo/redo) ----
    def state(self):
        return copy.deepcopy((self.nodes, self.materials, list(self.selection), self.render))

    def set_state(self, st):
        self.nodes, self.materials, self.selection, self.render = copy.deepcopy(st)

    def world_bounds(self):
        pts = []
        for n in self.nodes.values():
            if n.kind == "mesh" and n.mesh is not None and len(n.mesh.v) and self.is_visible(n):
                M = self.world_matrix(n)
                lo, hi = n.mesh.bounds()
                c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                pts.append(c @ M[:3, :3].T + M[:3, 3])
        if not pts:
            return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
        p = np.vstack(pts)
        return p.min(0), p.max(0)
