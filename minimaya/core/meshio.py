"""Importar/Exportar: OBJ (nativo), glTF 2.0 (export propio con buffers embebidos;
import de .gltf/.glb vía trimesh). FBX es un formato propietario: conviértelo a glTF/OBJ."""
import base64
import json
import os
import numpy as np
from .mesh import Mesh


# ------------------------------------------------------------------ OBJ
def import_obj(path):
    verts, objs, cur = [], {}, "polyMesh"
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                verts.append([float(x) for x in p[1:4]])
            elif p[0] in ("o", "g") and len(p) > 1:
                cur = p[1]
            elif p[0] == "f":
                idx = []
                for tok in p[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                objs.setdefault(cur, []).append(idx)
    V = np.array(verts, float).reshape(-1, 3)
    out = []
    for name, faces in objs.items():
        m = Mesh(V, faces)
        m.compact()
        out.append((name, m))
    return out


def export_obj(path, scene):
    base = os.path.splitext(path)[0]
    lines = [f"mtllib {os.path.basename(base)}.mtl"]
    off_v = off_t = 1
    used = set()
    for n in scene.nodes.values():
        if n.kind != "mesh" or not n.mesh.f or not scene.is_visible(n):
            continue
        M = scene.world_matrix(n)
        V = n.mesh.v @ M[:3, :3].T + M[:3, 3]
        uv = n.mesh.corner_uv()
        mat = scene.materials.get(n.material, scene.materials[1])
        used.add(mat.id)
        lines.append(f"o {n.name}\nusemtl {mat.name}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
        lines += [f"vt {u:.6f} {w:.6f}" for u, w in uv]
        c = 0
        for f in n.mesh.f:
            lines.append("f " + " ".join(f"{vi + off_v}/{c + k + off_t}" for k, vi in enumerate(f)))
            c += len(f)
        off_v += len(V); off_t += len(uv)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    mtl = []
    for mid in used:
        m = scene.materials[mid]
        mtl.append(f"newmtl {m.name}\nKd {m.albedo[0]:.4f} {m.albedo[1]:.4f} {m.albedo[2]:.4f}\n"
                   f"Pm {m.metallic:.3f}\nPr {m.roughness:.3f}")
        if m.albedo_tex:
            mtl.append(f"map_Kd {m.albedo_tex}")
        if m.normal_tex:
            mtl.append(f"norm {m.normal_tex}")
    with open(base + ".mtl", "w") as fh:
        fh.write("\n".join(mtl) + "\n")


# ------------------------------------------------------------------ glTF
def _data_uri(path):
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


def export_gltf(path, scene):
    blob = bytearray()
    views, accs, meshes, nodes, mats, images, textures = [], [], [], [], [], [], []
    mat_index = {}

    def add_view(arr, target=None):
        data = arr.tobytes()
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), **({"target": target} if target else {})})
        blob.extend(data)
        return len(views) - 1

    def add_tex(p):
        if not p or not os.path.exists(p):
            return None
        images.append({"uri": _data_uri(p)})
        textures.append({"source": len(images) - 1})
        return len(textures) - 1

    def material(mid):
        if mid in mat_index:
            return mat_index[mid]
        m = scene.materials.get(mid, scene.materials[1])
        pbr = {"baseColorFactor": [*m.albedo, 1.0], "metallicFactor": m.metallic, "roughnessFactor": m.roughness}
        t = add_tex(m.albedo_tex)
        if t is not None:
            pbr["baseColorTexture"] = {"index": t}
        d = {"name": m.name, "pbrMetallicRoughness": pbr, "doubleSided": True}
        t = add_tex(m.normal_tex)
        if t is not None:
            d["normalTexture"] = {"index": t}
        mats.append(d)
        mat_index[mid] = len(mats) - 1
        return mat_index[mid]

    for n in scene.nodes.values():
        if n.kind != "mesh" or not n.mesh.f or not scene.is_visible(n):
            continue
        M = scene.world_matrix(n)
        a = n.mesh.render_arrays()
        pos = (a["pos"].astype(float) @ M[:3, :3].T + M[:3, 3]).astype("f4")
        nrm = (a["nrm"].astype(float) @ np.linalg.inv(M[:3, :3])).astype("f4")
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        uv = a["uv"].copy(); uv[:, 1] = 1 - uv[:, 1]
        idx = np.arange(len(pos), dtype="u4")
        bi = len(accs)
        accs.append({"bufferView": add_view(idx, 34963), "componentType": 5125, "count": len(idx), "type": "SCALAR"})
        accs.append({"bufferView": add_view(pos, 34962), "componentType": 5126, "count": len(pos), "type": "VEC3",
                     "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
        accs.append({"bufferView": add_view(nrm, 34962), "componentType": 5126, "count": len(nrm), "type": "VEC3"})
        accs.append({"bufferView": add_view(uv.astype("f4"), 34962), "componentType": 5126, "count": len(uv), "type": "VEC2"})
        meshes.append({"name": n.name, "primitives": [{"attributes": {"POSITION": bi + 1, "NORMAL": bi + 2, "TEXCOORD_0": bi + 3},
                                                      "indices": bi, "material": material(n.material)}]})
        nodes.append({"name": n.name, "mesh": len(meshes) - 1})
    doc = {"asset": {"version": "2.0", "generator": "Mini-Maya"}, "scene": 0,
           "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
           "materials": mats, "accessors": accs, "bufferViews": views,
           "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}]}
    if images:
        doc["images"], doc["textures"] = images, textures
    with open(path, "w") as fh:
        json.dump(doc, fh)


def import_gltf(path):
    try:
        import trimesh
    except ImportError as e:
        raise RuntimeError("Instala 'trimesh' para importar glTF/GLB (pip install trimesh).") from e
    sc = trimesh.load(path, force="scene")
    out = []
    for i, g in enumerate(sc.dump()):
        if hasattr(g, "faces") and len(g.faces):
            out.append((f"gltfMesh{i + 1}", Mesh(np.asarray(g.vertices), np.asarray(g.faces))))
    return out
