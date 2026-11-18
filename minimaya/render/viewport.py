"""Viewport 3D (QOpenGLWidget + ModernGL): cámara estilo Maya, PBR + sombras + IBL,
grid, gizmos W/E/R, selección de objetos y de componentes (vértice/arista/cara)."""
import os
import numpy as np
import moderngl
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel

from ..core.camera import Camera
from ..core.mathutils import (gl, look_at, ortho, normalize, axis_rotation, euler_to_matrix,
                              matrix_to_euler, ray_triangles)
from . import shaders as S
from .env import EnvMap
from .gizmo import Gizmo, pick, closest_param, ring_angle

SEL, ACTIVE, WIRE, ORANGE = (0.35, 1.0, 0.45, 1.0), (1.0, 1.0, 1.0, 1.0), (0.62, 0.68, 0.80, 1.0), (1.0, 0.6, 0.1, 1.0)
SHADOW_RES = 2048


def U(prog, name, value):
    """Asigna un uniform ignorando los que el compilador de GLSL eliminó."""
    try:
        u = prog[name]
    except KeyError:
        return
    if isinstance(value, (bytes, bytearray)):
        u.write(value)
    else:
        u.value = value


def mat3_bytes(m):
    return np.ascontiguousarray(np.asarray(m).T, "f4").tobytes()


class Viewport(QOpenGLWidget):
    selectionChanged = Signal()
    transformChanged = Signal()
    editStarted = Signal()
    status = Signal(str)

    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.camera = Camera()
        self.display = "shaded"            # wire | shaded | wireshaded | textured
        self.tool = "select"               # select | move | rotate | scale
        self.comp_mode = "object"          # object | vertex | edge | face
        self.comp_sel = {"v": set(), "e": set(), "f": set()}
        self.show_shadows, self.show_grid = True, True
        self.env = EnvMap.procedural()
        self._env_dirty = True
        self.ctx = None
        self.cache, self.tex_cache = {}, {}
        self.nav = None
        self.drag = None
        self.hover_axis = None
        self.last = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.hud = QLabel(self)
        self.hud.setStyleSheet("color:#cfcfcf;background:transparent;padding:4px;")
        self.hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hud.move(6, 2)
        self.update_hud()

    # ------------------------------------------------------------------ estado
    def set_env(self, env):
        self.env, self._env_dirty = env, True
        self.update()

    def set_display(self, mode):
        self.display = mode
        self.update_hud(); self.update()

    def set_tool(self, tool):
        self.tool = tool
        self.update_hud(); self.update()

    def set_comp_mode(self, mode):
        self.comp_mode = mode
        self.clear_comp()
        self.update_hud(); self.update()

    def clear_comp(self):
        for s in self.comp_sel.values():
            s.clear()

    def update_hud(self):
        names = {"wire": "Wireframe", "shaded": "Shaded", "wireshaded": "Wireframe on Shaded", "textured": "Textured"}
        self.hud.setText(f"persp  |  {names[self.display]}  |  tool: {self.tool}  |  {self.comp_mode}")
        self.hud.adjustSize()

    def active_node(self):
        for i in reversed(self.scene.selection):
            n = self.scene.nodes.get(i)
            if n is not None and n.kind == "mesh":
                return n
        return None

    def _last_node(self):
        for i in reversed(self.scene.selection):
            if i in self.scene.nodes:
                return self.scene.nodes[i]
        return None

    def selected_verts(self):
        n = self.active_node()
        if n is None:
            return []
        m, vs = n.mesh, set(self.comp_sel["v"])
        for a, b in self.comp_sel["e"]:
            vs |= {a, b}
        for fi in self.comp_sel["f"]:
            if fi < len(m.f):
                vs |= set(m.f[fi])
        return sorted(v for v in vs if v < len(m.v))

    def frame_selected(self):
        nodes = [self.scene.nodes[i] for i in self.scene.selection if i in self.scene.nodes]
        if not nodes:
            nodes = [n for n in self.scene.nodes.values() if n.kind == "mesh"]
        pts = []
        for n in nodes:
            M = self.scene.world_matrix(n)
            if n.kind == "mesh" and len(n.mesh.v):
                lo, hi = n.mesh.bounds()
                c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                pts.append(c @ M[:3, :3].T + M[:3, 3])
            else:
                pts.append(M[:3, 3][None])
        if pts:
            p = np.vstack(pts)
            self.camera.frame((p.min(0) + p.max(0)) / 2, max(np.linalg.norm(p.max(0) - p.min(0)) / 2, 0.5))
        self.update()

    # ------------------------------------------------------------------ OpenGL
    def initializeGL(self):
        self.ctx = ctx = moderngl.create_context()
        self.p_pbr = ctx.program(vertex_shader=S.PBR_VS, fragment_shader=S.PBR_FS)
        self.p_shadow = ctx.program(vertex_shader=S.SHADOW_VS, fragment_shader=S.SHADOW_FS)
        self.p_line = ctx.program(vertex_shader=S.LINE_VS, fragment_shader=S.LINE_FS)
        self.p_vcol = ctx.program(vertex_shader=S.VCOL_VS, fragment_shader=S.VCOL_FS)
        self.p_solid = ctx.program(vertex_shader=S.SOLID_VS, fragment_shader=S.SOLID_FS)
        for name, unit in (("uAlbedoTex", 0), ("uNormalTex", 1), ("uShadow", 2), ("uEnv", 3), ("uIrr", 4)):
            U(self.p_pbr, name, unit)
        self.white = ctx.texture((1, 1), 4, bytes([255] * 4))
        self.shadow_tex = ctx.depth_texture((SHADOW_RES, SHADOW_RES))
        self.shadow_tex.compare_func = ""
        self.shadow_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.gizmo = Gizmo(ctx, self.p_solid)
        n, rows = 10, []
        for i in range(-n, n + 1):
            c = (0.52, 0.52, 0.54) if i % 5 == 0 else (0.36, 0.36, 0.38)
            rows.append([-n, 0, i, *((0.85, 0.25, 0.25) if i == 0 else c)]); rows.append([n, 0, i, *((0.85, 0.25, 0.25) if i == 0 else c)])
            rows.append([i, 0, -n, *((0.28, 0.45, 0.95) if i == 0 else c)]); rows.append([i, 0, n, *((0.28, 0.45, 0.95) if i == 0 else c)])
        self.grid_vbo = ctx.buffer(np.array(rows, "f4").tobytes())
        self.grid_vao = ctx.vertex_array(self.p_vcol, [(self.grid_vbo, "3f 3f", "in_pos", "in_col")])
        self.grid_n = len(rows)
        self.env_tex = self.irr_tex = None
        self._upload_env()

    def _upload_env(self):
        ctx = self.ctx
        for t in (self.env_tex, self.irr_tex):
            if t is not None:
                t.release()
        rgb = self.env.rgb
        h, w, _ = rgb.shape
        self.env_tex = ctx.texture((w, h), 3, np.ascontiguousarray(rgb).tobytes(), dtype="f4")
        self.env_tex.build_mipmaps()
        self.env_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.env_tex.repeat_x, self.env_tex.repeat_y = True, False
        self.env_max_lod = float(np.floor(np.log2(max(w, h))))
        ih, iw, _ = self.env.irr.shape
        self.irr_tex = ctx.texture((iw, ih), 3, np.ascontiguousarray(self.env.irr).tobytes(), dtype="f4")
        self.irr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.irr_tex.repeat_x, self.irr_tex.repeat_y = True, False
        self._env_dirty = False

    def _texture(self, path):
        if not path:
            return None
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return None
        t = self.tex_cache.get(key)
        if t is None:
            try:
                img = Image.open(path).convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            except Exception:
                return None
            t = self.ctx.texture(img.size, 4, img.tobytes())
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.anisotropy = 8.0
            t.repeat_x = t.repeat_y = True
            self.tex_cache[key] = t
        return t

    def _gpu(self, node):
        ctx, m = self.ctx, node.mesh
        c = self.cache.get(node.id)
        if c is not None and c["ver"] == m.version:
            return c
        if c is not None:
            for b in c["bufs"]:
                b.release()
        a = m.render_arrays()
        c = dict(ver=m.version, n=len(a["pos"]), bufs=[], tri_pos=a["pos"].astype(float).reshape(-1, 3, 3), tri_face=a["tri_face"])
        if c["n"]:
            vbo = ctx.buffer(np.hstack([a["pos"], a["nrm"], a["uv"]]).astype("f4").tobytes())
            c["vao"] = ctx.vertex_array(self.p_pbr, [(vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")])
            c["vao_sh"] = ctx.vertex_array(self.p_shadow, [(vbo, "3f 5x4", "in_pos")])
            c["bufs"] += [c["vao"], c["vao_sh"], vbo]
        e = m.edge_lines()
        c["ne"] = len(e)
        if len(e):
            ebo = ctx.buffer(e.tobytes())
            c["vao_e"] = ctx.vertex_array(self.p_line, [(ebo, "3f", "in_pos")])
            c["bufs"] += [c["vao_e"], ebo]
        self.cache[node.id] = c
        return c

    def _lines(self, pts, color, mvp, mode=moderngl.LINES, bias=0.0, size=1.0, rnd=0):
        if len(pts) == 0:
            return
        vbo = self.ctx.buffer(np.ascontiguousarray(pts, "f4").tobytes())
        vao = self.ctx.vertex_array(self.p_line, [(vbo, "3f", "in_pos")])
        U(self.p_line, "uMVP", gl(mvp)); U(self.p_line, "uColor", tuple(color))
        U(self.p_line, "uBias", bias); U(self.p_line, "uPointSize", size); U(self.p_line, "uRound", rnd)
        vao.render(mode)
        vao.release(); vbo.release()

    def _edges(self, g, color, mvp, bias=0.0006):
        if g["ne"]:
            U(self.p_line, "uMVP", gl(mvp)); U(self.p_line, "uColor", tuple(color))
            U(self.p_line, "uBias", bias); U(self.p_line, "uPointSize", 1.0); U(self.p_line, "uRound", 0)
            g["vao_e"].render(moderngl.LINES)

    # ------------------------------------------------------------------ luces
    def _collect_lights(self):
        out = []
        for n in self.scene.nodes.values():
            if n.kind == "light" and self.scene.is_visible(n):
                M = self.scene.world_matrix(n)
                out.append(dict(dir=n.light["type"] == "directional", pos=M[:3, 3].copy(), d=n.light_dir(M),
                                col=np.array(n.light["color"]) * n.light["intensity"], node=n))
        if not out:
            d = normalize(self.camera.target - self.camera.eye + np.array([0.0, -0.6, 0.0]) * self.camera.distance * 0.3)
            out.append(dict(dir=True, pos=np.zeros(3), d=d, col=np.array([3.0, 3.0, 3.0]), node=None))
        return out

    def _shadow_pass(self, nodes, lights):
        li = next((i for i, L in enumerate(lights[:8]) if L["dir"]), None)
        if li is None:
            return None
        lo, hi = self.scene.world_bounds()
        c, r = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2 + 0.5
        d = lights[li]["d"]
        up = (0, 1, 0) if abs(d[1]) < 0.99 else (1, 0, 0)
        LVP = ortho(-r, r, -r, r, 0.1, r * 4) @ look_at(c - d * r * 2, c, up)
        self.shadow_fbo.use()
        self.shadow_fbo.viewport = (0, 0, SHADOW_RES, SHADOW_RES)
        self.shadow_fbo.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        for n, M in nodes:
            g = self._gpu(n)
            if "vao_sh" in g:
                U(self.p_shadow, "uMVP", gl(LVP @ M))
                g["vao_sh"].render()
        return li, LVP

    # ------------------------------------------------------------------ dibujo
    def paintGL(self):
        if self._env_dirty:
            self._upload_env()
        ctx = self.ctx
        W, H = max(self.width(), 1), max(self.height(), 1)
        ratio = self.devicePixelRatioF()
        pw, ph = int(W * ratio), int(H * ratio)
        screen = ctx.detect_framebuffer(self.defaultFramebufferObject())
        for k in [k for k in self.cache if k not in self.scene.nodes]:
            for b in self.cache.pop(k)["bufs"]:
                b.release()
        cam = self.camera
        V, P = cam.view_matrix(), cam.proj_matrix(W / H)
        VP = P @ V
        lights = self._collect_lights()
        nodes = [(n, self.scene.world_matrix(n)) for n in self.scene.nodes.values()
                 if n.kind == "mesh" and n.mesh.f and self.scene.is_visible(n)]
        shadow = None
        if self.show_shadows and self.display != "wire" and nodes:
            shadow = self._shadow_pass(nodes, lights)

        screen.use()
        screen.viewport = (0, 0, pw, ph)
        screen.clear(0.27, 0.27, 0.285, 1.0)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if self.show_grid:
            U(self.p_vcol, "uMVP", gl(VP))
            self.grid_vao.render(moderngl.LINES, vertices=self.grid_n)

        sel = set(self.scene.selection)
        act = self._last_node()
        act_id = act.id if act else None
        if self.display != "wire" and nodes:
            self._setup_pbr(VP, lights, shadow)
        for n, M in nodes:
            g = self._gpu(n)
            color = ACTIVE if n.id == act_id else SEL if n.id in sel else WIRE
            if self.display == "wire":
                self._edges(g, color, VP @ M, 0.0)
                continue
            if g["n"]:
                self._draw_pbr(n, M, g)
            if self.display == "wireshaded":
                self._edges(g, color if n.id in sel else (0.1, 0.1, 0.12, 1.0), VP @ M)
            elif n.id in sel:
                self._edges(g, color, VP @ M)

        self._draw_helpers(VP, sel)
        if self.comp_mode != "object":
            self._draw_components(VP)
        self._draw_gizmo(VP, V)
        self._draw_axes(screen, ratio, pw, ph)

    def _setup_pbr(self, VP, lights, shadow):
        p = self.p_pbr
        U(p, "uVP", gl(VP)); U(p, "uCam", tuple(self.camera.eye))
        pos, col = np.zeros((8, 4), "f4"), np.zeros((8, 4), "f4")
        for i, L in enumerate(lights[:8]):
            pos[i, :3] = -L["d"] if L["dir"] else L["pos"]
            pos[i, 3] = 0.0 if L["dir"] else 1.0
            col[i, :3] = L["col"]
        U(p, "uNLights", min(len(lights), 8)); U(p, "uLPos", pos.tobytes()); U(p, "uLCol", col.tobytes())
        U(p, "uEnvInt", float(self.scene.render.env_intensity)); U(p, "uEnvMaxLod", self.env_max_lod)
        U(p, "uShadowOn", 1 if shadow else 0)
        if shadow:
            U(p, "uShadowIdx", shadow[0]); U(p, "uLightVP", gl(shadow[1]))
        else:
            U(p, "uShadowIdx", -1); U(p, "uLightVP", gl(np.eye(4)))
        self.shadow_tex.use(2); self.env_tex.use(3); self.irr_tex.use(4)

    def _draw_pbr(self, node, M, g):
        p = self.p_pbr
        mat = self.scene.materials.get(node.material, self.scene.materials[1])
        U(p, "uModel", gl(M)); U(p, "uNMat", mat3_bytes(np.linalg.inv(M[:3, :3]).T))
        U(p, "uAlbedo", tuple(mat.albedo)); U(p, "uRough", float(mat.roughness)); U(p, "uMetal", float(mat.metallic))
        at = nt = None
        if self.display == "textured":
            at, nt = self._texture(mat.albedo_tex), self._texture(mat.normal_tex)
        (at or self.white).use(0); (nt or self.white).use(1)
        U(p, "uHasAlbedo", 1 if at else 0); U(p, "uHasNormal", 1 if nt else 0)
        U(p, "uNormalStrength", float(mat.normal_strength))
        g["vao"].render(moderngl.TRIANGLES)

    def _draw_helpers(self, VP, sel):
        """Iconos de luces y pivotes de grupos."""
        for n in self.scene.nodes.values():
            if n.kind == "mesh" or not self.scene.is_visible(n):
                continue
            M = self.scene.world_matrix(n)
            o = M[:3, 3]
            color = SEL if n.id in sel else (1.0, 0.9, 0.3, 1.0)
            r = 0.3
            pts = [o - [r, 0, 0], o + [r, 0, 0], o - [0, r, 0], o + [0, r, 0], o - [0, 0, r], o + [0, 0, r]]
            if n.kind == "light" and n.light["type"] == "directional":
                d = n.light_dir(M)
                pts += [o, o + d * 1.5, o + d * 1.5, o + d * 1.2 + normalize(np.cross(d, (0, 1, 0.01))) * 0.15]
            if n.kind == "group":
                color = SEL if n.id in sel else (0.7, 0.7, 1.0, 1.0)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._lines(np.array(pts), color, VP)
            self.ctx.enable(moderngl.DEPTH_TEST)

    def _draw_components(self, VP):
        n = self.active_node()
        if n is None or not self.scene.is_visible(n) or not n.mesh.f:
            return
        M = self.scene.world_matrix(n)
        MVP = VP @ M
        m, g = n.mesh, self._gpu(n)
        self._edges(g, (0.85, 0.87, 0.92, 1.0), MVP, 0.0008)
        sel = self.comp_sel
        if sel["f"]:
            tris = []
            for fi in sel["f"]:
                if fi < len(m.f):
                    f = m.f[fi]
                    tris += [x for k in range(1, len(f) - 1) for x in (m.v[f[0]], m.v[f[k]], m.v[f[k + 1]])]
            self._lines(np.array(tris), (1.0, 0.55, 0.1, 0.45), MVP, moderngl.TRIANGLES, 0.0016)
            ed = []
            for fi in sel["f"]:
                if fi < len(m.f):
                    f = m.f[fi]
                    ed += [x for i in range(len(f)) for x in (m.v[f[i]], m.v[f[(i + 1) % len(f)]])]
            self._lines(np.array(ed), ORANGE, MVP, bias=0.002)
        if sel["e"]:
            ed = [x for a, b in sel["e"] if a < len(m.v) and b < len(m.v) for x in (m.v[a], m.v[b])]
            self._lines(np.array(ed), ORANGE, MVP, bias=0.002)
        if self.comp_mode == "vertex":
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._lines(m.v, (0.35, 0.65, 1.0, 1.0), MVP, moderngl.POINTS, 0.0, 7.0, 1)
            if sel["v"]:
                self._lines(m.v[sorted(v for v in sel["v"] if v < len(m.v))], ORANGE, MVP, moderngl.POINTS, 0.0, 10.0, 1)
            self.ctx.enable(moderngl.DEPTH_TEST)

    def _gizmo_state(self):
        if self.tool == "select":
            return None
        node = self._last_node()
        if node is None:
            return None
        M = self.scene.world_matrix(node)
        if self.comp_mode != "object" and node.kind == "mesh":
            vs = self.selected_verts()
            if not vs:
                return None
            origin = (node.mesh.v[vs] @ M[:3, :3].T + M[:3, 3]).mean(0)
            basis = np.eye(3)
        else:
            origin = M[:3, 3].copy()
            basis = np.eye(3) if self.tool != "scale" else M[:3, :3] / np.maximum(np.linalg.norm(M[:3, :3], axis=0), 1e-9)
        return origin, basis, float(np.linalg.norm(origin - self.camera.eye) * 0.16)

    def _draw_gizmo(self, VP, V):
        st = self._gizmo_state()
        if st:
            self.gizmo.draw(self.tool, st[0], st[1], st[2], VP, V, self.hover_axis)

    def _draw_axes(self, screen, ratio, pw, ph):
        ctx = self.ctx
        size = int(96 * ratio)
        screen.viewport = (int(8 * ratio), int(8 * ratio), size, size)
        R = self.camera.view_matrix()[:3, :3]
        cols = [(0.92, 0.22, 0.22), (0.3, 0.85, 0.3), (0.28, 0.48, 1.0)]
        glyph = {0: [((-.5, -.5), (.5, .5)), ((-.5, .5), (.5, -.5))],
                 1: [((-.5, .5), (0, 0)), ((.5, .5), (0, 0)), ((0, 0), (0, -.5))],
                 2: [((-.5, .5), (.5, .5)), ((.5, .5), (-.5, -.5)), ((-.5, -.5), (.5, -.5))]}
        rows = []
        for i in range(3):
            tip = R[:, i]
            rows += [[0, 0, 0, *cols[i]], [*tip, *cols[i]]]
            c = tip * 1.3
            for a, b in glyph[i]:
                rows += [[c[0] + a[0] * 0.18, c[1] + a[1] * 0.18, 0, *cols[i]], [c[0] + b[0] * 0.18, c[1] + b[1] * 0.18, 0, *cols[i]]]
        vbo = ctx.buffer(np.array(rows, "f4").tobytes())
        vao = ctx.vertex_array(self.p_vcol, [(vbo, "3f 3f", "in_pos", "in_col")])
        ctx.disable(moderngl.DEPTH_TEST)
        U(self.p_vcol, "uMVP", gl(ortho(-1.7, 1.7, -1.7, 1.7, -5, 5)))
        vao.render(moderngl.LINES)
        ctx.enable(moderngl.DEPTH_TEST)
        vao.release(); vbo.release()
        screen.viewport = (0, 0, pw, ph)

    # ------------------------------------------------------------------ proyección / picking
    def _vp(self):
        return self.camera.proj_matrix(self.width() / max(self.height(), 1)) @ self.camera.view_matrix()

    def _project(self, pts):
        W, H = self.width(), self.height()
        h = np.c_[pts, np.ones(len(pts))] @ self._vp().T
        w = h[:, 3]
        ok = w > 1e-6
        w = np.where(ok, w, 1e-6)
        return np.c_[(h[:, 0] / w * 0.5 + 0.5) * W, (1 - (h[:, 1] / w * 0.5 + 0.5)) * H], ok

    def _ray(self, pos):
        W, H = self.width(), self.height()
        x, y = pos.x() / W * 2 - 1, 1 - pos.y() / H * 2
        inv = np.linalg.inv(self._vp())
        n, f = inv @ [x, y, -1, 1], inv @ [x, y, 1, 1]
        o = n[:3] / n[3]
        return o, normalize(f[:3] / f[3] - o)

    def _pick_object(self, pos):
        ro, rd = self._ray(pos)
        best, bid = np.inf, None
        mouse = np.array([pos.x(), pos.y()])
        for n in self.scene.nodes.values():
            if not self.scene.is_visible(n):
                continue
            M = self.scene.world_matrix(n)
            if n.kind == "mesh" and n.mesh.f:
                g = self._gpu(n)
                if g["n"] == 0:
                    continue
                Mi = np.linalg.inv(M)
                t = ray_triangles(Mi[:3, :3] @ ro + Mi[:3, 3], Mi[:3, :3] @ rd, g["tri_pos"]).min()
            else:
                sp, ok = self._project(M[:3, 3][None])
                t = np.linalg.norm(M[:3, 3] - ro) if ok[0] and np.linalg.norm(sp[0] - mouse) < 14 else np.inf
            if t < best:
                best, bid = t, n.id
        return bid

    def _pick_component(self, pos, additive):
        node = self.active_node()
        m, M, mode = node.mesh, self.scene.world_matrix(node), self.comp_mode
        key = {"vertex": "v", "edge": "e", "face": "f"}[mode]
        mouse = np.array([pos.x(), pos.y()])
        sp, ok = self._project(m.v @ M[:3, :3].T + M[:3, 3])
        picked = None
        if mode == "vertex" and len(sp):
            d = np.linalg.norm(sp - mouse, axis=1); d[~ok] = 1e9
            if d.min() < 14:
                picked = int(d.argmin())
        elif mode == "edge":
            es = np.array(m.edges(), int).reshape(-1, 2)
            if len(es):
                a, b = sp[es[:, 0]], sp[es[:, 1]]
                ab = b - a
                t = np.clip(((mouse - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
                d = np.linalg.norm(a + ab * t[:, None] - mouse, axis=1)
                d[~(ok[es[:, 0]] & ok[es[:, 1]])] = 1e9
                if d.min() < 10:
                    picked = (int(es[d.argmin(), 0]), int(es[d.argmin(), 1]))
        elif mode == "face":
            g = self._gpu(node)
            if g["n"]:
                ro, rd = self._ray(pos)
                Mi = np.linalg.inv(M)
                t = ray_triangles(Mi[:3, :3] @ ro + Mi[:3, 3], Mi[:3, :3] @ rd, g["tri_pos"])
                if np.isfinite(t.min()):
                    picked = int(g["tri_face"][t.argmin()])
        if picked is None:
            if not additive:
                self.clear_comp()
        else:
            if additive:
                self.comp_sel[key] ^= {picked}
            else:
                self.clear_comp(); self.comp_sel[key].add(picked)
        self.update()

    def _click_select(self, pos, additive):
        node = self.active_node()
        if self.comp_mode != "object" and node is not None:
            self._pick_component(pos, additive)
            return
        nid = self._pick_object(pos)
        sel = self.scene.selection
        if nid is None:
            if not additive:
                sel.clear()
        elif additive:
            sel.remove(nid) if nid in sel else sel.append(nid)
        else:
            sel[:] = [nid]
        self.clear_comp()
        self.selectionChanged.emit()
        self.update()

    # ------------------------------------------------------------------ gizmo: arrastre
    def _begin_drag(self, axis, pos):
        origin, basis, size = self._gizmo_state()
        ro, rd = self._ray(pos)
        self.editStarted.emit()
        d = dict(tool=self.tool, axis=axis, origin=origin, basis=basis, size=size, mouse0=np.array([pos.x(), pos.y()]), t0=0.0, a0=0.0)
        if axis < 3:
            A = basis[:, axis]
            if self.tool == "rotate":
                d["a0"] = ring_angle(origin, A, ro, rd) or 0.0
            else:
                d["t0"] = closest_param(origin, A, ro, rd)
        if self.comp_mode != "object" and self.active_node() is not None:
            n = self.active_node(); M = self.scene.world_matrix(n)
            d.update(comp=True, node=n.id, v0=n.mesh.v.copy(), verts=self.selected_verts(), M=M, Minv=np.linalg.inv(M))
        else:
            d.update(comp=False, nodes=[(i, self.scene.nodes[i].t.copy(), self.scene.nodes[i].r.copy(), self.scene.nodes[i].s.copy())
                                        for i in self.scene.selection if i in self.scene.nodes])
        self.drag = d

    def _update_drag(self, pos):
        d = self.drag
        ro, rd = self._ray(pos)
        axis, tool = d["axis"], d["tool"]
        A = d["basis"][:, axis] if axis < 3 else None
        delta, R, f = np.zeros(3), np.eye(3), np.ones(3)
        if tool == "move":
            delta = (closest_param(d["origin"], A, ro, rd) - d["t0"]) * A
        elif tool == "rotate":
            a = ring_angle(d["origin"], A, ro, rd)
            if a is None:
                return
            R = axis_rotation(A, np.degrees(a - d["a0"]))
        else:
            if axis < 3:
                f[axis] = max(1 + (closest_param(d["origin"], A, ro, rd) - d["t0"]) / d["size"], 0.01)
            else:
                f[:] = max(1 + (d["mouse0"][1] - pos.y()) * 0.01, 0.01)
        if d["comp"]:
            node = self.scene.nodes.get(d["node"])
            if node is None:
                return
            M, verts, c = d["M"], d["verts"], d["origin"]
            w = d["v0"][verts] @ M[:3, :3].T + M[:3, 3]
            if tool == "move":
                w2 = w + delta
            elif tool == "rotate":
                w2 = c + (w - c) @ R.T
            else:
                w2 = c + (w - c) @ (d["basis"] @ np.diag(f) @ d["basis"].T).T
            node.mesh.v = d["v0"].copy()
            node.mesh.v[verts] = (w2 - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
            node.mesh.touch()
        else:
            for nid, t0, r0, s0 in d["nodes"]:
                n = self.scene.nodes.get(nid)
                if n is None:
                    continue
                parent = self.scene.nodes.get(n.parent)
                Pw = self.scene.world_matrix(parent)[:3, :3] if parent else np.eye(3)
                if tool == "move":
                    n.t = t0 + np.linalg.solve(Pw, delta)
                elif tool == "rotate":
                    Rp = Pw / np.maximum(np.linalg.norm(Pw, axis=0), 1e-9)
                    n.r = matrix_to_euler(Rp.T @ R @ Rp @ euler_to_matrix(r0))
                else:
                    n.s = s0 * f
        self.transformChanged.emit()
        self.update()

    # ------------------------------------------------------------------ eventos
    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        self.last = pos
        btn, mods = e.button(), e.modifiers()
        if mods & Qt.KeyboardModifier.AltModifier:
            self.nav = {Qt.MouseButton.LeftButton: "orbit", Qt.MouseButton.MiddleButton: "pan",
                        Qt.MouseButton.RightButton: "zoom"}.get(btn)
            return
        if btn == Qt.MouseButton.MiddleButton:
            self.nav = "pan"
        elif btn == Qt.MouseButton.LeftButton:
            st = self._gizmo_state()
            axis = None
            if st:
                ro, rd = self._ray(pos)
                axis = pick(self.tool, *st, np.array([pos.x(), pos.y()]), ro, rd, self._project)
            if axis is not None:
                self._begin_drag(axis, pos)
            else:
                self._click_select(pos, bool(mods & Qt.KeyboardModifier.ShiftModifier))

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self.nav and self.last is not None:
            dx, dy = pos.x() - self.last.x(), pos.y() - self.last.y()
            if self.nav == "orbit":
                self.camera.orbit(dx, dy)
            elif self.nav == "pan":
                self.camera.pan(dx, dy, self.height())
            else:
                self.camera.zoom_drag(dx, dy)
            self.update()
        elif self.drag:
            self._update_drag(pos)
        elif self.tool != "select":
            st = self._gizmo_state()
            ax = None
            if st:
                ro, rd = self._ray(pos)
                ax = pick(self.tool, *st, np.array([pos.x(), pos.y()]), ro, rd, self._project)
            if ax != self.hover_axis:
                self.hover_axis = ax
                self.update()
        self.last = pos

    def mouseReleaseEvent(self, e):
        self.nav = None
        self.drag = None
        self.update()

    def wheelEvent(self, e):
        self.camera.zoom_wheel(e.angleDelta().y() / 120.0)
        self.update()
