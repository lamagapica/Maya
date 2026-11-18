"""Ventana principal: menús, shelf, viewport, Outliner, Channel Box y Material Editor."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QDockWidget, QFileDialog, QMainWindow, QMessageBox, QTabWidget, QToolBar

from ..core import meshio, primitives
from ..core.modeling import bevel_edges, catmull_clark, extrude_faces
from ..core.scene import Scene
from ..render.env import EnvMap
from ..render.viewport import Viewport
from .channel_box import ChannelBox
from .icons import make_icon
from .material_editor import MaterialEditor
from .outliner import Outliner
from .render_view import RenderView
from .uv_editor import UVEditor

PRIMS = {"cube": primitives.cube, "sphere": primitives.sphere, "cylinder": primitives.cylinder,
         "cone": primitives.cone, "plane": primitives.plane, "torus": primitives.torus}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini-Maya"); self.resize(1560, 920)
        self.scene = Scene()
        self.undo_stack, self.redo_stack = [], []
        self.viewport = Viewport(self.scene)
        self.setCentralWidget(self.viewport)
        self.outliner = Outliner(self.scene)
        self.channel = ChannelBox(self.scene)
        self.materials = MaterialEditor(self.scene, self.selected_meshes)
        self.uv_editor = self.render_view = None
        self._dock("Outliner", self.outliner, Qt.DockWidgetArea.LeftDockWidgetArea)
        tabs = QTabWidget(); tabs.addTab(self.channel, "Channel Box"); tabs.addTab(self.materials, "Hypershade")
        self._dock("Attribute Editor", tabs, Qt.DockWidgetArea.RightDockWidgetArea)
        self.resizeDocks([self.findChild(QDockWidget, "Attribute Editor")], [340], Qt.Orientation.Horizontal)

        self._build_actions()
        self._build_menus()
        self._build_toolbars()

        self.viewport.selectionChanged.connect(self._on_viewport_selection)
        self.viewport.transformChanged.connect(self.channel.refresh)
        self.viewport.editStarted.connect(self.push_undo)
        self.outliner.selectionEdited.connect(self._on_outliner_selection)
        self.outliner.renamed.connect(self.channel.refresh)
        self.outliner.deleteRequested.connect(self.delete_selected)
        self.outliner.groupRequested.connect(self.group_selected)
        for w in (self.channel, self.materials):
            w.beforeChange.connect(self.push_undo)
            w.changed.connect(self._after_edit)
        self._default_scene()

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self); d.setObjectName(title); d.setWidget(widget)
        self.addDockWidget(area, d)
        return d

    # ------------------------------------------------------------------ acciones
    def act(self, text, slot, shortcut=None, icon=None, checkable=False, tip=None):
        a = QAction(make_icon(icon) if icon else None, text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        a.setToolTip(f"{tip or text}" + (f"  ({shortcut})" if shortcut else ""))
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def _build_actions(self):
        vp = self.viewport
        self.a_new = self.act("New Scene", self.new_scene, "Ctrl+N")
        self.a_import = self.act("Import…", self.import_file, "Ctrl+I")
        self.a_export = self.act("Export All…", self.export_file, "Ctrl+Shift+E")
        self.a_undo = self.act("Undo", self.undo, "Ctrl+Z"); self.a_redo = self.act("Redo", self.redo, "Ctrl+Y")
        self.a_delete = self.act("Delete", self.delete_selected, "Delete")
        self.a_dup = self.act("Duplicate", self.duplicate_selected, "Ctrl+D")
        self.a_group = self.act("Group", self.group_selected, "Ctrl+G")
        self.a_frame = self.act("Frame Selection", vp.frame_selected, "F")
        self.prim_acts = {k: self.act(k.capitalize(), lambda _=False, k=k: self.create_primitive(k), icon=k, tip=f"Create {k}") for k in PRIMS}
        self.a_plight = self.act("Point Light", lambda: self.create_light("point"), icon="light")
        self.a_dlight = self.act("Directional Light", lambda: self.create_light("directional"), icon="light")
        self.a_extrude = self.act("Extrude", self.op_extrude, "Ctrl+E", "extrude", tip="Extrude faces (F11)")
        self.a_bevel = self.act("Bevel", self.op_bevel, "Ctrl+B", "bevel", tip="Bevel edges (F10)")
        self.a_subdiv = self.act("Smooth (Catmull-Clark)", self.op_subdivide, "3", "subdivide")
        self.a_planar = self.act("Planar Mapping", lambda: self.set_uv("planar"))
        self.a_box = self.act("Cubic Mapping", lambda: self.set_uv("box"))
        self.a_uved = self.act("UV Editor…", self.open_uv_editor)
        self.a_render = self.act("Render Frame", self.render_frame, "Ctrl+R", "render")
        self.a_hdri = self.act("Load HDRI…", self.load_hdri)
        self.a_shadows = self.act("Shadows", lambda c: self._set_flag("show_shadows", c), checkable=True); self.a_shadows.setChecked(True)
        self.a_grid = self.act("Grid", lambda c: self._set_flag("show_grid", c), checkable=True); self.a_grid.setChecked(True)
        self.tool_group = QActionGroup(self)
        self.tool_acts = {}
        for tool, key in (("select", "Q"), ("move", "W"), ("rotate", "E"), ("scale", "R")):
            a = self.act(tool.capitalize(), lambda _=False, t=tool: self.set_tool(t), key, tool, True, f"{tool.capitalize()} tool")
            self.tool_group.addAction(a); self.tool_acts[tool] = a
        self.tool_acts["select"].setChecked(True)
        self.mode_group = QActionGroup(self)
        self.mode_acts = {}
        for mode, key, icon in (("object", "F8", "obj"), ("vertex", "F9", "vert"), ("edge", "F10", "edge"), ("face", "F11", "face")):
            a = self.act(f"{mode.capitalize()} mode", lambda _=False, m=mode: self.set_comp_mode(m), key, icon, True)
            self.mode_group.addAction(a); self.mode_acts[mode] = a
        self.mode_acts["object"].setChecked(True)
        self.disp_group = QActionGroup(self)
        self.disp_acts = {}
        for mode, key, text in (("wire", "4", "Wireframe"), ("shaded", "5", "Shaded"), ("textured", "6", "Textured"),
                                ("wireshaded", "7", "Wireframe on Shaded")):
            a = self.act(text, lambda _=False, m=mode: self.viewport.set_display(m), key, None, True)
            self.disp_group.addAction(a); self.disp_acts[mode] = a
        self.disp_acts["shaded"].setChecked(True)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("File"); m.addActions([self.a_new, self.a_import, self.a_export])
        m = mb.addMenu("Edit"); m.addActions([self.a_undo, self.a_redo]); m.addSeparator(); m.addActions([self.a_dup, self.a_group, self.a_delete])
        m = mb.addMenu("Create"); m.addActions(list(self.prim_acts.values())); m.addSeparator(); m.addActions([self.a_plight, self.a_dlight])
        m = mb.addMenu("Modify"); m.addActions(list(self.tool_acts.values()))
        m = mb.addMenu("Mesh"); m.addActions([self.a_extrude, self.a_bevel, self.a_subdiv])
        m = mb.addMenu("Components"); m.addActions(list(self.mode_acts.values()))
        m = mb.addMenu("Display"); m.addActions(list(self.disp_acts.values())); m.addSeparator(); m.addActions([self.a_shadows, self.a_grid, self.a_frame])
        m = mb.addMenu("UV"); m.addActions([self.a_planar, self.a_box, self.a_uved])
        m = mb.addMenu("Render"); m.addActions([self.a_render, self.a_hdri])

    def _build_toolbars(self):
        shelf = QToolBar("Shelf"); shelf.setMovable(False)
        shelf.addActions(list(self.prim_acts.values())); shelf.addActions([self.a_plight]); shelf.addSeparator()
        shelf.addActions([self.a_extrude, self.a_bevel, self.a_subdiv]); shelf.addSeparator()
        shelf.addActions(list(self.mode_acts.values())); shelf.addSeparator(); shelf.addAction(self.a_render)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, shelf)
        tools = QToolBar("Tools"); tools.setMovable(False); tools.addActions(list(self.tool_acts.values()))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tools)

    # ------------------------------------------------------------------ estado / undo
    def _default_scene(self):
        n = self.scene.add_mesh(primitives.cube(2.0), "pCube"); n.t = np.array([0.0, 1.0, 0.0])
        self.scene.add_light("directional")
        self.scene.selection[:] = [n.id]
        self.refresh_all()
        self.viewport.camera.frame([0, 1, 0], 3.0)

    def refresh_all(self):
        self.outliner.refresh(); self.channel.refresh(); self.materials.refresh()
        if self.uv_editor:
            self.uv_editor.refresh()
        self.viewport.update()

    def _after_edit(self):
        self.viewport.update()

    def push_undo(self):
        self.undo_stack.append(self.scene.state()); self.redo_stack.clear()
        del self.undo_stack[:-60]

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.scene.state()); self.scene.set_state(self.undo_stack.pop())
            self.viewport.clear_comp(); self.refresh_all()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.scene.state()); self.scene.set_state(self.redo_stack.pop())
            self.viewport.clear_comp(); self.refresh_all()

    def _on_viewport_selection(self):
        self.outliner.sync_selection(); self.channel.refresh()
        if self.uv_editor:
            self.uv_editor.refresh()

    def _on_outliner_selection(self, ids):
        self.scene.selection[:] = ids
        self.viewport.clear_comp(); self.channel.refresh(); self.viewport.update()

    def selected_meshes(self):
        return [self.scene.nodes[i] for i in self.scene.selection if i in self.scene.nodes and self.scene.nodes[i].kind == "mesh"]

    def active_mesh_node(self):
        s = self.selected_meshes()
        return s[-1] if s else None

    def msg(self, text):
        self.statusBar().showMessage(text, 5000)

    def set_tool(self, t):
        self.viewport.set_tool(t); self.tool_acts[t].setChecked(True)

    def set_comp_mode(self, m):
        self.viewport.set_comp_mode(m); self.mode_acts[m].setChecked(True)

    def _set_flag(self, name, v):
        setattr(self.viewport, name, bool(v)); self.viewport.update()

    # ------------------------------------------------------------------ edición de escena
    def new_scene(self):
        self.push_undo()
        self.scene.nodes.clear(); self.scene.selection.clear()
        self.viewport.clear_comp(); self._default_scene()

    def create_primitive(self, kind):
        self.push_undo()
        n = self.scene.add_mesh(PRIMS[kind](), "p" + kind.capitalize())
        self.scene.selection[:] = [n.id]
        self.set_comp_mode("object"); self.refresh_all()

    def create_light(self, kind):
        self.push_undo()
        n = self.scene.add_light(kind)
        self.scene.selection[:] = [n.id]; self.refresh_all()

    def delete_selected(self):
        if self.scene.selection:
            self.push_undo(); self.scene.delete(list(self.scene.selection)); self.viewport.clear_comp(); self.refresh_all()

    def duplicate_selected(self):
        if self.scene.selection:
            self.push_undo(); self.scene.selection[:] = self.scene.duplicate(list(self.scene.selection)); self.refresh_all()

    def group_selected(self):
        if self.scene.selection:
            self.push_undo(); g = self.scene.add_group(list(self.scene.selection)); self.scene.selection[:] = [g.id]; self.refresh_all()

    # ------------------------------------------------------------------ modelado
    def op_extrude(self):
        n, faces = self.active_mesh_node(), self.viewport.comp_sel["f"]
        if n is None or self.viewport.comp_mode != "face" or not faces:
            return self.msg("Extrude: selecciona caras (F11) y haz clic en ellas.")
        self.push_undo()
        extrude_faces(n.mesh, sorted(faces), self.channel.extrude_d.value())
        self.viewport.comp_sel["v"].clear(); self.viewport.comp_sel["e"].clear()
        self.viewport.update(); self.msg("Extrude OK: mueve las caras con W.")

    def op_bevel(self):
        n = self.active_mesh_node()
        if n is None:
            return self.msg("Bevel: selecciona un objeto.")
        edges = self.viewport.comp_sel["e"] if self.viewport.comp_mode == "edge" and self.viewport.comp_sel["e"] else n.mesh.edges()
        self.push_undo()
        if not bevel_edges(n.mesh, list(edges), self.channel.bevel_w.value()):
            self.undo_stack.pop(); return self.msg("Bevel: aristas no válidas (necesitan 2 caras).")
        self.viewport.clear_comp(); self.viewport.update()

    def op_subdivide(self):
        n = self.active_mesh_node()
        if n is None:
            return self.msg("Subdivide: selecciona un objeto.")
        self.push_undo()
        catmull_clark(n.mesh, self.channel.sub_levels.value()); n.mesh.smooth = True
        self.viewport.clear_comp(); self.channel.refresh(); self.viewport.update()

    # ------------------------------------------------------------------ UV / materiales
    def set_uv(self, mode):
        n = self.active_mesh_node()
        if n:
            self.push_undo(); n.mesh.uv_mode = mode; n.mesh.touch()
            if self.uv_editor:
                self.uv_editor.refresh()
            self.viewport.update()

    def open_uv_editor(self):
        if self.uv_editor is None:
            self.uv_editor = UVEditor(self.scene, self.active_mesh_node)
            self.uv_editor.beforeChange.connect(self.push_undo); self.uv_editor.changed.connect(self.viewport.update)
        self.uv_editor.refresh(); self.uv_editor.show(); self.uv_editor.raise_()

    # ------------------------------------------------------------------ I/O
    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Importar", "", "Modelos (*.obj *.gltf *.glb)")
        if not path:
            return
        try:
            items = meshio.import_obj(path) if path.lower().endswith(".obj") else meshio.import_gltf(path)
        except Exception as e:
            return QMessageBox.warning(self, "Importar", str(e))
        self.push_undo()
        ids = [self.scene.add_mesh(m, name).id for name, m in items]
        self.scene.selection[:] = ids; self.refresh_all(); self.viewport.frame_selected()

    def export_file(self):
        path, flt = QFileDialog.getSaveFileName(self, "Exportar", "scene.obj", "Wavefront OBJ (*.obj);;glTF 2.0 (*.gltf)")
        if not path:
            return
        if "gltf" in flt and not path.lower().endswith(".gltf"):
            path += ".gltf"
        elif "obj" in flt and not path.lower().endswith(".obj"):
            path += ".obj"
        (meshio.export_gltf if path.endswith(".gltf") else meshio.export_obj)(path, self.scene)
        self.msg(f"Exportado: {path}")

    def load_hdri(self):
        path, _ = QFileDialog.getOpenFileName(self, "HDRI", "", "Entorno (*.hdr *.png *.jpg *.jpeg)")
        if path:
            try:
                self.viewport.set_env(EnvMap.from_file(path))
            except Exception as e:
                QMessageBox.warning(self, "HDRI", str(e))

    def render_frame(self):
        if not any(n.kind == "mesh" for n in self.scene.nodes.values()):
            return self.msg("No hay geometría que renderizar.")
        if self.render_view:
            self.render_view.close()
        self.render_view = RenderView(self.scene, self.viewport.camera, self.viewport.env)
        self.render_view.show()
