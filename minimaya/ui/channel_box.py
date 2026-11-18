"""Channel Box / Attribute Editor: transformación, atributos de objeto, render y herramientas."""
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)


def dspin(lo=-1e6, hi=1e6, step=0.1, dec=3):
    s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(dec); s.setKeyboardTracking(False)
    return s


class ChannelBox(QScrollArea):
    beforeChange = Signal()     # pedir snapshot de undo
    changed = Signal()          # el viewport debe repintar

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self._pending = False
        self.setWidgetResizable(True)
        root = QWidget(); lay = QVBoxLayout(root)

        g = QGroupBox("Channel Box"); f = QFormLayout(g)
        self.name = QLineEdit(); f.addRow("Name", self.name)
        self.sp = {}
        for key, label in (("t", "Translate"), ("r", "Rotate"), ("s", "Scale")):
            for i, ax in enumerate("XYZ"):
                w = dspin(step=1.0 if key == "r" else 0.1)
                w.valueChanged.connect(self._edit_transform); w.editingFinished.connect(self._done)
                self.sp[(key, i)] = w
                f.addRow(f"{label} {ax}", w)
        self.vis = QCheckBox("Visibility"); self.vis.toggled.connect(self._edit_vis); f.addRow(self.vis)
        lay.addWidget(g)

        g2 = QGroupBox("Attributes"); f2 = QFormLayout(g2)
        self.smooth = QCheckBox("Smooth shading"); self.smooth.toggled.connect(self._edit_smooth)
        self.mat = QComboBox(); self.mat.activated.connect(self._edit_mat)
        self.rough, self.metal = dspin(0, 1, 0.05, 2), dspin(0, 1, 0.05, 2)
        self.rough.valueChanged.connect(self._edit_mat_params); self.metal.valueChanged.connect(self._edit_mat_params)
        self.ltype = QComboBox(); self.ltype.addItems(["point", "directional"]); self.ltype.activated.connect(self._edit_light)
        self.lint = dspin(0, 1e5, 0.5, 2); self.lint.valueChanged.connect(self._edit_light)
        self.lcol = QPushButton("Color"); self.lcol.clicked.connect(self._pick_light_color)
        for lbl, w in (("", self.smooth), ("Material", self.mat), ("Roughness", self.rough), ("Metallic", self.metal),
                       ("Light type", self.ltype), ("Intensity", self.lint), ("Light color", self.lcol)):
            f2.addRow(lbl, w)
        self.g2 = g2
        lay.addWidget(g2)

        g3 = QGroupBox("Render Settings"); f3 = QFormLayout(g3)
        r = scene.render
        self.rs = {}
        for attr, label, w in (("width", "Width", QSpinBox()), ("height", "Height", QSpinBox()), ("spp", "Samples", QSpinBox()),
                               ("bounces", "Bounces", QSpinBox()), ("ao_radius", "AO radius", dspin(0.01, 100, 0.1, 2)),
                               ("ao_strength", "AO strength", dspin(0, 1, 0.1, 2)), ("env_intensity", "Env intensity", dspin(0, 20, 0.1, 2))):
            if isinstance(w, QSpinBox):
                w.setRange(1, 4096 if attr in ("width", "height") else 1024)
            w.valueChanged.connect(lambda v, a=attr: self._edit_render(a, v))
            self.rs[attr] = w; f3.addRow(label, w)
        lay.addWidget(g3)

        g4 = QGroupBox("Modeling Tool Settings"); f4 = QFormLayout(g4)
        self.extrude_d, self.bevel_w = dspin(-100, 100, 0.1, 3), dspin(0.001, 100, 0.05, 3)
        self.extrude_d.setValue(0.5); self.bevel_w.setValue(0.15)
        self.sub_levels = QSpinBox(); self.sub_levels.setRange(1, 4); self.sub_levels.setValue(1)
        f4.addRow("Extrude distance", self.extrude_d); f4.addRow("Bevel width", self.bevel_w); f4.addRow("Subdiv levels", self.sub_levels)
        lay.addWidget(g4)
        lay.addStretch(1)
        self.setWidget(root)
        self.name.editingFinished.connect(self._edit_name)
        self.refresh()

    # ---------------------------------------------------------------- lectura
    def node(self):
        for i in reversed(self.scene.selection):
            if i in self.scene.nodes:
                return self.scene.nodes[i]
        return None

    def refresh(self):
        n = self.node()
        widgets = [self.name, self.vis, *self.sp.values()]
        for w in widgets:
            w.blockSignals(True)
        for w in (self.smooth, self.mat, self.rough, self.metal, self.ltype, self.lint):
            w.blockSignals(True)
        for w in widgets + [self.smooth, self.mat, self.rough, self.metal, self.ltype, self.lint, self.lcol]:
            w.setEnabled(n is not None)
        if n:
            self.name.setText(n.name); self.vis.setChecked(n.visible)
            for key, arr in (("t", n.t), ("r", n.r), ("s", n.s)):
                for i in range(3):
                    self.sp[(key, i)].setValue(float(arr[i]))
            is_mesh, is_light = n.kind == "mesh", n.kind == "light"
            for w in (self.smooth, self.mat, self.rough, self.metal):
                w.setEnabled(is_mesh)
            for w in (self.ltype, self.lint, self.lcol):
                w.setEnabled(is_light)
            if is_mesh:
                self.smooth.setChecked(n.mesh.smooth)
                self.mat.clear()
                for m in self.scene.materials.values():
                    self.mat.addItem(m.name, m.id)
                idx = self.mat.findData(n.material)
                self.mat.setCurrentIndex(max(idx, 0))
                m = self.scene.materials.get(n.material)
                if m:
                    self.rough.setValue(m.roughness); self.metal.setValue(m.metallic)
            if is_light:
                self.ltype.setCurrentText(n.light["type"]); self.lint.setValue(n.light["intensity"])
        for k, w in self.rs.items():
            w.blockSignals(True); w.setValue(getattr(self.scene.render, k)); w.blockSignals(False)
        for w in widgets + [self.smooth, self.mat, self.rough, self.metal, self.ltype, self.lint]:
            w.blockSignals(False)

    # ---------------------------------------------------------------- escritura
    def _begin(self):
        if not self._pending:
            self.beforeChange.emit()
            self._pending = True

    def _done(self):
        self._pending = False

    def _edit_transform(self):
        n = self.node()
        if not n:
            return
        self._begin()
        for key, attr in (("t", n.t), ("r", n.r), ("s", n.s)):
            for i in range(3):
                attr[i] = self.sp[(key, i)].value()
        self.changed.emit()

    def _edit_name(self):
        n = self.node()
        if n and self.name.text() and self.name.text() != n.name:
            self.beforeChange.emit(); n.name = self.name.text(); self.changed.emit()

    def _edit_vis(self, v):
        n = self.node()
        if n:
            self.beforeChange.emit(); n.visible = v; self.changed.emit()

    def _edit_smooth(self, v):
        n = self.node()
        if n and n.kind == "mesh":
            self.beforeChange.emit(); n.mesh.smooth = v; n.mesh.touch(); self.changed.emit()

    def _edit_mat(self, _i):
        n = self.node()
        if n and n.kind == "mesh":
            self.beforeChange.emit(); n.material = self.mat.currentData(); self.changed.emit(); self.refresh()

    def _edit_mat_params(self):
        n = self.node()
        m = self.scene.materials.get(n.material) if n and n.kind == "mesh" else None
        if m:
            self._begin(); m.roughness, m.metallic = self.rough.value(), self.metal.value(); self.changed.emit()

    def _edit_light(self, *_):
        n = self.node()
        if n and n.kind == "light":
            self._begin(); n.light["type"] = self.ltype.currentText(); n.light["intensity"] = self.lint.value(); self.changed.emit()

    def _pick_light_color(self):
        n = self.node()
        if n and n.kind == "light":
            c = QColorDialog.getColor(QColor.fromRgbF(*n.light["color"]), self)
            if c.isValid():
                self.beforeChange.emit(); n.light["color"] = [c.redF(), c.greenF(), c.blueF()]; self.changed.emit()

    def _edit_render(self, attr, v):
        setattr(self.scene.render, attr, v)
        self.changed.emit()
