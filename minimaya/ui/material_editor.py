"""Editor de materiales (Hypershade simplificado): PBR con albedo, roughness, metallic y normal map."""
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QColorDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
                               QListWidget, QPushButton, QVBoxLayout, QWidget)

IMG = "Imágenes (*.png *.jpg *.jpeg *.bmp *.tga)"


class MaterialEditor(QWidget):
    beforeChange = Signal()
    changed = Signal()

    def __init__(self, scene, get_selected_mesh_nodes):
        super().__init__()
        self.scene, self._sel = scene, get_selected_mesh_nodes
        lay = QVBoxLayout(self)
        self.list = QListWidget(); self.list.currentRowChanged.connect(self._load)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("New", self._new), ("Delete", self._delete), ("Assign to selection", self._assign)):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        lay.addLayout(row)
        f = QFormLayout(); lay.addLayout(f)
        self.name = QLineEdit(); self.name.editingFinished.connect(self._apply)
        self.color = QPushButton("Base color"); self.color.clicked.connect(self._pick_color)
        self.rough, self.metal, self.nstr = (QDoubleSpinBox() for _ in range(3))
        for w, hi in ((self.rough, 1), (self.metal, 1), (self.nstr, 5)):
            w.setRange(0, hi); w.setSingleStep(0.05); w.setDecimals(2); w.valueChanged.connect(self._apply)
        self.atex, self.ntex = QLineEdit(), QLineEdit()
        for e in (self.atex, self.ntex):
            e.setReadOnly(True)
        f.addRow("Name", self.name); f.addRow("Albedo", self.color)
        f.addRow("Roughness", self.rough); f.addRow("Metallic", self.metal)
        for label, edit, key in (("Albedo map", self.atex, "albedo_tex"), ("Normal map", self.ntex, "normal_tex")):
            h = QHBoxLayout(); h.addWidget(edit)
            b1, b2 = QPushButton("…"), QPushButton("✕")
            b1.clicked.connect(lambda _=False, k=key: self._browse(k)); b2.clicked.connect(lambda _=False, k=key: self._clear(k))
            h.addWidget(b1); h.addWidget(b2); f.addRow(label, h)
        f.addRow("Normal strength", self.nstr)
        self._loading = False
        self.refresh()

    def current(self):
        it = self.list.currentItem()
        return self.scene.materials.get(it.data(256)) if it else None

    def refresh(self):
        cur = self.current().id if self.current() else None
        self._loading = True
        self.list.clear()
        for m in self.scene.materials.values():
            self.list.addItem(m.name); self.list.item(self.list.count() - 1).setData(256, m.id)
            if m.id == cur:
                self.list.setCurrentRow(self.list.count() - 1)
        if self.list.currentRow() < 0 and self.list.count():
            self.list.setCurrentRow(0)
        self._loading = False
        self._load()

    def _load(self, *_):
        m = self.current()
        self._loading = True
        if m:
            self.name.setText(m.name); self.rough.setValue(m.roughness); self.metal.setValue(m.metallic)
            self.nstr.setValue(m.normal_strength); self.atex.setText(m.albedo_tex or ""); self.ntex.setText(m.normal_tex or "")
            c = QColor.fromRgbF(*m.albedo)
            self.color.setStyleSheet(f"background:{c.name()};color:{'#000' if c.lightness() > 128 else '#fff'};")
        self._loading = False

    def _apply(self):
        m = self.current()
        if m and not self._loading:
            self.beforeChange.emit()
            m.name, m.roughness, m.metallic, m.normal_strength = self.name.text() or m.name, self.rough.value(), self.metal.value(), self.nstr.value()
            self.changed.emit()

    def _pick_color(self):
        m = self.current()
        if m:
            c = QColorDialog.getColor(QColor.fromRgbF(*m.albedo), self)
            if c.isValid():
                self.beforeChange.emit(); m.albedo = [c.redF(), c.greenF(), c.blueF()]; self._load(); self.changed.emit()

    def _browse(self, key):
        m = self.current()
        path, _ = QFileDialog.getOpenFileName(self, "Textura", "", IMG)
        if m and path:
            self.beforeChange.emit(); setattr(m, key, path); self._load(); self.changed.emit()

    def _clear(self, key):
        m = self.current()
        if m:
            self.beforeChange.emit(); setattr(m, key, None); self._load(); self.changed.emit()

    def _new(self):
        self.beforeChange.emit(); self.scene.add_material(); self.refresh(); self.list.setCurrentRow(self.list.count() - 1); self.changed.emit()

    def _delete(self):
        m = self.current()
        if m and m.id != 1:
            self.beforeChange.emit()
            for n in self.scene.nodes.values():
                if n.material == m.id:
                    n.material = 1
            del self.scene.materials[m.id]; self.refresh(); self.changed.emit()

    def _assign(self):
        m = self.current()
        if m:
            self.beforeChange.emit()
            for n in self._sel():
                n.material = m.id
            self.changed.emit()
