"""UV Editor 2D: muestra las UV de la malla activa sobre la textura y permite
cambiar proyección (planar/cúbica), eje, escala y desplazamiento."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class UVCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.mesh, self.pix, self.zoom = None, None, 1.0
        self.setMinimumSize(420, 420)

    def wheelEvent(self, e):
        self.zoom = float(np.clip(self.zoom * (1.1 ** (e.angleDelta().y() / 120)), 0.2, 20)); self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(38, 38, 38))
        side = min(self.width(), self.height()) * 0.8 * self.zoom
        ox, oy = (self.width() - side) / 2, (self.height() - side) / 2
        box = QRectF(ox, oy, side, side)
        if self.pix:
            p.drawPixmap(box.toRect(), self.pix)
        else:
            p.fillRect(box, QColor(70, 70, 70))
        p.setPen(QPen(QColor(150, 150, 150), 1)); p.drawRect(box)
        m = self.mesh
        if m is None or not m.f:
            return
        uv = m.corner_uv()
        p.setPen(QPen(QColor(255, 170, 60), 1))
        s = 0
        for f in m.f:
            k = len(f)
            pts = [QPointF(ox + uv[s + i, 0] * side, oy + (1 - uv[s + i, 1]) * side) for i in range(k)]
            for i in range(k):
                p.drawLine(pts[i], pts[(i + 1) % k])
            s += k


class UVEditor(QWidget):
    beforeChange = Signal()
    changed = Signal()

    def __init__(self, scene, get_node):
        super().__init__()
        self.setWindowTitle("UV Editor"); self.resize(760, 520)
        self.scene, self.get_node, self._loading = scene, get_node, False
        lay = QHBoxLayout(self)
        self.canvas = UVCanvas(); lay.addWidget(self.canvas, 1)
        side = QVBoxLayout(); lay.addLayout(side)
        f = QFormLayout(); side.addLayout(f)
        self.mode = QComboBox(); self.mode.addItems(["box", "planar"])
        self.axis = QComboBox(); self.axis.addItems(["x", "y", "z"])
        self.su, self.sv, self.ou, self.ov = (QDoubleSpinBox() for _ in range(4))
        for w in (self.su, self.sv):
            w.setRange(0.001, 1000); w.setSingleStep(0.1)
        for w in (self.ou, self.ov):
            w.setRange(-1000, 1000); w.setSingleStep(0.1)
        for w in (self.mode, self.axis):
            w.activated.connect(self._apply)
        for w in (self.su, self.sv, self.ou, self.ov):
            w.valueChanged.connect(self._apply)
        f.addRow("Projection", self.mode); f.addRow("Planar axis", self.axis)
        f.addRow("Scale U", self.su); f.addRow("Scale V", self.sv); f.addRow("Offset U", self.ou); f.addRow("Offset V", self.ov)
        b = QPushButton("Reset"); b.clicked.connect(self._reset); side.addWidget(b); side.addStretch(1)
        self.refresh()

    def refresh(self):
        n = self.get_node()
        m = n.mesh if n else None
        self.canvas.mesh = m
        self.canvas.pix = None
        if n:
            mat = self.scene.materials.get(n.material)
            if mat and mat.albedo_tex:
                img = QImage(mat.albedo_tex)
                self.canvas.pix = QPixmap.fromImage(img) if not img.isNull() else None
        self._loading = True
        if m:
            self.mode.setCurrentText(m.uv_mode); self.axis.setCurrentText(m.uv_axis)
            self.su.setValue(m.uv_scale[0]); self.sv.setValue(m.uv_scale[1]); self.ou.setValue(m.uv_offset[0]); self.ov.setValue(m.uv_offset[1])
        self._loading = False
        self.canvas.update()

    def _apply(self, *_):
        n = self.get_node()
        if not n or self._loading:
            return
        self.beforeChange.emit()
        m = n.mesh
        m.uv_mode, m.uv_axis = self.mode.currentText(), self.axis.currentText()
        m.uv_scale, m.uv_offset = [self.su.value(), self.sv.value()], [self.ou.value(), self.ov.value()]
        m.touch(); self.canvas.update(); self.changed.emit()

    def _reset(self):
        for w, v in ((self.su, 1), (self.sv, 1), (self.ou, 0), (self.ov, 0)):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        self._apply()
