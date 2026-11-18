"""Render View: ejecuta el path tracer en un hilo y muestra el resultado progresivo."""
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget)
from ..render.pathtracer import PathTracer


class Worker(QThread):
    progress = Signal(int, object)

    def __init__(self, tracer):
        super().__init__()
        self.tracer, self._stop = tracer, False

    def stop(self):
        self._stop = True

    def run(self):
        self.tracer.render(lambda i, img: self.progress.emit(i, img.copy()), lambda: self._stop)


class RenderView(QWidget):
    def __init__(self, scene, camera, env):
        super().__init__()
        self.setWindowTitle("Render View"); self.resize(640, 460)
        self.tracer = PathTracer(scene, camera, env, scene.render)
        self.spp, self.img = scene.render.spp, None
        lay = QVBoxLayout(self)
        self.label = QLabel("Renderizando…"); sc = QScrollArea(); sc.setWidget(self.label); sc.setWidgetResizable(True)
        lay.addWidget(sc, 1)
        self.bar = QProgressBar(); self.bar.setRange(0, self.spp); lay.addWidget(self.bar)
        row = QHBoxLayout()
        self.stop_b, self.save_b = QPushButton("Stop"), QPushButton("Save PNG…")
        self.stop_b.clicked.connect(self._stop); self.save_b.clicked.connect(self._save); self.save_b.setEnabled(False)
        row.addWidget(self.stop_b); row.addWidget(self.save_b); lay.addLayout(row)
        self.worker = Worker(self.tracer)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(lambda: self.stop_b.setEnabled(False))
        self.worker.start()

    def _on_progress(self, i, img):
        h, w, _ = img.shape
        self.img = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.label.setPixmap(QPixmap.fromImage(self.img))
        self.bar.setValue(i); self.save_b.setEnabled(True)

    def _stop(self):
        self.worker.stop()

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar render", "render.png", "PNG (*.png)")
        if path and self.img is not None:
            self.img.save(path if path.lower().endswith(".png") else path + ".png")

    def closeEvent(self, e):
        self.worker.stop(); self.worker.wait(3000)
        super().closeEvent(e)
