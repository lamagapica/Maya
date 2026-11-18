"""Cámara orbital estilo Maya: Alt+LMB orbitar, Alt+MMB panear, Alt+RMB / rueda zoom."""
import numpy as np
from .mathutils import look_at, perspective, normalize


class Camera:
    def __init__(self):
        self.target = np.zeros(3)
        self.distance = 12.0
        self.yaw = 40.0      # grados alrededor de Y
        self.pitch = 28.0    # grados de elevación
        self.fov = 45.0
        self.near, self.far = 0.05, 2000.0

    @property
    def eye(self):
        y, p = np.radians(self.yaw), np.radians(self.pitch)
        d = np.array([np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)])
        return self.target + self.distance * d

    def view_matrix(self):
        return look_at(self.eye, self.target, (0, 1, 0))

    def proj_matrix(self, aspect):
        return perspective(self.fov, aspect, self.near, self.far)

    def basis(self):
        f = normalize(self.target - self.eye)
        right = normalize(np.cross(f, (0, 1, 0)))
        up = np.cross(right, f)
        return right, up, f

    def orbit(self, dx, dy):
        self.yaw -= dx * 0.4
        self.pitch = float(np.clip(self.pitch + dy * 0.4, -89.0, 89.0))

    def pan(self, dx, dy, height_px):
        right, up, _ = self.basis()
        k = 2 * self.distance * np.tan(np.radians(self.fov) / 2) / max(height_px, 1)
        self.target = self.target + (-right * dx + up * dy) * k

    def zoom_drag(self, dx, dy):
        self.distance = float(np.clip(self.distance * np.exp(-(dx - dy) * 0.005), 0.05, 5000))

    def zoom_wheel(self, steps):
        self.distance = float(np.clip(self.distance * (0.88 ** steps), 0.05, 5000))

    def frame(self, center, radius):
        self.target = np.asarray(center, float)
        self.distance = max(radius, 0.05) / np.sin(np.radians(self.fov) / 2) * 1.1
