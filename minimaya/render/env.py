"""Mapa de entorno equirectangular (HDRI). Proceduralmente por defecto, o .hdr / LDR."""
import numpy as np
from PIL import Image


def read_hdr(path):
    """Lector Radiance .hdr (RGBE, plano o RLE) -> float32 (H,W,3)."""
    with open(path, "rb") as fh:
        buf = fh.read()
    i = buf.index(b"\n\n") + 2
    j = buf.index(b"\n", i)
    parts = buf[i:j].split()
    h, w = int(parts[1]), int(parts[3])
    i = j + 1
    img = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        if 8 <= w < 32768 and buf[i] == 2 and buf[i + 1] == 2 and not (buf[i + 2] & 0x80):
            i += 4
            for c in range(4):
                x = 0
                while x < w:
                    n = buf[i]; i += 1
                    if n > 128:
                        n -= 128
                        img[y, x:x + n, c] = buf[i]; i += 1
                    else:
                        img[y, x:x + n, c] = np.frombuffer(buf[i:i + n], np.uint8); i += n
                    x += n
        else:
            img[y] = np.frombuffer(buf[i:i + 4 * w], np.uint8).reshape(w, 4); i += 4 * w
    e = img[..., 3].astype(np.int32)
    f = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)
    return (img[..., :3].astype(np.float32) * f[..., None]).astype(np.float32)


def _bilinear(img, d):
    h, w, _ = img.shape
    u = np.arctan2(d[:, 2], d[:, 0]) / (2 * np.pi) + 0.5
    v = np.arccos(np.clip(d[:, 1], -1, 1)) / np.pi
    x, y = u * w - 0.5, v * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    xa, xb = x0 % w, (x0 + 1) % w
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    return (img[ya, xa] * (1 - fx) * (1 - fy) + img[ya, xb] * fx * (1 - fy)
            + img[yb, xa] * (1 - fx) * fy + img[yb, xb] * fx * fy)


class EnvMap:
    def __init__(self, rgb, name="procedural"):
        self.rgb = np.ascontiguousarray(rgb, dtype=np.float32)
        self.name = name
        chans = [np.asarray(Image.fromarray(self.rgb[..., c], mode="F").resize((32, 16), Image.BOX)) for c in range(3)]
        self.irr = np.stack(chans, axis=-1).astype(np.float32)      # irradiancia aproximada (promedio por bloque)

    @staticmethod
    def procedural(w=256, h=128):
        v = (np.arange(h) + 0.5) / h
        u = (np.arange(w) + 0.5) / w
        th, ph = v[:, None] * np.pi, (u[None, :] - 0.5) * 2 * np.pi
        d = np.stack([np.sin(th) * np.cos(ph), np.cos(th) * np.ones_like(ph), np.sin(th) * np.sin(ph)], -1)
        t = np.clip(d[..., 1], -1, 1)[..., None]
        zenith, horizon, ground = np.array([0.25, 0.45, 0.85]), np.array([0.85, 0.9, 0.95]), np.array([0.22, 0.2, 0.18])
        up = horizon + (zenith - horizon) * np.clip(t, 0, 1) ** 0.6
        sky = np.where(t >= 0, up, ground + (horizon * 0.25 - ground) * (1 + t) ** 4)
        sun = np.array([0.5, 0.7, 0.4]); sun /= np.linalg.norm(sun)
        sky = sky + np.array([1.0, 0.9, 0.7]) * 6.0 * (np.clip((d @ sun), 0, 1) ** 220)[..., None]
        return EnvMap(sky.astype(np.float32))

    @staticmethod
    def from_file(path):
        if path.lower().endswith(".hdr"):
            rgb = read_hdr(path)
        else:
            rgb = (np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0) ** 2.2
        return EnvMap(rgb, path)

    def sample(self, dirs):
        return _bilinear(self.rgb, dirs)

    def irradiance(self, dirs):
        return _bilinear(self.irr, dirs)
