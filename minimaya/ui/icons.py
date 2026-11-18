"""Iconos vectoriales dibujados con QPainter (sin archivos externos)."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF, QBrush


def _poly(*pts):
    return QPolygonF([QPointF(x, y) for x, y in pts])


def make_icon(name, size=36):
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    line, fill, acc = QColor(215, 215, 215), QColor(95, 100, 110), QColor(255, 160, 40)
    p.setPen(QPen(line, 1.8)); p.setBrush(QBrush(fill))
    s = size / 36.0
    p.scale(s, s)
    if name == "cube":
        p.drawPolygon(_poly(18, 5, 30, 11, 30, 25, 18, 31, 6, 25, 6, 11)); p.drawLine(18, 17, 18, 31); p.drawLine(18, 17, 6, 11); p.drawLine(18, 17, 30, 11)
    elif name == "sphere":
        p.drawEllipse(QRectF(5, 5, 26, 26)); p.drawEllipse(QRectF(5, 14, 26, 8)); p.drawLine(18, 5, 18, 31)
    elif name == "cylinder":
        p.drawRect(QRectF(7, 10, 22, 17)); p.drawEllipse(QRectF(7, 6, 22, 8)); p.drawArc(QRectF(7, 23, 22, 8), 0, -180 * 16)
    elif name == "cone":
        p.drawPolygon(_poly(18, 5, 30, 27, 6, 27)); p.drawEllipse(QRectF(6, 23, 24, 8))
    elif name == "plane":
        p.drawPolygon(_poly(4, 24, 14, 10, 32, 10, 22, 24)); p.drawLine(9, 17, 27, 17); p.drawLine(18, 10, 13, 24)
    elif name == "torus":
        p.drawEllipse(QRectF(4, 9, 28, 18)); p.drawEllipse(QRectF(12, 14, 12, 8))
    elif name == "extrude":
        p.drawPolygon(_poly(6, 26, 22, 26, 28, 20, 12, 20)); p.setPen(QPen(acc, 2.4)); p.drawLine(17, 18, 17, 5); p.drawLine(12, 10, 17, 5); p.drawLine(22, 10, 17, 5)
    elif name == "bevel":
        p.drawPolygon(_poly(6, 28, 6, 14, 14, 6, 30, 6, 30, 28)); p.setPen(QPen(acc, 2.4)); p.drawLine(6, 14, 14, 6)
    elif name == "subdivide":
        p.drawRect(QRectF(6, 6, 24, 24)); p.setPen(QPen(acc, 1.6)); p.drawLine(18, 6, 18, 30); p.drawLine(6, 18, 30, 18)
    elif name == "render":
        p.drawRoundedRect(QRectF(4, 9, 28, 19), 3, 3); p.setBrush(QBrush(acc)); p.drawEllipse(QRectF(13, 13, 10, 10))
    elif name == "light":
        p.setBrush(QBrush(QColor(255, 230, 120))); p.drawEllipse(QRectF(11, 8, 14, 14)); p.drawLine(15, 27, 21, 27); p.drawLine(16, 30, 20, 30)
    elif name == "select":
        p.setBrush(QBrush(line)); p.drawPolygon(_poly(9, 5, 9, 28, 15, 22, 20, 31, 24, 29, 19, 20, 27, 20))
    elif name == "move":
        p.setPen(QPen(line, 2)); p.drawLine(18, 4, 18, 32); p.drawLine(4, 18, 32, 18)
        for pts in ((18, 4, 14, 9, 22, 9), (18, 32, 14, 27, 22, 27), (4, 18, 9, 14, 9, 22), (32, 18, 27, 14, 27, 22)):
            p.drawPolygon(_poly((pts[0], pts[1]), (pts[2], pts[3]), (pts[4], pts[5])))
    elif name == "rotate":
        p.setBrush(Qt.BrushStyle.NoBrush); p.drawArc(QRectF(5, 5, 26, 26), 40 * 16, 270 * 16); p.drawPolygon(_poly(27, 6, 31, 14, 22, 13))
    elif name == "scale":
        p.drawRect(QRectF(5, 15, 16, 16)); p.drawRect(QRectF(19, 5, 12, 12)); p.drawLine(21, 15, 29, 7)
    elif name in ("obj", "vert", "edge", "face"):
        if name == "obj":
            p.drawPolygon(_poly(18, 5, 30, 11, 30, 25, 18, 31, 6, 25, 6, 11))
        elif name == "vert":
            p.setBrush(QBrush(acc)); [p.drawEllipse(QRectF(x - 3, y - 3, 6, 6)) for x, y in ((8, 8), (28, 10), (10, 28), (27, 27))]
        elif name == "edge":
            p.setPen(QPen(acc, 3)); p.drawLine(7, 27, 29, 9)
        else:
            p.setBrush(QBrush(acc)); p.drawPolygon(_poly(7, 10, 27, 7, 30, 27, 10, 30))
    p.end()
    return QIcon(pm)
