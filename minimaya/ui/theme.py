from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    c = QColor
    p.setColor(QPalette.ColorRole.Window, c(68, 68, 68)); p.setColor(QPalette.ColorRole.WindowText, c(221, 221, 221))
    p.setColor(QPalette.ColorRole.Base, c(43, 43, 43)); p.setColor(QPalette.ColorRole.AlternateBase, c(55, 55, 55))
    p.setColor(QPalette.ColorRole.Text, c(221, 221, 221)); p.setColor(QPalette.ColorRole.Button, c(82, 82, 82))
    p.setColor(QPalette.ColorRole.ButtonText, c(221, 221, 221)); p.setColor(QPalette.ColorRole.ToolTipBase, c(30, 30, 30))
    p.setColor(QPalette.ColorRole.ToolTipText, c(230, 230, 230)); p.setColor(QPalette.ColorRole.Highlight, c(82, 133, 166))
    p.setColor(QPalette.ColorRole.HighlightedText, c(255, 255, 255)); p.setColor(QPalette.ColorRole.Link, c(110, 170, 220))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, c(120, 120, 120))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, c(120, 120, 120))
    app.setPalette(p)
    app.setStyleSheet("""
        QDockWidget::title { background:#3a3a3a; padding:4px; }
        QToolBar { background:#4a4a4a; border:0; spacing:3px; padding:2px; }
        QToolButton:checked { background:#5285a6; border-radius:3px; }
        QGroupBox { border:1px solid #333; margin-top:8px; padding-top:6px; }
        QGroupBox::title { subcontrol-origin: margin; left:8px; color:#bbb; }
        QMenuBar, QMenu { background:#3c3c3c; }
        QMenu::item:selected { background:#5285a6; }
    """)
