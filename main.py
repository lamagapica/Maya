"""Mini-Maya — punto de entrada. Ejecuta:  python main.py"""
import sys
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication


def main():
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)                     # 4.1 core funciona en Windows/Linux/macOS; el driver puede dar 4.5+
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    from minimaya.ui.theme import apply_dark_theme
    from minimaya.ui.main_window import MainWindow
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
