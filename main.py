from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from label3d import Label3D
from animator.animator_v2 import Animator, VideoAnimator

class MainWindow(QMainWindow):
    def __init__(self, ):
        super().__init__()

        self.setWindowTitle("3D Labeling Tool")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.label3d = Label3D()
        layout.addWidget(self.label3d)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()