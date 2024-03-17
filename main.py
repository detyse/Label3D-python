from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from label3d import Label3D

class MainWindow(QMainWindow):
    def __init__(self, ):
        super().__init__()

        self.setWindowTitle("3D Labeling Tool")




if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()