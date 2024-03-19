from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

# class CustomWidget(QWidget):
#     def __init__(self, ):
#         super().__init__()

#     def keyPressEvent(self, event):
#         print("keyPressEvent", event.key())
#         super().keyPressEvent(event)

# if __name__ == "__main__":
#     app = QApplication([])

#     widget = CustomWidget()
#     widget.show()

#     simulatedEvent = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier, 'A')
#     QCoreApplication.postEvent(widget, simulatedEvent)

#     app.exec()



# from PyQt5.QtCore import QObject, QEvent

class KeyPressEater(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            print(f"Key {event.key()} is pressed")
            return True  # Event is eaten
        return False  # Event is not eaten and will be sent to the target widget
    
class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.lineEdit = QLineEdit(self)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.lineEdit)
        self.setLayout(self.layout)

        # Create an instance of the event filter
        self.eater = KeyPressEater()
        # Install the event filter on the QLineEdit
        self.lineEdit.installEventFilter(self.eater)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Event Filter Example')
        self.show()

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec())