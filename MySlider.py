from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSlider


class MySlider(QSlider):  # 继承QSlider
    customSliderClicked = pyqtSignal(int)  # 创建信号，用于发射当前Slider的value

    def __init__(self, parent=None):
        super(QSlider, self).__init__(parent)

    def mousePressEvent(self, QMouseEvent):
        """重写鼠标点击事件"""
        super().mousePressEvent(QMouseEvent)
        pos = QMouseEvent.pos().x() / self.width()
        value = round(pos * (self.maximum() - self.minimum()) + self.minimum())
        self.setValue(value)  # 设定滑动条滑块位置为鼠标点击处
        self.customSliderClicked.emit(value)  # 发送信号
