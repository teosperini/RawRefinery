from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QPainter
from PySide6.QtCore import Qt, QSize, QRect, Signal

class ClickableImageLabel(QLabel):
    # Signal to emit when the image is clicked, passing mapped coordinates
    imageClicked = Signal(float, float)

    def __init__(self, parent=None, proportional=False):
        super().__init__(parent)
        self.original_image_size = QSize() # To store the original QImage size
        self.proportional = proportional
        self._pixmap = QPixmap()
        self._scaled = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        # The image must never dictate the size of the window.
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def set_image(self, qimage: QImage):
        self.original_image_size = qimage.size()
        self._pixmap = QPixmap.fromImage(qimage)
        self._rescale()
        self.update()

    def _rescale(self):
        if self._pixmap.isNull():
            self._scaled = QPixmap()
            return
        self._scaled = self._pixmap.scaled(self.size() * self.devicePixelRatioF(),
                                           Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation)
        self._scaled.setDevicePixelRatio(self.devicePixelRatioF())

    def _image_rect(self) -> QRect:
        """Area of the label actually covered by the image (keeps aspect ratio)."""
        if self._scaled.isNull():
            return QRect()
        ratio = self._scaled.devicePixelRatio()
        w = round(self._scaled.width() / ratio)
        h = round(self._scaled.height() / ratio)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def paintEvent(self, event):
        if self._scaled.isNull():
            super().paintEvent(event)  # placeholder text
            return
        painter = QPainter(self)
        painter.drawPixmap(self._image_rect().topLeft(), self._scaled)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            rect = self._image_rect()
            image_width = self.original_image_size.width()
            image_height = self.original_image_size.height()

            pos = event.position().toPoint()
            # Ignore clicks on the empty bars around the image
            if rect.width() > 0 and rect.height() > 0 and rect.contains(pos):
                ratio_x = (pos.x() - rect.x()) / rect.width()
                ratio_y = (pos.y() - rect.y()) / rect.height()

                if self.proportional:
                    # Position as a fraction of the image
                    self.imageClicked.emit(ratio_x, ratio_y)
                else:
                    # Map click coordinates back to original QImage dimensions
                    self.imageClicked.emit(int(ratio_x * image_width), int(ratio_y * image_height))

        super().mousePressEvent(event)
