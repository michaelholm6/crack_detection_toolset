import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from utils import WorkflowCancelled


class ScaleBarPicker(QtWidgets.QGraphicsView):
    """Zoomable/pannable view for clicking the two ends of a scale bar.

    Clicks are recorded in scene coordinates, which map 1:1 to original image
    pixels regardless of zoom, so the user can zoom in for sub-pixel precision
    instead of being limited by the on-screen display size. Mouse wheel zooms,
    right-drag pans.
    """

    def __init__(self, image, line_thickness, suppress_instructions=False):
        super().__init__()
        self.setWindowTitle("Draw Scale Bar")

        self.image = image.copy()
        self.line_thickness = line_thickness
        self.scale_bar_pts = []   # list of QtCore.QPointF in image coordinates
        self.um_per_pixel = None
        self._asking = False

        # Scene with the full-resolution image
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        qimg = QtGui.QImage(img_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.pixmap_item = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))

        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        self.pan_active = False
        self.last_mouse_pos = None
        self._initial_fit_done = False

        # Show the instructions first; only open the scale-bar window once the
        # user dismisses them.
        if not suppress_instructions:
            QtWidgets.QMessageBox.information(
                None, "Instructions",
                "Click exactly two points on the ends of the scale bar.\n\n"
                "Zoom in with the mouse wheel and right-click-drag to pan so you "
                "can place each point precisely.\n"
                "Press 'Esc' to cancel and close the application.\n")

        self.showMaximized()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_fit_done:
            self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
            self._initial_fit_done = True

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Escape:
            sys.exit(0)  # quit python script entirely
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        self.viewport().update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self._asking or len(self.scale_bar_pts) >= 2:
                return
            self.scale_bar_pts.append(self.mapToScene(event.pos()))
            self.viewport().update()
            if len(self.scale_bar_pts) == 2:
                self._asking = True
                QtCore.QTimer.singleShot(100, self.ask_real_length)
        elif event.button() == QtCore.Qt.RightButton:
            self.pan_active = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan_active and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self.pan_active = False
            self.setCursor(QtCore.Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)  # draw the image normally
        if not self.scale_bar_pts:
            return

        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)

        # Map scene (image) coordinates to viewport so markers stay a constant
        # on-screen size no matter the zoom level.
        pts_vp = [self.mapFromScene(pt) for pt in self.scale_bar_pts]
        for p in pts_vp:
            painter.drawEllipse(p, 5, 5)
        if len(pts_vp) == 2:
            painter.drawLine(pts_vp[0], pts_vp[1])
        painter.end()

    def ask_real_length(self):
        length, ok = QtWidgets.QInputDialog.getDouble(
            self, "Scale Bar Length",
            "Enter real-world length of the scale bar (in millimeters):",
            0.0,
            0.0001,
            1e6,
            4
        )

        if ok and length > 0:
            p1, p2 = self.scale_bar_pts
            pixel_length = np.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            if pixel_length > 0:
                self.um_per_pixel = (length * 1000) / pixel_length
                self.close()
                return

        # Cancelled or invalid: reset so the user can pick again
        self.scale_bar_pts = []
        self._asking = False
        self.viewport().update()


def get_scale_from_user(image, line_thickness, suppress_instructions=False):
    """Show an interactive window for calibrating the physical scale.

    The user clicks two endpoints on a scale bar and enters its real-world
    length in millimeters. The view can be zoomed and panned so the endpoints
    can be placed with sub-pixel precision. Returns µm per pixel as a float.
    Raises ValueError if the user closes without completing calibration.
    """
    app = QtWidgets.QApplication.instance()
    close_app_after = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        close_app_after = True

    picker = ScaleBarPicker(image, line_thickness, suppress_instructions)
    picker.show()
    app.exec_()

    scale = getattr(picker, "um_per_pixel", None)

    if close_app_after:
        app.quit()

    if scale is None:
        # Window was closed (X'd out) without completing calibration
        raise WorkflowCancelled("Scale-bar calibration was cancelled.")

    return scale
