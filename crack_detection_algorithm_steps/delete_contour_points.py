import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import utils

class ContourEditor(QtWidgets.QGraphicsView):
    def __init__(self, image, contours, parent=None):
        super().__init__(parent)
        self.image = image
        self.original_contours = contours
        self.contours = [c.copy() for c in contours]

        # Scene setup
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Transformations
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.zoom = .001
        self.pan_active = False
        self.last_mouse_pos = None
        self.undo_stack = []

        # Drawing and selection
        self.drawing = False
        self.rect_item = None
        self.start_point = None
        self.initial_fit_done = False

        # Initial render
        self.update_display()

    def update_display(self):
        # Draw image and contours
        img = self.image.copy()
        
        # Loop through contours and connect the points with red lines
        for cnt in self.contours:
            # Convert contour points to a list of tuples
            points = [tuple(pt[0]) for pt in cnt]
            
            # Draw red lines connecting the points in the contour
            cv2.polylines(img, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=2)  # Red lines
            
            # Draw green points for each contour point
            for point in points:
                cv2.circle(img, point, 1, (0, 255, 0), -1)  # Green points with radius 3

        # Convert the image to QImage for PyQt rendering
        height, width, _ = img.shape
        qimage = QtGui.QImage(img.data, width, height, 3 * width, QtGui.QImage.Format_BGR888)
        self.pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimage))

        if not self.initial_fit_done:
            # Get the size of the viewport (the window size)
            view_size = self.viewport().size()

            # Calculate the zoom factor to fit the image vertically (without margin)
            zoom_factor_vertical = 3*view_size.height() / height

            # If zooming out beyond this factor, limit it
            if self.zoom < zoom_factor_vertical:
                self.zoom = zoom_factor_vertical

            # Apply the zoom factor to the image
            self.resetTransform()  # Reset any previous transformations
            self.scale(self.zoom, self.zoom)  # Apply the zoom factor

            # Center the image vertically in the window
            self.centerOn(self.pixmap_item)

            # Mark the initial fit as done to prevent resetting on each update
            self.initial_fit_done = True
            
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_U:
            if self.undo_stack:
                self.contours = self.undo_stack.pop()
                self.update_display()
        else:
            super().keyPressEvent(event)
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lasso_points = [self.mapToScene(event.pos())]  # Start new lasso
            if self.rect_item:
                self.scene.removeItem(self.rect_item)
                self.rect_item = None
            if hasattr(self, 'lasso_item') and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
                self.lasso_item = None
        elif event.button() == QtCore.Qt.RightButton:
            self.pan_active = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing:
            point = self.mapToScene(event.pos())
            self.lasso_points.append(point)
            if hasattr(self, 'lasso_item') and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
            polygon = QtGui.QPolygonF(self.lasso_points)
            self.lasso_item = self.scene.addPolygon(polygon, QtGui.QPen(QtGui.QColor("red")))
        elif self.pan_active:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == QtCore.Qt.LeftButton:
            self.drawing = False
            if hasattr(self, 'lasso_points') and len(self.lasso_points) > 2:
                polygon = QtGui.QPolygonF(self.lasso_points)
                self.delete_points_in_polygon(polygon)
                if self.lasso_item:
                    self.scene.removeItem(self.lasso_item)
                    self.lasso_item = None
            self.update_display()
        elif event.button() == QtCore.Qt.RightButton:
            self.pan_active = False
            self.setCursor(QtCore.Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Zoom on scroll
        delta = event.angleDelta().y()
        zoom_factor = 1.2 if delta > 0 else 1 / 1.2
        self.zoom *= zoom_factor
        self.scale(zoom_factor, zoom_factor)

    def delete_points_in_polygon(self, polygon):
        # Save a deep copy of current contours to undo stack
        self.undo_stack.append([c.copy() for c in self.contours])

        new_contours = []
        for cnt in self.contours:
            new_cnt = []
            for pt in cnt:
                x, y = pt[0]
                pointf = QtCore.QPointF(x, y)
                if not polygon.containsPoint(pointf, QtCore.Qt.OddEvenFill):
                    new_cnt.append([[x, y]])
            if new_cnt:
                new_contours.append(np.array(new_cnt, dtype=np.int32))
        self.contours = new_contours

    def get_edited_contours(self):
        return self.contours


def run_contour_editor_qt(image, contours, show_instructions):
    """
    Launches the PyQt contour editor window.

    Args:
        image (np.ndarray): BGR image.
        contours (list of np.ndarray): List of contours.

    Returns:
        list of np.ndarray: Edited contours.
    """
    
    if show_instructions:
        utils.show_instructions("Instructions", 'Use your mouse and left click and drag to circle any points you want to delete. Use your scroll wheel to zoom in and out. Use the right mouse button to pan the image. Press "U" to undo the last deletion. X out of the window when you are done.')
        
    
    app = QtWidgets.QApplication(sys.argv)
    editor = ContourEditor(image, contours)
    editor.setWindowTitle("Contour Editor (PyQt)")
    editor.showMaximized()

    result = {}

    def on_exit():
        result['contours'] = editor.get_edited_contours()

    app.aboutToQuit.connect(on_exit)
    app.exec_()
    return result['contours']