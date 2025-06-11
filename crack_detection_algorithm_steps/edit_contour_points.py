import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import utils

class ContourEditor(QtWidgets.QGraphicsView):
    #TODO: Add option to translate selected contours by dragging mouse
    #TODO: Add option to rotate selected contours by dragging mouse
    def __init__(self, image, contours, parent=None):
        super().__init__(parent)
        self.image = image
        self.original_contours = contours
        self.contours = [c.copy() for c in contours]
        self.selected_points = set()
        self.scaling_active = False
        self.mouse_pos = None
        self.scaling_initial_distance = None
        self.contours_original_for_scaling = None
        self.moving_active = False
        self.move_start_mouse_pos = None
        self.contours_original_for_move = None
        self.scaling_reference = {}            # Stores original positions of selected points
        self.contour_scale_factors = {}  
        self.rotating_active = False
        self.rotation_reference = {}
        self.rotation_start_mouse_pos = None
        self.contours_original_for_rotation = None 
        self.creating_contour = False
        self.new_contour_points = [] 
        self.currently_creating_contour = False
        self.current_mode = "Selection"

        # Scene setup
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Transformations
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setMouseTracking(True)
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
        for c_idx, cnt in enumerate(self.contours):
            if self.currently_creating_contour and c_idx == len(self.contours) - 1:
                points = [tuple(pt[0]) for pt in cnt]
                cv2.polylines(img, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=2)
                for pt_idx, point in enumerate(points):
                    if (c_idx, pt_idx) in self.selected_points:
                        cv2.circle(img, point, 2, (255, 0, 0), -1)  # Blue highlight
                    else:
                        cv2.circle(img, point, 1, (0, 255, 0), -1)  # Green default
            
            else:            
                points = [tuple(pt[0]) for pt in cnt]
                cv2.polylines(img, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=2)
                for pt_idx, point in enumerate(points):
                    if (c_idx, pt_idx) in self.selected_points:
                        cv2.circle(img, point, 2, (255, 0, 0), -1)  # Blue highlight
                    else:
                        cv2.circle(img, point, 1, (0, 255, 0), -1)  # Green default
                    
        if (self.scaling_active or self.rotating_active) and self.mouse_pos:
            img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
            x1, y1 = int(self.mouse_pos.x()), int(self.mouse_pos.y())
            x2, y2 = int(img_center.x()), int(img_center.y())
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

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
            
        # if self.creating_contour and len(self.new_contour_points) > 0:
        #     for pt in self.new_contour_points:
        #         cv2.circle(img, pt, 3, (0, 255, 0), -1)  # Red points

        #     for i in range(1, len(self.new_contour_points)):
        #         if i > 0:
        #             pt1 = self.new_contour_points[i - 1]
        #             pt2 = self.new_contour_points[i]
        #             cv2.line(img, pt1, pt2, (0, 0, 255), 2)  # Green lines
        
    def paintEvent(self, event):
        super().paintEvent(event)  # Draw scene normally

        if self.current_mode:
            painter = QtGui.QPainter(self.viewport())
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0)))  # Red text
            font = QtGui.QFont("Arial", 20)
            font.setBold(True)
            painter.setFont(font)

            painter.drawText(10, 30, f"Mode: {self.current_mode}")

            painter.end()
            
    def keyPressEvent(self, event):
        print(f"Key pressed: {event.key()}")
        
        if event.key() == QtCore.Qt.Key_U:
            if not self.creating_contour and not self.scaling_active and not self.moving_active and not self.rotating_active:
            
                if self.undo_stack:
                    self.contours = self.undo_stack.pop()
                    self.selected_points.clear()
                    self.update_display()

        elif event.key() == QtCore.Qt.Key_D:
            if not self.creating_contour and not self.scaling_active and not self.moving_active and not self.rotating_active:
                self.delete_selected_points()
                self.update_display()

        elif event.key() == QtCore.Qt.Key_S:
            if not self.creating_contour and not self.moving_active and not self.rotating_active:
                self.scaling_active = not self.scaling_active
                if self.scaling_active:
                        self.current_mode = "Scaling" if self.scaling_active else ""
                        self.viewport().update()
                        self.undo_stack.append([c.copy() for c in self.contours])
                        cursor_pos = QtGui.QCursor.pos()
                        local_pos = self.mapFromGlobal(cursor_pos)
                        self.mouse_pos = self.mapToScene(local_pos)

                        # Save reference positions of selected points
                        self.scaling_reference = {}
                        self.contour_scale_factors = {}
                        self.contours_original_for_scaling = [c.copy() for c in self.contours]
                        
                        for c_idx, cnt in enumerate(self.contours):
                            any_selected = False
                            for pt_idx, pt in enumerate(cnt):
                                if tuple((c_idx, pt_idx)) in self.selected_points:
                                    self.scaling_reference[(c_idx, pt_idx)] = tuple(pt[0])
                                    any_selected = True
                            if any_selected:
                                self.contour_scale_factors[c_idx] = 1.0

                        # Record initial distance from center
                        img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
                        mouse_vec = np.array([
                            self.mouse_pos.x() - img_center.x(),
                            self.mouse_pos.y() - img_center.y()
                        ])
                        self.scaling_initial_distance = np.linalg.norm(mouse_vec)

                else:
                    self.current_mode = "Selection"
                    self.viewport().update()
                    self.mouse_pos = None
                    self.scaling_reference = {}
                    self.contour_scale_factors = {}

                self.update_display()
            
        elif event.key() == QtCore.Qt.Key_M:
            if not self.creating_contour and not self.scaling_active and not self.rotating_active:
                self.moving_active = not self.moving_active
                if self.moving_active:
                    self.current_mode = "Moving" if self.moving_active else ""
                    self.viewport().update()
                    self.undo_stack.append([c.copy() for c in self.contours])
                    cursor_pos = QtGui.QCursor.pos()
                    local_pos = self.mapFromGlobal(cursor_pos)
                    self.move_start_mouse_pos = self.mapToScene(local_pos)
                    self.contours_original_for_move = [c.copy() for c in self.contours]
                else:
                    self.current_mode = "Selection"
                    self.viewport().update()
                    self.move_start_mouse_pos = None
                    self.contours_original_for_move = None
                self.update_display()
            
        elif event.key() == QtCore.Qt.Key_R:
            if not self.creating_contour and not self.scaling_active and not self.moving_active:
                self.rotating_active = not self.rotating_active
                if self.rotating_active:
                    self.current_mode = "Rotating" if self.rotating_active else ""
                    self.viewport().update()
                    self.undo_stack.append([c.copy() for c in self.contours])
                    cursor_pos = QtGui.QCursor.pos()
                    local_pos = self.mapFromGlobal(cursor_pos)
                    self.mouse_pos = self.mapToScene(local_pos)
                    self.rotation_reference = {}
                    self.contours_original_for_rotation = [c.copy() for c in self.contours]

                    img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
                    start_vec = np.array([
                        self.mouse_pos.x() - img_center.x(),
                        self.mouse_pos.y() - img_center.y()
                    ])
                    self.rotation_start_angle = np.arctan2(start_vec[1], start_vec[0])

                    for c_idx, cnt in enumerate(self.contours):
                        for pt_idx, pt in enumerate(cnt):
                            if (c_idx, pt_idx) in self.selected_points:
                                self.rotation_reference[(c_idx, pt_idx)] = tuple(pt[0])
                else:
                    self.current_mode = "Selection"
                    self.viewport().update()
                    self.mouse_pos = None
                    self.rotation_reference = {}
                    self.contours_original_for_rotation = None
                    self.rotation_start_angle = None
                self.update_display()
            
        elif event.key() == QtCore.Qt.Key_C:
            if self.moving_active or self.scaling_active or self.rotating_active:
                return  # Don't allow drawing during other modes

            if self.creating_contour:
                # Finish and close contour
                self.current_mode = "Selection"
                self.viewport().update()
                self.currently_creating_contour = False
                if len(self.new_contour_points) <= 3:
                    self.contours.pop()  
                self.new_contour_points.clear()
                self.creating_contour = False
            else:
                # Begin new contour
                self.current_mode = "Contour Creation" if not self.creating_contour else ""
                self.viewport().update()
                self.undo_stack.append([c.copy() for c in self.contours])
                self.creating_contour = True
                self.new_contour_points.clear()
                self.contours.append(np.array([], dtype=np.int32).reshape((-1, 1, 2)))

            self.update_display()

        else:
            super().keyPressEvent(event)
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Deactivate all modes
            if not self.creating_contour:
                self.scaling_active = False
                self.moving_active = False
                self.rotating_active = False
                self.mouse_pos = None
                self.scaling_reference = {}
                self.rotation_reference = {}
                self.contours_original_for_move = None
                self.contours_original_for_rotation = None
                self.rotation_start_angle = None
                self.currently_creating_contour = False
                self.current_mode = "Selection"
                self.viewport().update()

            # Start lasso drawing
            if not self.creating_contour and not self.scaling_active and not self.moving_active and not self.rotating_active:
                self.drawing = True
                self.lasso_points = [self.mapToScene(event.pos())]

            if self.rect_item:
                self.scene.removeItem(self.rect_item)
                self.rect_item = None
            if hasattr(self, 'lasso_item') and self.lasso_item:
                self.scene.removeItem(self.lasso_item)
                self.lasso_item = None
            if self.creating_contour:
                if not self.currently_creating_contour:
                    self.currently_creating_contour = True
                scene_pos = self.mapToScene(event.pos())
                x, y = int(scene_pos.x()), int(scene_pos.y())
                self.new_contour_points.append((x, y))
                self.contours[-1] = np.array(self.new_contour_points, dtype=np.int32).reshape((-1, 1, 2))

            self.update_display()

        elif event.button() == QtCore.Qt.RightButton:
            self.pan_active = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

        super().mousePressEvent(event)
        
    def move_selected_points(self):
        if not self.selected_points or not self.move_start_mouse_pos or not self.mouse_pos:
            return

        # Calculate movement delta from move start
        dx = self.mouse_pos.x() - self.move_start_mouse_pos.x()
        dy = self.mouse_pos.y() - self.move_start_mouse_pos.y()

        # Reset contours to original snapshot to avoid cumulative translation
        self.contours = [c.copy() for c in self.contours_original_for_move]

        for c_idx, cnt in enumerate(self.contours):
            for pt_idx, pt in enumerate(cnt):
                if (c_idx, pt_idx) in self.selected_points:
                    x, y = pt[0]
                    new_x = int(x + dx)
                    new_y = int(y + dy)
                    cnt[pt_idx][0] = [new_x, new_y]

    def mouseMoveEvent(self, event):
        self.mouse_pos = self.mapToScene(event.pos())
        if self.drawing:
            point = self.mouse_pos
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
            self.viewport().update()
        elif self.scaling_active:
            self.scale_selected_points()
        elif self.moving_active:
            self.move_selected_points()
        elif self.rotating_active:
            self.rotate_selected_points()
        self.update_display()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == QtCore.Qt.LeftButton:
            self.drawing = False

            # If fewer than 3 points, treat it as a click: deselect all
            if hasattr(self, 'lasso_points') and len(self.lasso_points) < 3:
                self.selected_points.clear()

            elif hasattr(self, 'lasso_points') and len(self.lasso_points) >= 3:
                polygon = QtGui.QPolygonF(self.lasso_points)
                self.select_points_in_polygon(polygon)

            if hasattr(self, 'lasso_item') and self.lasso_item:
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

    def select_points_in_polygon(self, polygon):
        self.selected_points.clear()
        for c_idx, cnt in enumerate(self.contours):
            for pt_idx, pt in enumerate(cnt):
                x, y = pt[0]
                pointf = QtCore.QPointF(x, y)
                if polygon.containsPoint(pointf, QtCore.Qt.OddEvenFill):
                    self.selected_points.add((c_idx, pt_idx))

    def get_edited_contours(self):
        return self.contours
    
    def delete_selected_points(self):
        new_contours = []
        for c_idx, cnt in enumerate(self.contours):
            new_cnt = []
            for pt_idx, pt in enumerate(cnt):
                if (c_idx, pt_idx) not in self.selected_points:
                    new_cnt.append(pt)
            if new_cnt:
                new_contours.append(np.array(new_cnt, dtype=np.int32))
        self.undo_stack.append(self.contours.copy())
        self.contours = new_contours
        self.selected_points.clear()

    def rotate_selected_points(self):
        if not self.rotation_reference or not self.mouse_pos or self.rotation_start_angle is None:
            return

        img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
        current_vec = np.array([
            self.mouse_pos.x() - img_center.x(),
            self.mouse_pos.y() - img_center.y()
        ])
        current_angle = np.arctan2(current_vec[1], current_vec[0])

        angle_delta = current_angle - self.rotation_start_angle

        for (c_idx, pt_idx), orig_pt in self.rotation_reference.items():
            cnt = self.contours_original_for_rotation[c_idx]

            # Compute centroid of the contour
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            ox, oy = orig_pt
            dx, dy = ox - cx, oy - cy
            radius = np.sqrt(dx**2 + dy**2)
            original_angle = np.arctan2(dy, dx)
            new_angle = original_angle + angle_delta

            new_x = cx + radius * np.cos(new_angle)
            new_y = cy + radius * np.sin(new_angle)

            self.contours[c_idx][pt_idx][0] = [int(new_x), int(new_y)]
        
    def scale_selected_points(self):
        if not self.selected_points or not self.mouse_pos or not self.scaling_initial_distance:
            return

        img_center = QtCore.QPointF(self.image.shape[1] / 2, self.image.shape[0] / 2)
        mouse_vec = np.array([
            self.mouse_pos.x() - img_center.x(),
            self.mouse_pos.y() - img_center.y()
        ])
        current_distance = np.linalg.norm(mouse_vec)

        # Relative scaling factor from initial mouse distance
        new_scale = max(0.1, min(5.0, current_distance / self.scaling_initial_distance))

        updated_contours = set()

        for (c_idx, pt_idx), orig_pt in self.scaling_reference.items():
            cnt_orig = self.contours_original_for_scaling[c_idx]
            cnt = self.contours[c_idx]

            # Compute contour centroid from original
            M = cv2.moments(cnt_orig)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            ox, oy = cnt_orig[pt_idx][0]
            dx, dy = ox - cx, oy - cy
            new_x = int(cx + dx * new_scale)
            new_y = int(cy + dy * new_scale)
            cnt[pt_idx][0] = [new_x, new_y]

            updated_contours.add(c_idx)

        for c_idx in updated_contours:
            self.contour_scale_factors[c_idx] = new_scale
        


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
        utils.show_instructions(message="""This screen let's you manually edit contours. To select points, circle them with your cursor.\n 
                \n -Press 'C' to create a new contour by clicking to create points. The contour will be auto-completed after leaving this mode.
                \n -Press 'D' to delete selected points.
                \n -Press 'U' to undo the last action.
                \n -Press 'S' to scale selected points.
                \n -Press 'M' to move selected points.
                \n -Press 'R' to rotate selected points.
                \n -Use the mouse wheel to zoom in and out.
                \n \n To exit a mode, press the corresponding key again or left click.""")
        
    
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