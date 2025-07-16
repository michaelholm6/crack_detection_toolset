import sys
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import tkinter as tk

def detect_edges(blurred, crack_expansion, model_path='model.yml.gz', confidence_threshold=0.15):
    if getattr(sys, 'frozen', False):
        model_path = os.path.join(sys._MEIPASS, model_path)
    else:
        model_path = model_path

    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    blurred_float = blurred.astype(np.float32) / 255.0
    if len(blurred_float.shape) == 2:
        blurred_float = cv2.cvtColor(blurred_float, cv2.COLOR_GRAY2BGR)
    edges = edge_detector.detectEdges(blurred_float)
    filtered_edges = (edges > confidence_threshold).astype(np.uint8) * 255

    kernel = np.ones((crack_expansion, crack_expansion), np.uint8)
    dilated = cv2.dilate(filtered_edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded

def find_and_filter_contours(dilated, min_area=10, max_rectangularity=0.8):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_box_area = w * h
        if bounding_box_area == 0:
            continue
        rectangularity = area / bounding_box_area
        if rectangularity > max_rectangularity:
            continue
        filtered_contours.append(cnt)

    return filtered_contours

# --- Custom slider to jump directly to mouse click ---
class ClickableSlider(QtWidgets.QSlider):
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Compute relative position (0.0 to 1.0)
            pos_ratio = event.x() / self.width()
            pos_ratio = max(0.0, min(1.0, pos_ratio))

            # Compute exact value in slider range
            value_range = self.maximum() - self.minimum()
            new_val = round(self.minimum() + pos_ratio * value_range)

            self.setValue(new_val)
            event.accept()
        super().mousePressEvent(event)

class CrackDetectionGUI(QtWidgets.QWidget):
    def __init__(self, original_image, blurred_image, area_of_interest_pts, model_path='model.yml.gz'):
        super().__init__()
        
        self.original_image = original_image.copy()
        self.blurred_image = blurred_image
        self.model_path = model_path
        self.area_of_interest_pts = area_of_interest_pts

        self.confidence_threshold = 0.15
        self.crack_expansion = 3

        self.last_applied_confidence = self.confidence_threshold
        self.last_applied_expansion = self.crack_expansion

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Sliders
        self.conf_slider = ClickableSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.confidence_threshold * 100))
        self.conf_slider.setFixedHeight(40)
        self.conf_slider.valueChanged.connect(self.update_from_sliders)

        self.crack_slider = ClickableSlider(QtCore.Qt.Horizontal)
        self.crack_slider.setMinimum(1)
        self.crack_slider.setMaximum(20)
        self.crack_slider.setValue(self.crack_expansion)
        self.crack_slider.setFixedHeight(40)
        self.crack_slider.valueChanged.connect(self.update_from_sliders)

        # Spin boxes
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setDecimals(2)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(self.confidence_threshold)
        self.conf_spin.editingFinished.connect(self.update_from_spin_boxes)

        self.crack_spin = QtWidgets.QSpinBox()
        self.crack_spin.setRange(0, 50)
        self.crack_spin.setValue(self.crack_expansion)
        self.crack_spin.editingFinished.connect(self.update_from_spin_boxes)

        font = QtGui.QFont()
        font.setPointSize(16)

        # Create labels with question mark tooltips
        self.conf_label_container = self.make_label_with_tooltip(
            "Confidence:",
            "Sets the confidence threshold for edge detection.\n"
            "Higher values include only stronger edges,\n"
            "lower values include more edges and noise."
        )
        self.crack_label_container = self.make_label_with_tooltip(
            "Crack expansion:",
            "Controls morphological expansion of cracks.\n"
            "Higher values connect nearby edges.\n"
        )

        # Layouts
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)

        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(self.conf_label_container)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spin)

        crack_layout = QtWidgets.QHBoxLayout()
        crack_layout.addWidget(self.crack_label_container)
        crack_layout.addWidget(self.crack_slider)
        crack_layout.addWidget(self.crack_spin)

        layout.addLayout(conf_layout)
        layout.addLayout(crack_layout)

        self.setLayout(layout)
        self.setWindowTitle("Crack Detection GUI")

        self.final_contours = []

        self.showMaximized()
        self.update_image()
        
    def keyPressEvent(self, event: QtGui.QKeyEvent):
            if event.key() == QtCore.Qt.Key_Escape:
                sys.exit(0)  # quit python script entirely
            else:
                super().keyPressEvent(event)

    def make_label_with_tooltip(self, text, tooltip_text):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Use system default font, just make it larger
        font = QtGui.QFont()
        font.setPointSize(16)

        label = QtWidgets.QLabel(text)
        label.setFont(font)
        label.setToolTip(tooltip_text)

        icon_label = QtWidgets.QLabel()
        icon_pix = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        icon_label.setPixmap(icon_pix.pixmap(16, 16))
        icon_label.setToolTip(tooltip_text)

        layout.addWidget(label)
        layout.addWidget(icon_label)
        # NO stretch here
        return container
        
    def clip_contours_to_aoi(self, contours, mask_shape, polygon_pts):
        new_contours = []

        aoimask = np.zeros(mask_shape, dtype=np.uint8)
        polygon_np = np.array(polygon_pts, np.int32)
        cv2.fillPoly(aoimask, [polygon_np], 255)

        for cnt in contours:
            cnt_mask = np.zeros(mask_shape, dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            clipped_mask = cv2.bitwise_and(cnt_mask, aoimask)

            new_cnts, _ = cv2.findContours(clipped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for new_cnt in new_cnts:
                if cv2.contourArea(new_cnt) > 0:
                    new_contours.append(new_cnt)

        return new_contours
        
    def update_from_spin_boxes(self):
        new_conf = self.conf_spin.value()
        new_exp = self.crack_spin.value()

        if (
            np.isclose(new_conf, self.last_applied_confidence, atol=1e-6)
            and new_exp == self.last_applied_expansion
        ):
            return

        self.confidence_threshold = new_conf
        self.crack_expansion = new_exp

        self.conf_slider.blockSignals(True)
        self.crack_slider.blockSignals(True)
        self.conf_slider.setValue(int(self.confidence_threshold * 100))
        self.crack_slider.setValue(self.crack_expansion)
        self.conf_slider.blockSignals(False)
        self.crack_slider.blockSignals(False)

        self.update_image()

        self.last_applied_confidence = self.confidence_threshold
        self.last_applied_expansion = self.crack_expansion
        
    def update_from_sliders(self):
        raw_value = self.conf_slider.value() / 100.0
        self.confidence_threshold = max(0.01, raw_value)
        self.crack_expansion = self.crack_slider.value()

        self.conf_spin.blockSignals(True)
        self.crack_spin.blockSignals(True)
        self.conf_spin.setValue(self.confidence_threshold)
        self.crack_spin.setValue(self.crack_expansion)
        self.conf_spin.blockSignals(False)
        self.crack_spin.blockSignals(False)

        self.update_image()

        self.last_applied_confidence = self.confidence_threshold
        self.last_applied_expansion = self.crack_expansion
        
    def update_image(self):
        edges = detect_edges(self.blurred_image, self.crack_expansion, self.model_path, self.confidence_threshold)

        contours = find_and_filter_contours(edges)

        if self.area_of_interest_pts:
            contours = self.clip_contours_to_aoi(contours, self.blurred_image.shape[:2], self.area_of_interest_pts)

        self.final_contours = contours

        display_img = self.original_image.copy()
        cv2.drawContours(display_img, contours, -1, (0, 0, 255), 2)

        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = display_img_rgb.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(display_img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)

        label_width = self.image_label.width()
        label_height = self.image_label.height()
        scaled_pixmap = pixmap.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)
        
    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

    def get_final_contours(self):
        return self.final_contours

def detect_cracks(original_image, blurred_image, area_of_interest_pts, suppress_instructions=False):
    
    if not suppress_instructions:
        root = tk.Tk()
        root.title("Instructions")

        # Your instructions text
        instructions = (
            "Use the sliders or spin boxes to adjust:\n"
            "- Confidence threshold: higher filters out weak edges.\n"
            "- Crack expansion: controls crack thickness and connectivity.\n"
            "The image updates in real-time.\n"
            "Press ESC to exit and close the application.\n"
            "Close the window when done.\n"
)

        label = tk.Label(root, text=instructions, justify="left", padx=20, pady=20, font=("Helvetica", 12))
        label.pack()

        # Center the window
        root.update_idletasks()  # Make sure geometry is calculated
        window_width = root.winfo_width()
        window_height = root.winfo_height()

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        root.geometry(f"+{x}+{y}")

        # Wait for user to press OK to continue
        ok_button = tk.Button(root, text="OK", command=root.destroy, padx=10, pady=5)
        ok_button.pack(pady=(0, 20))

        root.mainloop() 
    
    app = QtWidgets.QApplication.instance()
    close_app_after = False

    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        close_app_after = True

    gui = CrackDetectionGUI(original_image, blurred_image, area_of_interest_pts)
    gui.show()

    app.exec_()

    contours = gui.get_final_contours()

    if close_app_after:
        app.quit()

    return contours