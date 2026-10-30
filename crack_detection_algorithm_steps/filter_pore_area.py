import sys
import cv2
import numpy as np
import tkinter as tk
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel
import utils


def show_instructions_window_pore_area_filter():
    root = tk.Tk()
    root.title("Instructions")

    instructions = (
        "Instructions for Filtering Pores by Area:\n\n"
        "- Pores (shown in red) can be filtered out by their real-world area.\n"
        "- Use the Min and Max controls to set the area range, in square\n"
        "  micrometers, that a pore must fall within to be kept.\n"
        "- Pores outside the range are shown grayed out and will be removed\n"
        "  from the results when you continue.\n"
        "- Cracks (shown in blue) are not affected by this filter.\n\n"
        "When you're done, click the blue OK button to continue."
    )

    label = tk.Label(root, text=instructions, justify="left", padx=20, pady=20, font=("Helvetica", 12))
    label.pack()

    root.update_idletasks()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"+{x}+{y}")

    ok_button = tk.Button(root, text="OK", command=root.destroy, padx=10, pady=5)
    ok_button.pack(pady=(0, 20))

    root.mainloop()


class PoreAreaFilter(QtWidgets.QWidget):
    """Lets the user keep only pores whose real-world area falls in [min, max].

    Pores outside the range are previewed in gray and dropped from the
    results on OK. Cracks are always kept, regardless of this filter.
    """

    SLIDER_STEPS = 1000

    def __init__(self, image, contours_and_circularity, um_per_pixel, circularity_threshold, line_thickness, parent=None):
        super().__init__(parent)
        self.image = image
        self.contours_and_circularity = contours_and_circularity
        self.um_per_pixel = um_per_pixel
        self.circularity_threshold = circularity_threshold
        self.line_thickness = line_thickness
        self.accepted = False

        pore_areas_um2 = [
            cv2.contourArea(cnt) * (um_per_pixel ** 2)
            for cnt, circ in contours_and_circularity
            if circ >= circularity_threshold
        ]
        self.max_possible_area = max(pore_areas_um2) if pore_areas_um2 else 1.0
        if self.max_possible_area <= 0:
            self.max_possible_area = 1.0

        self.min_area = 0.0
        self.max_area = self.max_possible_area

        self.image_label = QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        self.min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_slider.setRange(0, self.SLIDER_STEPS)
        self.min_slider.setValue(0)

        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.min_spin.setDecimals(4)
        self.min_spin.setRange(0.0, self.max_possible_area)
        self.min_spin.setValue(0.0)

        self.max_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_slider.setRange(0, self.SLIDER_STEPS)
        self.max_slider.setValue(self.SLIDER_STEPS)

        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin.setDecimals(4)
        self.max_spin.setRange(0.0, self.max_possible_area)
        self.max_spin.setValue(self.max_possible_area)

        self.min_slider.valueChanged.connect(self.on_min_slider_changed)
        self.min_spin.valueChanged.connect(self.on_min_spin_changed)
        self.max_slider.valueChanged.connect(self.on_max_slider_changed)
        self.max_spin.valueChanged.connect(self.on_max_spin_changed)

        question_icon_label = QLabel()
        style = self.style()
        question_icon = style.standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
        pixmap = question_icon.pixmap(16, 16)
        question_icon_label.setPixmap(pixmap)
        question_icon_label.setToolTip(
            "Pores (red) with real-world area outside the [Min, Max] range\n"
            "below are shown grayed out and will be discarded when you continue.\n"
            "Cracks (blue) are never affected by this filter."
        )

        header_layout = QHBoxLayout()
        header_label = QLabel("Filter pores by real-world area:")
        font = header_label.font()
        font.setPointSize(16)
        header_label.setFont(font)
        header_layout.addWidget(header_label)
        header_layout.addWidget(question_icon_label)
        header_layout.addStretch()

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min area (µm²):"))
        min_row.addWidget(self.min_slider)
        min_row.addWidget(self.min_spin)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max area (µm²):"))
        max_row.addWidget(self.max_slider)
        max_row.addWidget(self.max_spin)

        self.ok_button = utils.make_ok_button("OK — Continue")
        self.ok_button.clicked.connect(self.accept_and_continue)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addLayout(header_layout)
        layout.addLayout(min_row)
        layout.addLayout(max_row)
        layout.addWidget(self.ok_button)
        self.setLayout(layout)

        self.update_display()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Escape:
            sys.exit(0)  # quit python script entirely
        else:
            super().keyPressEvent(event)

    def _slider_to_area(self, value):
        return (value / self.SLIDER_STEPS) * self.max_possible_area

    def _area_to_slider(self, area):
        return int(round((area / self.max_possible_area) * self.SLIDER_STEPS))

    def on_min_slider_changed(self, value):
        area = min(self._slider_to_area(value), self.max_area)
        self.min_area = area
        self.min_spin.blockSignals(True)
        self.min_spin.setValue(area)
        self.min_spin.blockSignals(False)
        self.update_display()

    def on_min_spin_changed(self, value):
        area = min(value, self.max_area)
        self.min_area = area
        self.min_slider.blockSignals(True)
        self.min_slider.setValue(self._area_to_slider(area))
        self.min_slider.blockSignals(False)
        self.update_display()

    def on_max_slider_changed(self, value):
        area = max(self._slider_to_area(value), self.min_area)
        self.max_area = area
        self.max_spin.blockSignals(True)
        self.max_spin.setValue(area)
        self.max_spin.blockSignals(False)
        self.update_display()

    def on_max_spin_changed(self, value):
        area = max(value, self.min_area)
        self.max_area = area
        self.max_slider.blockSignals(True)
        self.max_slider.setValue(self._area_to_slider(area))
        self.max_slider.blockSignals(False)
        self.update_display()

    def update_display(self):
        img = self.image.copy()
        for cnt, circ in self.contours_and_circularity:
            if circ < self.circularity_threshold:
                color = (255, 0, 0)  # Crack: blue, always kept
            else:
                area_um2 = cv2.contourArea(cnt) * (self.um_per_pixel ** 2)
                if self.min_area <= area_um2 <= self.max_area:
                    color = (0, 0, 255)  # Kept pore: red
                else:
                    color = (128, 128, 128)  # Filtered-out pore: gray
            cv2.drawContours(img, [cnt], -1, color, self.line_thickness)

        display_img, _ = utils.resize_for_display(img, *utils.get_screen_size(0.8))
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimage = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimage))

    def accept_and_continue(self):
        self.accepted = True
        self.close()

    def get_filtered_contours(self):
        result = []
        for cnt, circ in self.contours_and_circularity:
            if circ < self.circularity_threshold:
                result.append((cnt, circ))
                continue
            area_um2 = cv2.contourArea(cnt) * (self.um_per_pixel ** 2)
            if self.min_area <= area_um2 <= self.max_area:
                result.append((cnt, circ))
        return result


def filter_pores_by_area(image, contours_and_circularity, um_per_pixel, circularity_threshold, line_thickness, suppress_instructions=False):
    """Interactively filter pore contours by real-world area.

    Pores (circularity >= circularity_threshold) whose area in um^2 falls
    outside the user-chosen [min, max] range are discarded; cracks are
    unaffected. Area is computed using um_per_pixel, so the range is set
    after scale calibration and expressed in real-world units.
    Returns the filtered list of (contour, circularity) pairs.
    """
    if not suppress_instructions:
        show_instructions_window_pore_area_filter()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    widget = PoreAreaFilter(image, contours_and_circularity, um_per_pixel, circularity_threshold, line_thickness)
    widget.setWindowTitle("Filter Pores by Area")
    widget.setWindowFlags(widget.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
    widget.showMaximized()

    app.exec_()

    if not widget.accepted:
        raise utils.WorkflowCancelled("Pore area filtering was cancelled.")

    return widget.get_filtered_contours()
