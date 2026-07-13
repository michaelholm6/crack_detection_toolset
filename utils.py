import cv2
import ctypes
import tkinter as tk
from tkinter import messagebox
from PyQt5 import QtWidgets, QtCore


_OK_BUTTON_STYLE = """
    QPushButton {
        background-color: #0078d7;
        color: white;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 16px;
    }
    QPushButton:hover {
        background-color: #005a9e;
    }
    QPushButton:pressed {
        background-color: #004578;
    }
"""


def make_ok_button(text="OK"):
    """Return a styled blue button used to advance to the next pipeline step."""
    button = QtWidgets.QPushButton(text)
    button.setStyleSheet(_OK_BUTTON_STYLE)
    button.setCursor(QtCore.Qt.PointingHandCursor)
    return button


class WorkflowCancelled(Exception):
    """Raised when the user closes (X's out of) a step window instead of
    advancing with the OK button, so the pipeline can stop gracefully."""
    pass

def resize_for_display(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = 1.0
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img, scale

def get_screen_size(scale=0.9):
    user32 = ctypes.windll.user32
    screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return int(screen_width * scale), int(screen_height * scale)

def show_instructions(title="Instructions", message="Click two points to draw the scale bar."):
    root = tk.Tk()
    root.withdraw()  # Hide main window
    messagebox.showinfo(title, message)
    root.destroy()