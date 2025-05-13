import cv2
import ctypes
import tkinter as tk
from tkinter import messagebox

def resize_for_display(img, max_width, max_height):
    h, w = img.shape[:2]
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