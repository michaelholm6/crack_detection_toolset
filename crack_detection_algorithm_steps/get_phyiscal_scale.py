import utils
import tkinter as tk
from tkinter import simpledialog
import numpy as np
import cv2

scale_bar_pts = []
drawing_done = False

def draw_scale_bar(event, x, y, flags, param):
    global scale_bar_pts, drawing_done

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(scale_bar_pts) < 2:
            scale_bar_pts.append((x, y))

            # Always redraw everything from the original image
            img_display = param.copy()

            # Draw all points so far
            for pt in scale_bar_pts:
                cv2.circle(img_display, pt, 5, (0, 0, 255), -1)

            # If two points, draw the line too
            if len(scale_bar_pts) == 2:
                cv2.line(img_display, scale_bar_pts[0], scale_bar_pts[1], (0, 0, 255), 2)
                drawing_done = True

            cv2.imshow("Draw Scale Bar", img_display)

def prompt_for_real_length(prompt="Enter real-world length of the scale bar (in mm):"):
    root = tk.Tk()
    root.withdraw()  # Hide main tkinter window
    user_input = simpledialog.askfloat("Scale Bar Length", prompt)
    root.destroy()
    return user_input

def get_scale_from_user(image, show_instructions):
    global drawing_done, scale_bar_pts

    display_image, _ = utils.resize_for_display(image, *utils.get_screen_size())
    cv2.imshow("Draw Scale Bar", display_image)
    cv2.setMouseCallback("Draw Scale Bar", draw_scale_bar, display_image)
    
    if show_instructions:
        utils.show_instructions("Instructions", "Click two points to draw the scale bar.")

    while not drawing_done:
        cv2.waitKey(1)

    (x1, y1), (x2, y2) = scale_bar_pts
    pixel_length = np.hypot(x2 - x1, y2 - y1)
    print(f"Scale bar pixel length: {pixel_length:.2f} px")

    real_length_um = prompt_for_real_length() * 1000
    um_per_pixel = real_length_um / pixel_length
    print(f"Scale: {um_per_pixel:.4f} µm/pixel")

    cv2.destroyWindow("Draw Scale Bar")
    return um_per_pixel