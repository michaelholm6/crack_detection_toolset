import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from crack_detection_algorithm_steps.compute_crack_skeleton import measure_skeleton_length

def display_statistics(cracks_and_crack_circularities, um_per_pixel, skeleton, area_of_interest, circularity_threshold):
    """Compute and display defect statistics in a popup dialog.

    Reports total area of interest, crack/pore areas and area fractions,
    skeleton length, and skeleton length fraction — all converted to µm units
    using um_per_pixel. Contours below circularity_threshold are classed as
    cracks; those above are pores.
    """

    skeleton_length = measure_skeleton_length(skeleton)
    real_skeleton_length = skeleton_length * um_per_pixel
    defect_area = 0
    crack_area = 0
    pore_area = 0
    
    for cnt, circularity in cracks_and_crack_circularities:
        area = cv2.contourArea(cnt)
        defect_area += area
        
        if circularity < circularity_threshold:
            crack_area += area
        else:
            pore_area += area 

    defect_area_um2 = defect_area * (um_per_pixel ** 2)
    crack_area_um2 = crack_area * (um_per_pixel ** 2)
    pore_area_um2 = pore_area * (um_per_pixel ** 2)
    area_of_interest_area = cv2.contourArea(np.array(area_of_interest, dtype=np.int32))
    aoi_area_um2 = area_of_interest_area * (um_per_pixel ** 2)

    # Guard against a degenerate (zero-area) region of interest
    if aoi_area_um2 <= 0:
        defect_area_fraction = crack_area_fraction = pore_area_fraction = 0.0
        skeleton_length_fraction = 0.0
    else:
        defect_area_fraction = defect_area_um2 / aoi_area_um2
        crack_area_fraction = crack_area_um2 / aoi_area_um2
        pore_area_fraction = pore_area_um2 / aoi_area_um2
        # Skeleton length per unit area, in real units (µm / µm² = 1/µm) to match the "/um" label below
        skeleton_length_fraction = real_skeleton_length / aoi_area_um2
    
    text = (
        "-" * 20 + "\n\n"
        f"Total Area of Interest: {area_of_interest_area * (um_per_pixel ** 2):.5f} um^2\n"
        f"Defect Area: {defect_area_um2:.5f} um^2\n"
        f"Defect Area Fraction: {defect_area_fraction:.5f}\n"
        f"Crack Area: {crack_area_um2:.5f} um^2\n"
        f"Crack Area Fraction: {crack_area_fraction:.5f}\n"
        f"Pore Area: {pore_area_um2:.5f} um^2\n"
        f"Pore Area Fraction: {pore_area_fraction:.5f}\n"
        f"Skeleton Length: {real_skeleton_length:.5f} um\n"
        f"Skeleton Length Fraction: {skeleton_length_fraction:.5f}/um\n\n"
        + "-" * 20
    )
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Analysis Summary", text)
    root.destroy()
    