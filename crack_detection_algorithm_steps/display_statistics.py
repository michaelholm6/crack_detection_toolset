import numpy as np
import cv2

def display_statistics(cracks_and_crack_circularities, um_per_pixel, skeleton, area_of_interest, circularity_threshold):

    skeleton_length = np.sum(skeleton > 0)
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
    defect_area_fraction = defect_area_um2 / (area_of_interest_area * (um_per_pixel ** 2))
    crack_area_fraction = crack_area_um2 / (area_of_interest_area * (um_per_pixel ** 2))
    pore_area_fraction = pore_area_um2 / (area_of_interest_area * (um_per_pixel ** 2))
    
    
    skeleton_length_fraction = skeleton_length / area_of_interest_area
    
    print("-" * 20)
    print(f"\nSkeleton Length: {real_skeleton_length:.5f} um")
    print(f"Defect Area: {defect_area_um2:.5f} um^2")
    print(f"Defect Area Fraction: {defect_area_fraction:.5f}")
    print(f"Crack Area: {crack_area_um2:.5f} um^2")
    print(f"Crack Area Fraction: {crack_area_fraction:.5f}")
    print(f"Pore Area: {pore_area_um2:.5f} um^2")
    print(f"Pore Area Fraction: {pore_area_fraction:.5f}")
    print(f"Skeleton Lenth: {real_skeleton_length:.5f} um")
    print(f"Skeleton Length Fraction: {skeleton_length_fraction:.5f}/um\n")
    print("-" * 20)
    