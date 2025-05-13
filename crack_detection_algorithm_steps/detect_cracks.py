import numpy as np
import cv2
import sys
import os

def detect_edges(blurred,  crack_expansion, model_path='model.yml.gz', confidence_threshold=0.15):
        # Load the pretrained Structured Edge Detection model
    if getattr(sys, 'frozen', False):
    # If the app is frozen (running as a bundled exe)
        model_path = os.path.join(sys._MEIPASS, model_path)
    else:
    # If running as a script
        model_path = model_path
    
    
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    
    # Convert the image to float32 and normalize to [0, 1]
    blurred_float = blurred.astype(np.float32) / 255.0

    # If the image is grayscale, convert it to a 3-channel image
    if len(blurred_float.shape) == 2:  # Grayscale image (single channel)
        blurred_float = cv2.cvtColor(blurred_float, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

    # Detect edges (returns an edge map with confidence values)
    edges = edge_detector.detectEdges(blurred_float)  # The image is now in the correct format
    
    # Filter edges based on the confidence threshold
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
            continue  # Skip contours with too small area

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue  # Skip contours with zero perimeter

        # Calculate rectangularity (aspect ratio of the bounding box)
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_box_area = w * h
        if bounding_box_area == 0:
            continue  # Skip if bounding box area is zero

        rectangularity = area / bounding_box_area  # Between 0 and 1

        if rectangularity > max_rectangularity:
            continue  # Skip contours that are too perfectly rectangular

        filtered_contours.append(cnt)

    return filtered_contours

def detect_cracks(image, crack_expansion, confidence_threshold=0.15, min_area=10, max_rectangularity=0.8):
    # Detect edges
    dilated = detect_edges(image, crack_expansion, confidence_threshold=confidence_threshold)

    # Find and filter contours
    contours = find_and_filter_contours(dilated, min_area=min_area, max_rectangularity=max_rectangularity)

    return contours