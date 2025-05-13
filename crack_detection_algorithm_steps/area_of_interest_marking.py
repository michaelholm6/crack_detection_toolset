import cv2
import utils

def get_polygon_from_user(image, show_instructions):
    """
    Display the image, let the user draw a polygon,
    and return the list of (x, y) points.

    - Left-click to add points
    - Press 'c' to close the polygon and finish
    - Press 'r' to reset and start over
    - Press 'ESC' to cancel and return None
    """
    
    if show_instructions:
    
        utils.show_instructions("Instructions", 'Outline the area of interest you\'d like to analyze. Left click to add points to your outline. Press \'c\' to close the polygon and finish marking the area of interest. Press \'r\' to reset and start over. Press \'ESC\' to cancel the process.')
    
    polygon_points = []
    temp_image, scale = utils.resize_for_display(image, *utils.get_screen_size())

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_image
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append([x, y])
            # Draw current points
            if len(polygon_points) > 1:
                cv2.line(temp_image, polygon_points[-2], polygon_points[-1], (0, 255, 0), 2)
            cv2.circle(temp_image, (x, y), 3, (0, 0, 255), -1)

    cv2.namedWindow("Draw Polygon")
    cv2.setMouseCallback("Draw Polygon", mouse_callback)

    while True:
        cv2.imshow("Draw Polygon", temp_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if len(polygon_points) > 2:
                # Close the polygon by drawing line from last to first point
                cv2.line(temp_image, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)
                cv2.imshow("Draw Polygon", temp_image)
                cv2.waitKey(500)  # short pause so user sees the closed polygon
                break
            else:
                print("Need at least 3 points to form a polygon.")

        elif key == ord('r'):
            # Reset
            polygon_points = []
            temp_image, scale = utils.resize_for_display(image, *utils.get_screen_size())

        elif key == 27:  # ESC key
            points = None
            break
    

    cv2.destroyWindow("Draw Polygon")
    return [[point * 1/scale for point in points] for points in polygon_points] if polygon_points else None 