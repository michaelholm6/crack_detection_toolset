import cv2

def load_and_preprocess_image(path: str, blur_kernel_size: tuple=(5, 5), display_post_processed_image = False, clip_limit = False, tile_grid_size = False) -> tuple:
    """
    Load an image from a file and preprocess it by converting to grayscale and applying Gaussian blur.

    Args:
        path (str): Path to the image file.
        blur_kernel_size (tuple, optional): Kernel size for the gaussian blur preprocessing step. 
        Defaults to (5, 5).

    Returns:
        tuple: A tuple containing the original image, the grayscale image, and the blurred image.
    """
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    post_processed = gray.copy()
    
    if clip_limit and tile_grid_size:
        CLAHE = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        post_processed = CLAHE.apply(gray)
    
    if blur_kernel_size[0] > 0 and blur_kernel_size[1] > 0:    
        post_processed = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    
    if display_post_processed_image:
        original = image.copy()
        post_processed = cv2.cvtColor(post_processed, cv2.COLOR_GRAY2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)
        thickness = 2
        position1 = (10, 30)
        position2 = (10, 30)

        cv2.putText(original, 'Original', position1, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(post_processed, 'Post Processed', position2, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Post-Processed Image', cv2.hconcat([original, post_processed]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image, gray, post_processed