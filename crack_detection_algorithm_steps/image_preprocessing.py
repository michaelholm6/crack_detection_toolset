import cv2

def load_and_preprocess_image(path: str, blur_kernel_size: tuple =(5, 5)) -> tuple:
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
    blurred = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    return image, gray, blurred