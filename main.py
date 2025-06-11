import argparse
from crack_detection_algorithm import main as crack_detection_main

def main():
    parser = argparse.ArgumentParser(description="Microcrack Detection CLI Tool")
    parser.add_argument('image_path', type=str, help="Path to input image.")
    parser.add_argument('--confidence_threshold', type=float, default=0.15, help="Edge detection confidence threshold (0-1). Increase this to detect fewer edges. Default is 0.15.")
    parser.add_argument('--clip_limit', type=float, default=2.0, help="Clip limit for CLAHE preprocessing step. Set to 0 to disable. Typical values range from 2.0 - 4.0. Higher values result in higher contrast. Default is 2.0.")
    parser.add_argument('--tile_grid_size', type=int, nargs=2, default=(8, 8), help="Tile grid size (width, height) for CLAHE preprocessing step. Smaller kernels result in larger local contrast. Set to (0, 0) to disable. Typical values are (8, 8) or (16, 16). Default is (8, 8).")
    parser.add_argument('--blur_kernel_size', type=int, nargs=2, default=(0, 0), help="Kernel size (width, height) for Gaussian blur preprocessing step. Blurring helps reduce noise in the image. Default is (0, 0).")
    parser.add_argument('--show_post_processed_image', action='store_true', help="If set, displays the post-processed image in a window after boosting contrast (if this is set) and applying Gaussian blur next to the original image.")
    parser.add_argument('--circularity', type=float, default=0.6, help="Minimum circularity threshold to define a contour as a pore (0-1). Increase this to decrease the number of detected pores. 1 means that a pore must be a perfect circle. Default is 0.6.")
    parser.add_argument('--rectangularity', type=float, default=0.8, help="Maximum rectangularity allowed (0-1). Decrease this to decrease the number of rectangular contours detected. Default is 0.8.")
    parser.add_argument('--crack_expansion', type=float, default=10, help="Size to expand the detected cracks (this expanding is then undone, so it doesn't skew crack area). Increase this number to solve issues like two cracks not being combined when they should be, or the edges of a crack being detected as two separate cracks. Default is 10.")
    parser.add_argument('--supress_instructions', action='store_false', help="If set, supresses instructions throughout the process.")
    parser.add_argument('--output_path', type=str, default='output.jpg', help="Path to save the output image with detected microcracks. Default is 'output.jpg'.")
    
    args = parser.parse_args()
    
    crack_detection_main(args)
    
if __name__ == "__main__":
    main()
    
    
    
    