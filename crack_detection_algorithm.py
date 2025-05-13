from crack_detection_algorithm_steps.image_preprocessing import *
from crack_detection_algorithm_steps.area_of_interest_marking import *
from crack_detection_algorithm_steps.detect_cracks import *
from crack_detection_algorithm_steps.clip_cracks_to_area_of_interest import *
from crack_detection_algorithm_steps.measure_contour_circularity import *
from crack_detection_algorithm_steps.get_phyiscal_scale import *
from crack_detection_algorithm_steps.compute_crack_skeleton import *
from crack_detection_algorithm_steps.display_statistics import *
from crack_detection_algorithm_steps.save_and_display_results import *
from crack_detection_algorithm_steps.delete_contour_points import *


    

def main(args):
    image, gray, blurred = load_and_preprocess_image(args.image_path)
    area_of_interest = get_polygon_from_user(image, args.supress_instructions)
    cracks = detect_cracks(blurred, args.crack_expansion, confidence_threshold=args.confidence_threshold)
    cracks = clip_contours_with_polygon(gray.shape, cracks, area_of_interest)
    cracks = run_contour_editor_qt(image, cracks, args.supress_instructions)
    crack_circularity = measure_contour_circularity(cracks)
    crack_and_crack_circularity = [(cnt, circ) for cnt, circ in zip(cracks, crack_circularity)]
    um_per_pixel = get_scale_from_user(image, args.supress_instructions)
    skeleton = compute_skeleton(crack_and_crack_circularity, gray.shape)
    display_statistics(crack_and_crack_circularity, um_per_pixel, skeleton, area_of_interest, args.circularity)
    save_and_display_results(image, crack_and_crack_circularity, skeleton, args.output_path)
    print(f"Results saved to {args.output_path}")