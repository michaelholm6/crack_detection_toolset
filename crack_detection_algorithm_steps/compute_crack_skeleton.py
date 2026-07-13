import numpy as np
import cv2
from skimage.morphology import skeletonize


# Kulpa chain-code length coefficients. Weighting orthogonal moves by 1 and
# diagonal moves by sqrt(2) (the naive estimator) is exact for axis-aligned and
# 45-degree lines but overestimates obliquely-oriented lines by up to ~6-7%.
# Kulpa's coefficients minimise the RMS length error across all line
# orientations, cutting the worst-case error to a few percent.
_ORTHO_WEIGHT = 0.948
_DIAG_WEIGHT = 1.340


def measure_skeleton_length(skeleton):
    """Estimate the length of a 1-pixel-wide skeleton, in pixels.

    Instead of counting skeleton pixels (which treats a diagonal step, true
    length sqrt(2), the same as an orthogonal step of length 1), this walks the
    skeleton's pixel-adjacency graph and counts orthogonal and diagonal moves,
    then combines them with Kulpa's chain-code coefficients so the estimate is
    accurate across all crack orientations rather than only axis-aligned/45-deg
    ones. The spurious diagonal that appears across a right-angle corner (where
    the two pixels already share an on-skeleton orthogonal neighbour) is
    discarded so corners are not double-counted. Every unordered neighbour pair
    is counted once, so branches and junctions sum to the total network length.

    Returns the length as a float (multiply by um_per_pixel for physical units).
    """
    s = np.asarray(skeleton) > 0
    if s.sum() < 2:
        return float(s.sum())  # 0 or a single isolated pixel

    # Orthogonal neighbour pairs (count each unordered pair once)
    right = s[:, :-1] & s[:, 1:]   # (r, c)-(r, c+1)
    down = s[:-1, :] & s[1:, :]    # (r, c)-(r+1, c)
    n_ortho = int(right.sum() + down.sum())

    # Diagonal neighbour pairs
    diag_dr = s[:-1, :-1] & s[1:, 1:]   # (r, c)-(r+1, c+1)
    diag_dl = s[:-1, 1:] & s[1:, :-1]   # (r, c+1)-(r+1, c)

    # Drop a diagonal when either shared orthogonal cell is also on the
    # skeleton: that means it's a right-angle corner already measured by the
    # two orthogonal edges, not a genuine diagonal run.
    genuine_dr = diag_dr & ~(s[:-1, 1:] | s[1:, :-1])
    genuine_dl = diag_dl & ~(s[:-1, :-1] | s[1:, 1:])
    n_diag = int(genuine_dr.sum() + genuine_dl.sum())

    return n_ortho * _ORTHO_WEIGHT + n_diag * _DIAG_WEIGHT


def compute_skeleton(contours, shape, circularity):
    """Compute the medial-axis skeleton of all crack contours.

    contours is a list of (contour, circularity_value) pairs. Only contours
    whose circularity_value is below the circularity threshold are included
    (i.e., cracks, not pores). Returns a boolean skeleton array of the given shape.
    """
    binary = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(binary, [cnt for cnt, cnt_circularity in contours if cnt_circularity < circularity], -1, 255, thickness=cv2.FILLED)
    skeleton = skeletonize(binary > 0)
    return skeleton