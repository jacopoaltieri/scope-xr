# SCOPE-XR (Single-image Characterization Of PErformance in X-Ray systems)
# Copyright (C) 2025  Jacopo Altieri
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np
from scipy.ndimage import center_of_mass


def detect_circle_hough(
    img: np.ndarray,
    dp: float,
    min_dist: float,
    param1: int,
    param2: int,
    min_radius: int,
    max_radius: int,
    output_path: str = None,
    debug: bool = False,
) -> tuple[float, float, float] | None:
    """
    Detect a single circle in a grayscale image using the Hough Circle Transform.

    Parameters
    ----------
    img
        2D array representing the input grayscale image.
    dp
        Inverse ratio of the accumulator resolution to the image resolution.
        For example, dp=1 means the accumulator has the same resolution as the image.
    min_dist
        Minimum distance between the centers of detected circles (in pixels).
    param1
        Higher threshold for the internal Canny edge detector (lower is half).
    param2
        Accumulator threshold for the circle centers at the detection stage.
        Smaller values will detect more circles (including false ones).
    min_radius
        Minimum circle radius (in pixels) to search for.
    max_radius
        Maximum circle radius (in pixels) to search for. If <= 0, no upper limit is applied.
    debug
        If True, display the detected circle overlaid on the image in a pop-up window.
        Defaults to False.

    Returns
    -------
    x: float
        x coordinate of the detected circle center.
    y: float
        y coordinate of the detected circle center.
    r: float
        radius of the detected circle.
    None
        If no circle is found.

    Raises
    ------
    FileNotFoundError
        If the input `img` is None.
    """
    if img is None:
        raise FileNotFoundError(f"Could not open '{img}'")
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.medianBlur(img_8bit, 5)

    # Perform Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius if max_radius > 0 else None,
    )

    if circles is None:
        print(f"No circles found")
        return None

    # Round and pick the strongest circle (first one)
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    output = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
    cv2.imwrite(f"{output_path}/detected_circle.png", output)

    if debug:
        display_scale = 0.5
        output_resized = cv2.resize(output, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow("Detected Circle", output_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x, y, r


def estimate_circle(cropped: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate circle parameters using Center of Mass and Equivalent Area.
    
    Methods:
    - Center (cx, cy): Calculated via center_of_mass on the binary mask.
    - Radius: Calculated from the area (Area = pi * r^2).
    """
    h, w = cropped.shape

    # 1. Thresholding to create a binary mask
    # We use a float cast to avoid overflow during min/max calc
    img_float = cropped.astype(np.float32)
    threshold = (np.min(img_float) + np.max(img_float)) / 2.0
    
    # Create binary mask (0 or 1)
    mask = img_float >= threshold

    # Handle empty image case
    if not np.any(mask):
        return w / 2.0, h / 2.0, 0.0

    # 2. Calculate Center (y, x)
    # This uses all pixels in the mask to find the geometric centroid
    cy, cx = center_of_mass(mask)

    # 3. Calculate Radius from Area
    # Area = number of pixels in the mask
    # Area = pi * r^2  ->  r = sqrt(Area / pi)
    area = np.sum(mask)
    radius_estimate = np.sqrt(area / np.pi)

    return float(cx), float(cy), float(radius_estimate)


def is_circle_centered(
    cropped: np.ndarray, cx: float, cy: float, margin: float = 0.1
) -> bool:
    """
    Check if the estimated circle center is within `margin` of the cropped image center
    in both the x- and y-directions. Returns True if it is, False otherwise.

    Parameters
    ----------
    cropped
        cropped image containing the circle
    cx
        x coordinate of the estimated circle center
    cy
        y coordinate of the estimated circle center
    margin
        allowable fraction of width/height (default 0.1)

    Returns
    -------
    bool
        True if the circle center is within the margin from the image center
    """
    h, w = cropped.shape[:2]
    center_x, center_y = w / 2, h / 2

    return (abs(cx - center_x) < margin * w) and (abs(cy - center_y) < margin * h)
