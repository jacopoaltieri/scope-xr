import numpy as np
import cv2


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
    img: 2D array representing the input grayscale image.
    dp: Inverse ratio of the accumulator resolution to the image resolution.
        For example, dp=1 means the accumulator has the same resolution as the image.
    min_dist: Minimum distance between the centers of detected circles (in pixels).
    param1: Higher threshold for the internal Canny edge detector (lower is half).
    param2: Accumulator threshold for the circle centers at the detection stage.
        Smaller values will detect more circles (including false ones).
    min_radius: Minimum circle radius (in pixels) to search for.
    max_radius: Maximum circle radius (in pixels) to search for. If <= 0, no upper limit is applied.
    debug: If True, display the detected circle overlaid on the image in a pop-up window.
        Defaults to False.

    Returns
    -------
    (x, y, r): Tuple giving the x coordinate, y coordinate of the circle center,
        and the radius r (all in pixels) of the strongest detected circle.
    None: If no circle is found.

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


def compute_com_profiles(cropped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute center-of-mass profiles along rows (x) and columns (y).
    Returns arrays of COM positions in local coordinates.
    """
    h, w = cropped.shape

    com_x = np.zeros(h)
    for i in range(h):
        row = cropped[i, :]
        total = row.sum()
        com_x[i] = (np.arange(w) * row).sum() / total

    com_y = np.zeros(w)
    for j in range(w):
        col = cropped[:, j]
        total = col.sum()
        com_y[j] = (np.arange(h) * col).sum() / total

    return com_x, com_y


        
def estimate_circle(cropped: np.ndarray, region_size: int = 10) -> tuple[float, float, float]:
    """
    Estimate the circle radius by sampling intensity profiles along
    horizontal and vertical directions, restricted to a region around the estimated center.
    """
    h, w = cropped.shape
    com_x, com_y = compute_com_profiles(cropped)
    cx_init = int(com_x.mean())
    cy_init = int(com_y.mean())

    # Define threshold relative to intensity range
    threshold = np.min(cropped) + (np.max(cropped) - np.min(cropped)) / 2

    # Limit y range for horizontal scan
    y_start = max(cy_init - region_size, 0)
    y_end = min(cy_init + region_size, h)

    x_left = np.zeros(y_end - y_start)
    x_right = np.zeros(y_end - y_start)

    for idx, y in enumerate(range(y_start, y_end)):
        row = cropped[y, :]
        left = np.argmax(row >= threshold)
        right = w - np.argmax(row[::-1] >= threshold) - 1
        x_left[idx] = left
        x_right[idx] = right

    # Limit x range for vertical scan
    x_start = max(cx_init - region_size, 0)
    x_end = min(cx_init + region_size, w)

    y_down = np.zeros(x_end - x_start)
    y_up = np.zeros(x_end - x_start)

    for idx, x in enumerate(range(x_start, x_end)):
        col = cropped[:, x]
        down = np.argmax(col >= threshold)
        up = h - np.argmax(col[::-1] >= threshold) - 1
        y_down[idx] = down
        y_up[idx] = up

    # Compute updated center and radii
    cx = np.round(np.mean((x_left + x_right) / 2))
    cy = np.round(np.mean((y_down + y_up) / 2))

    r_x = np.round(np.mean((x_right - x_left) / 2))
    r_y = np.round(np.mean((y_up - y_down) / 2))
    radius_estimate = np.round(np.mean([r_x, r_y]))

    return cx, cy, radius_estimate


def is_circle_centered(cropped, cx, cy, margin=0.1):
    """
    Check if the estimated circle center is within `margin` of the cropped image center
    in both the x- and y-directions. Returns True if it is, False otherwise.

    Parameters:
    - cropped: 2D or 3D NumPy array (h, w[, channels])
    - cx, cy: float or int, coordinates of the detected circle center
    - margin: float in (0,1), allowable fraction of width/height (default 0.1)

    Returns:
    - bool
    """
    h, w = cropped.shape[:2]
    center_x, center_y = w / 2, h / 2

    return (abs(cx - center_x) < margin * w) and (abs(cy - center_y) < margin * h)
