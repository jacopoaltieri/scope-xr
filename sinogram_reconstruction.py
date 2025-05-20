import os
import numpy as np
import cv2
import tifffile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import map_coordinates, shift
from skimage.transform import iradon
from scipy.special import erf
from scipy.optimize import curve_fit


def eval_minimum_magnification(a: int, n: int, p: int) -> int:
    """Evaluate the minimum magnification required to obtain a focal spot image involving a reasonable number n of pixels."""
    m = (a + n * p) / a
    return m


def eval_minimum_radius(n: int, p: int, m: int) -> int:
    """Evaluate the minimum disk radius required to obtain a focal spot image involving a reasonable number n of pixels."""
    r = (1 + n**2) * p / (2 * m)
    return r


def load_raw_as_ndarray(img_path: str) -> np.ndarray:
    """Load a raw image as a numpy ndarray using metadata from the corresponding XML file."""
    xml_path = os.path.splitext(img_path)[0] + ".xml"
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Metadata XML not found: '{xml_path}'")

    # Parse XML and get width and height
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame = root.find("frame")
    img_width = int(frame.find("imgWidth").text)
    img_height = int(frame.find("imgHeight").text)

    # Read and reshape raw data
    with open(img_path, "rb") as f:
        img = np.fromfile(f, dtype=np.uint16)
        img = img.reshape(img_height, img_width)
    return img


def detect_circle_hough(
    img: np.ndarray,
    dp: float,
    min_dist: float,
    param1: int,
    param2: int,
    min_radius: int,
    max_radius: int,
    debug: bool = False,
)-> tuple[float, float, float] | None:
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
    debug: If True, display the detected circle overlaid on the image in a pop‑up window.
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

    if debug:
        output = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

        display_scale = 0.5
        output_resized = cv2.resize(output, (0, 0), fx=display_scale, fy=display_scale)

        cv2.imshow("Detected Circle", output_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x, y, r


def crop_square_roi(
    img: np.ndarray,
    center: tuple[float, float],
    radius: float,
    width_factor: float = 1.5,
)-> np.ndarray:

    cx, cy = center
    half_w = int(radius * width_factor)

    x0 = max(cx - half_w, 0)
    x1 = min(cx + half_w, img.shape[1])
    y0 = max(cy - half_w, 0)
    y1 = min(cy + half_w, img.shape[0])

    cropped = img[y0:y1, x0:x1]
    return cropped


def compute_com_profiles(cropped: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
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


def estimate_circle(cropped: np.ndarray, com_x: float, com_y: float)-> tuple[float, float, float]:
    """
    Estimate the circle radius by sampling intensity profiles along
    horizontal and vertical directions from the estimated center.
    """
    h, w = cropped.shape
    cx = int(com_x.mean())
    cy = int(com_y.mean())

    # Define threshold relative to intensity range
    threshold = np.min(cropped) + (np.max(cropped) - np.min(cropped)) / 2

    # Horizontal scan
    x_left = np.zeros(h)
    x_right = np.zeros(h)

    for y in range(h):
        row = cropped[y, :]
        left = np.argmax(row >= threshold)
        right = w - np.argmax(row[::-1] >= threshold) - 1
        x_left[y] = left
        x_right[y] = right

    # Vertical scan
    y_down = np.zeros(w)
    y_up = np.zeros(w)

    for x in range(w):
        col = cropped[:, x]
        down = np.argmax(col >= threshold)
        up = h - np.argmax(col[::-1] >= threshold) - 1
        y_down[x] = down
        y_up[x] = up

    # Compute center and radii
    cx = np.round(np.mean((x_left + x_right) / 2))
    cy = np.round(np.mean((y_down + y_up) / 2))

    r_x = np.round(np.mean((x_right - x_left) / 2))
    r_y = np.round(np.mean((y_up - y_down) / 2))
    radius_estimate = np.round(np.mean([r_x, r_y]))

    return cx, cy, radius_estimate


def plot_circle_on_crop(
    cropped: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    output_path: str,
    show: bool = False,
)-> None:

    fig, ax = plt.subplots()
    ax.imshow(cropped, cmap="gray")

    # Draw circle
    ax.add_patch(Circle((cx, cy), radius, edgecolor="red", fill=False, linewidth=2))

    # Draw center
    ax.plot(cx, cy, "ro", markersize=5)

    ax.set_title("Estimated Radius with Center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path + "/circle_on_crop.png", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def compute_profiles_and_sinogram(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_angles: int = 360,
    profile_half_length: int = 64,
)-> tuple[np.ndarray, np.ndarray]:
    """
    Extract edge profiles around a circle and compute the sinogram.

    Args:
        cropped: 2D ndarray (grayscale image)
        cx, cy: center of the circle
        radius: radius of the circle
        n_angles: number of angles to sample (default 360)
        profile_half_length: number of pixels to sample on either side of the edge
    Returns:
        profiles: 2D array of extracted profiles for visualization
        sinogram: 2D array [angle_index, radial_profile]
    """
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    profile_length = 2 * profile_half_length

    profiles = np.zeros((n_angles, profile_length), dtype=np.float32)
    sinogram = np.zeros((n_angles, profile_length), dtype=np.float32)

    for i, theta in enumerate(angles):
        # Generate unit vector pointing outward from the circle
        nx = np.cos(theta)
        ny = np.sin(theta)

        # Sample points along the normal direction
        profile = np.zeros(profile_length)
        for j in range(profile_length):
            d = j - profile_half_length
            px = cx + (radius + d) * nx
            py = cy + (radius + d) * ny

            # Interpolate pixel value
            if 0 <= int(py) < img.shape[0] and 0 <= int(px) < img.shape[1]:
                profile[j] = map_coordinates(img, [[py], [px]], order=1)[0]
            else:
                profile[j] = 0  # Zero padding if outside image

        profiles[i, :] = profile  # Store the radial profile

    # Compute the derivative to obtain the sinogram
    sinogram = np.gradient(profiles, axis=1)
    return profiles.T, -sinogram.T


def find_best_center_shift(sinogram: np.ndarray, max_shift=None)-> int:
    """
    Find the vertical shift (in rows) that best centers the sinogram,
    by minimizing the symmetry error between the first 180° and flipped 180°.

    Parameters
    ----------
    sinogram: 2D ndarray, shape (n_rays, n_angles)
    max_shift:Maximum absolute shift (in rows) to try. Defaults to n_rays//4.

    Returns
    -------
    best_delta: The integer vertical shift that yields lowest symmetry error.
    """
    n_rays, n_angles = sinogram.shape
    if max_shift is None:
        max_shift = n_rays // 4

    half = n_angles // 2
    errors = {}
    for delta in range(-max_shift, max_shift + 1):
        # shift the sinogram up/down
        sino_shifted = shift(sinogram, shift=[delta, 0], order=1, mode="nearest")

        first = sino_shifted[:, :half]
        second = sino_shifted[:, half:]
        second_flipped = np.flip(second, axis=0)  # flip top<->bottom

        # compute mean squared difference
        err = np.mean((first - second_flipped[:, ::-1]) ** 2)
        errors[delta] = err

    # pick the delta with minimum error
    best_delta = min(errors, key=errors.get)
    return best_delta


def auto_center_sinogram(sinogram: np.ndarray, max_shift=None)-> tuple[np.ndarray, int]:
    """
    Auto center the sinogram by shifting it by the best symmetry based offset.

    Returns the centered sinogram and the applied shift.
    """
    delta = find_best_center_shift(sinogram, max_shift=max_shift)
    centered = shift(sinogram, shift=[delta, 0], order=1, mode="nearest")
    return centered, delta


def symmetrize_sinogram(sino360: np.ndarray)-> np.ndarray:
    """
    Takes sino360 of shape (n_rays, 360) and returns
    sino180 of shape (n_rays, 180) after averaging θ with θ+180.
    """
    n_rays, n_angles = sino360.shape
    assert n_angles % 2 == 0, "Need an even number of angles"
    half = n_angles // 2

    # Split into first half [0..half-1] and second half [half..]
    first = sino360[:, :half]
    second = sino360[:, half:]
    # Flip second in the angular axis so that θ+180 lines up with θ
    second_flipped = np.flip(second, axis=1)
    # Average
    sino180 = 0.5 * (first + second_flipped)
    return sino180


def reconstruct_focal_spot(sinogram: np.ndarray, symmetrize: bool = False)-> np.ndarray:
    """
    Reconstruct the focal spot from the sinogram using filtered back-projection.

    Args:
        sinogram: 2D array [angle_index, radial_profile]
        symmetrize: If True, symmetrize the sinogram before reconstruction.
    Returns:
        reconstruction: 2D array representing the focal spot
    """
    if symmetrize:
        sinogram = symmetrize_sinogram(sinogram)
        theta = np.linspace(0.0, 180.0, sinogram.shape[1], endpoint=False)
        reconstruction = iradon(sinogram, theta=theta, filter_name="hann", circle=True)
    else:
        theta = np.linspace(0.0, 360.0, sinogram.shape[1], endpoint=False)
        reconstruction = iradon(sinogram, theta=theta, filter_name="hann", circle=True)
    return reconstruction


def reconstruct_with_axis_shifts(
    sinogram:np.ndarray, output_tiff_path:str, shifts:list, filter_name:str="hann", circle:bool=True
)-> None:
    """
    For each vertical shift in `shifts`, shift the sinogram, reconstruct,
    and save all reconstructions in a single multi‐page TIFF.

    Parameters
    ----------
    sinogram: 2D array [angle_index, radial_profile]
    output_tiff_path: Path to the output multi‐page TIFF.
    shifts: Amounts (in rows) to shift the sinogram: positive shifts move the axis downward.
    filter_name: Filter to use in iradon.
    circle: Pass-through to skimage.transform.iradon.
    """
    reconstructions = []
    # Prepare angles for full 360° sinogram
    n_angles = sinogram.shape[1]
    theta = np.linspace(0.0, 360.0, n_angles, endpoint=False)

    for delta in shifts:
        # shift sinogram vertically:
        #   shifting by +delta moves content down, so the effective axis moves up
        shifted_sino = shift(sinogram, shift=[delta, 0], order=1, mode="nearest")

        # reconstruct
        rec = iradon(shifted_sino, theta=theta, filter_name=filter_name, circle=circle)
        reconstructions.append(rec.astype(np.float32))


    tifffile.imwrite(
        output_tiff_path, np.stack(reconstructions, axis=0), photometric="minisblack"
    )
    print(f"Saved {len(shifts)} reconstructions to '{output_tiff_path}'")


def fwhm(sinogram: np.ndarray) -> tuple[int, int, int]:
    """
    Compute the full width at half maximum (FWHM) of a 1D sinogram profile.

    Args:
        sinogram: 1D array representing a single profile from the sinogram.

    Returns:
        width_px: Width in pixels between half-maximum crossings.
        left_idx: Index of the left half-maximum crossing.
        right_idx: Index of the right half-maximum crossing.
    """
    half_max = (sinogram.max() + sinogram.min()) / 2.0
    n = len(sinogram)
    center = n // 2

    left_edge = None
    for i in range(center, -1, -1):
        if sinogram[i] < half_max:
            left_edge = i
            break
    left = (left_edge + 1) if left_edge is not None else 0

    right_edge = None
    for i in range(center, n):
        if sinogram[i] < half_max:
            right_edge = i
            break
    right = (right_edge - 1) if right_edge is not None else (n - 1)

    width = right - left
    return width, left, right


def erf_step(x: np.ndarray, A: float, x0: float, sigma: float, B: float) -> np.ndarray:
    """
    Error function step model for fitting profile edges.

    Args:
        x: Independent variable (e.g., pixel positions).
        A: Amplitude of the step.
        x0: Center position of the transition.
        sigma: Width of the transition (standard deviation).
        B: Background offset.

    Returns:
        Model values evaluated at x.
    """
    return A * erf((x - x0) / sigma) + B


def find_extreme_profiles_erf(profiles: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    Fit each angular profile to an error function step and compute its slope.

    Args:
        profiles: 2D array of shape [n_rays, n_angles] containing line profiles.

    Returns:
        wide_idx: Index of the profile with the steepest (widest) slope.
        narrow_idx: Index of the profile with the shallowest (narrowest) slope.
        slopes: 1D array of computed slopes for each profile.
    """
    n_rays, n_angles = profiles.shape
    x = np.arange(n_rays)
    slopes = np.zeros(n_angles, dtype=float)

    p0 = [profiles.max() - profiles.min(), n_rays / 2, n_rays / 8, profiles.min()]

    for i in range(n_angles):
        p = profiles[:, i]
        try:
            popt, _ = curve_fit(erf_step, x, p, p0=p0, maxfev=2000)
            A, x0, sigma, B = popt
            slope = A / (np.sqrt(np.pi) * sigma)
        except RuntimeError:
            slope = 0
        slopes[i] = slope

    wide_idx = int(np.argmax(slopes))
    narrow_idx = int(np.argmin(slopes))
    return wide_idx, narrow_idx, slopes


def main():
    img_path = r"C:\Users\jacop\Desktop\PhD\Focal Spot\ATS 6 mag\P146\00013_8ca7056d-d871-493e-ad31-99ce37ec2c6d_20250506_120021.2359.raw"
    out_dir = r".\output"

    pixel_size = 0.154  # mm

    os.makedirs(out_dir, exist_ok=True)

    img = load_raw_as_ndarray(img_path)

    hough_circle = detect_circle_hough(
        img,
        dp=1.3,
        min_dist=100,
        param1=70,
        param2=20,
        min_radius=10,
        max_radius=0,
        debug=False,
    )

    if hough_circle:
        x, y, r = hough_circle
        print(f"Detected circle via Hough transform: Center=({x}, {y}), Radius={r} px")

    cropped = crop_square_roi(img, center=(x, y), radius=r, width_factor=1.5)

    # Compute COM profiles
    com_x, com_y = compute_com_profiles(cropped)
    cx, cy, radius = estimate_circle(cropped, com_x, com_y)

    print(f"Estimated circle: Center=({cx}, {cy}), Radius={radius} px")

    plot_circle_on_crop(cropped, cx, cy, radius, out_dir)

    # Extract profiles and sinogram
    profiles, sinogram = compute_profiles_and_sinogram(cropped, cx, cy, radius)
    best_centered_sino, applied_shift = auto_center_sinogram(sinogram)
    axis_shifts = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

    out_tiff = os.path.join(out_dir, "recon_axis_shifts.tiff")
    reconstruct_with_axis_shifts(
        sinogram, out_tiff, shifts=axis_shifts, filter_name="hann", circle=True
    )

    sinogram = best_centered_sino[applied_shift:-applied_shift, :]

    profiles_path = os.path.join(out_dir, "profiles.png")
    plt.imsave(profiles_path, profiles, cmap="gray", origin="lower")
    print(f"Saved profiles to {profiles_path}")

    # 2) Save the sinogram
    sinogram_path = os.path.join(out_dir, "sinogram.png")
    plt.imsave(sinogram_path, sinogram, cmap="gray", origin="lower")
    print(f"Saved sinogram to {sinogram_path}")

    reconstruction = reconstruct_focal_spot(sinogram, symmetrize=False)
    recon_path = os.path.join(out_dir, "reconstruction.png")
    plt.imsave(recon_path, reconstruction, cmap="gray")
    print(f"Saved reconstruction to {recon_path}")

    # Display results
    plt.figure(figsize=(16, 8))

    # Plot profiles
    plt.subplot(1, 3, 1)
    plt.imshow(profiles, cmap="gray")
    plt.title("Aligned Profiles")
    plt.xlabel("Profile Index")
    plt.ylabel("Angle Index")

    # Plot sinogram
    plt.subplot(1, 3, 2)
    plt.imshow(sinogram, cmap="gray")
    plt.title("Sinogram")
    plt.xlabel("Angle Index")
    plt.ylabel("Radial Offset (px)")

    # Plot reconstructed focal spot
    plt.subplot(1, 3, 3)
    plt.imshow(reconstruction, cmap="gray")
    plt.title("Reconstructed Focal Spot")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    wide_idx, narrow_idx, slopes = find_extreme_profiles_erf(profiles)
    print(f"Widest edge at angle idx {wide_idx} (slope={slopes[wide_idx]:.3f})")
    print(f"Narrowest edge at angle idx {narrow_idx} (slope={slopes[narrow_idx]:.3f})")

    prof_wide_sino = sinogram[:, wide_idx]
    prof_narrow_sino = sinogram[:, narrow_idx]

    fw, lw, rw = fwhm(prof_wide_sino)
    fn, ln, rn = fwhm(prof_narrow_sino)
    print(f"Widest:   FWHM={fw}px (from {lw} to {rw})")
    print(f"Narrowest: FWHM={fn}px (from {ln} to {rn})")

    # 3) Build radial axis and plot:
    n_rays = sinogram.shape[0]
    radial = np.arange(n_rays) - n_rays // 2

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the two sinogram profiles
    ax.plot(radial, prof_wide_sino, label=f"Widest (idx={wide_idx})")
    ax.plot(radial, prof_narrow_sino, label=f"Narrowest (idx={narrow_idx})")

    # Compute half‑max levels
    half_w = (prof_wide_sino.max() + prof_wide_sino.min()) / 2.0
    half_n = (prof_narrow_sino.max() + prof_narrow_sino.min()) / 2.0

    # Draw the half‑max horizontal lines spanning between left/right edges
    ax.hlines(
        half_w,
        radial[lw],
        radial[rw],
        linestyles="--",
        color="red",
        label=f"Widest FWHM = {fw}px",
    )
    ax.hlines(
        half_n,
        radial[ln],
        radial[rn],
        linestyles="--",
        color="red",
        label=f"Narrowest FWHM = {fn}px",
    )

    # Enable grid and minor ticks
    ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.minorticks_on()

    ax.set_xlabel("Radial Offset (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title("Central FWHM on Sinogram Profiles")
    ax.legend(loc="upper right")
    plt.tight_layout()

    # Save figure
    out_path = os.path.join(out_dir, "central_fwhm_sinogram_profiles.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved central FWHM sinogram profiles to {out_path}")

    # ——— Plot sinogram with widest & narrowest profiles traced ———
    fig, ax = plt.subplots(figsize=(8, 6))
    # show the sinogram (rows = radial offsets, cols = angle indices)
    im = ax.imshow(sinogram, cmap="gray", aspect="auto")
    ax.set_title("Sinogram with Widest & Narrowest Profiles")
    ax.set_xlabel("Angle Index")
    ax.set_ylabel("Radial Offset (px)")

    # draw vertical lines at the two angle indices
    ax.axvline(
        wide_idx,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Widest (idx={wide_idx})",
    )
    ax.axvline(
        narrow_idx,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Narrowest (idx={narrow_idx})",
    )

    ax.legend(loc="upper right")
    plt.tight_layout()

    # save it
    traced_path = os.path.join(out_dir, "sinogram_traced_profiles.png")
    plt.savefig(traced_path, dpi=300)
    plt.close(fig)
    print(f"Saved sinogram with traced profiles to {traced_path}")


if __name__ == "__main__":
    main()
