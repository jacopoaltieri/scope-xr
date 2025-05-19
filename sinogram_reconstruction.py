import os
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
from skimage.transform import iradon, iradon_sart
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import xml.etree.ElementTree as ET


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
    # Construct path to XML file
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
    img_path,
    dp,
    min_dist,
    param1,
    param2,
    min_radius,
    max_radius,
    debug=False,
):
    img = load_raw_as_ndarray(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not open '{img_path}'")
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
        print(f"No circles found in '{os.path.basename(img_path)}'")
        return None

    # Round and pick the strongest circle (first one)
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    if debug:
        output = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

        display_scale = 0.5  # or another value like 0.3
        output_resized = cv2.resize(output, (0, 0), fx=display_scale, fy=display_scale)

        cv2.imshow("Detected Circle", output_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x, y, r


def crop_rect(img, center, radius, width_factor=1.5):
    """
    Crop a square region around the circle center with variable width.
    """
    cx, cy = center
    half_w = int(radius * width_factor)

    x0 = max(cx - half_w, 0)
    x1 = min(cx + half_w, img.shape[1])
    y0 = max(cy - half_w, 0)
    y1 = min(cy + half_w, img.shape[0])

    cropped = img[y0:y1, x0:x1]
    return cropped


def compute_com_profiles(cropped):
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


def estimate_circle(cropped, com_x, com_y):
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


def plot_circle_on_crop(cropped, cx, cy, max_radius,output_path):
    """
    Plot the cropped image with the circle and its center overlay.
    """
    fig, ax = plt.subplots()
    ax.imshow(cropped, cmap="gray")

    # Draw circle
    ax.add_patch(Circle((cx, cy), max_radius, edgecolor="red", fill=False, linewidth=2))

    # Draw center
    ax.plot(cx, cy, "ro", markersize=5)  # red dot at center

    ax.set_title("Estimated Radius with Center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path + "/circle_on_crop.png", dpi=300)
    plt.show()



def trace_profiles(img, cx, cy, radius, n_angles=360, profile_half_length=64):
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

    # Compute the derivative to obtain the LSF
    sinogram = np.gradient(profiles, axis=1)
    print(sinogram)
    return profiles.T, sinogram.T


def reconstruct_focal_spot(sinogram):
    """
    Reconstruct the focal spot from the sinogram using filtered back-projection.

    Args:
        sinogram: 2D array [angle_index, radial_profile]

    Returns:
        reconstruction: 2D array representing the focal spot
    """
    theta = np.linspace(0.0, 360.0, sinogram.shape[1], endpoint=False)
    reconstruction = iradon(sinogram, theta=theta, filter_name="hann", circle=True)
    # reconstruction = iradon_sart(sinogram, theta=theta)
    return reconstruction


def fwhm(profile):
    """Compute Full Width at Half Maximum of a 1D profile."""
    half_max = np.max(profile) / 2.0
    indices = np.where(profile >= half_max)[0]
    # if len(indices) < 2:
    #     return 0  # Too narrow or flat to measure
    return indices[-1] - indices[0]


def find_widest_and_narrowest_profiles(sinogram):
    """
    Now treats each COLUMN as one angular profile.
    sinogram: 2D array of shape (radial_samples, n_angles)
    """
    n_angles = sinogram.shape[1]
    widths = np.zeros(n_angles, dtype=float)
    for i in range(n_angles):
        profile = sinogram[:, i]
        widths[i] = fwhm(profile)
    print(f"Widths: {widths}")
    wide_idx  = np.argmax(widths)
    narrow_idx = np.argmin(widths)
    
    plt.figure()
    plt.plot(widths)
    plt.xlabel('Column index')
    plt.ylabel('Profile width (pixels)')
    plt.title('Profile width vs. Column')
    plt.scatter([narrow_idx, wide_idx], [widths[narrow_idx], widths[wide_idx]])
    plt.show()
    return wide_idx, narrow_idx, widths[wide_idx], widths[narrow_idx]


def main():
    img_path = r"C:\Users\jacop\Desktop\PhD\Focal Spot\ATS 6 mag\P146\00013_8ca7056d-d871-493e-ad31-99ce37ec2c6d_20250506_120021.2359.raw"
    out_dir = r".\output"
    os.makedirs(out_dir, exist_ok=True)
    
    
    img = load_raw_as_ndarray(img_path)
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hough_circle = detect_circle_hough(
        img_path,
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

        cropped = crop_rect(img_8bit, center=(x, y), radius=r, width_factor=1.5)

        # Compute COM profiles
        com_x, com_y = compute_com_profiles(cropped)
        cx, cy, radius = estimate_circle(cropped, com_x, com_y)

        print(f"Estimated circle: Center=({cx}, {cy}), Radius={radius} px")

        plot_circle_on_crop(cropped, cx, cy, radius,out_dir)

        # Extract profiles and sinogram
        profiles, sinogram = trace_profiles(cropped, cx, cy, radius)
        profiles_path = os.path.join(out_dir, "profiles.png")
        plt.imsave(profiles_path, profiles, cmap="gray", origin="lower")
        print(f"Saved profiles to {profiles_path}")

        # 2) Save the sinogram
        sinogram_path = os.path.join(out_dir, "sinogram.png")
        plt.imsave(sinogram_path, sinogram, cmap="gray", origin="lower")
        print(f"Saved sinogram to {sinogram_path}")

        reconstruction = reconstruct_focal_spot(sinogram)
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
        plt.imshow(sinogram, cmap="gray" )
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

    widest_idx, narrowest_idx, widest_width, narrowest_width = (
        find_widest_and_narrowest_profiles(sinogram)
    )
    

    # after computing sinogram and (widest_idx, narrowest_idx, ...)
    half_len = sinogram.shape[0] // 2
    radial_axis = np.arange(-half_len, half_len)

    widest_profile   = -sinogram[:, widest_idx]
    narrowest_profile = -sinogram[:, narrowest_idx]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Widest
    axs[0].plot(radial_axis, widest_profile)
    axs[0].set_title(f"Widest Profile\nAngle #{widest_idx} (Width = {widest_width})")
    axs[0].set_xlabel("Radial Offset (pixels)")
    axs[0].set_ylabel("Intensity")

    # Narrowest
    axs[1].plot(radial_axis, narrowest_profile)
    axs[1].set_title(f"Narrowest Profile\nAngle #{narrowest_idx} (Width = {narrowest_width})")
    axs[1].set_xlabel("Radial Offset (pixels)")
    axs[1].set_ylabel("Intensity")

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
