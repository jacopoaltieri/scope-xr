
import os
import numpy as np
import matplotlib.pyplot as plt


def eval_minimum_magnification(a: int, n: int, p: int) -> int:
    """Evaluate the minimum magnification required to obtain a focal spot image involving a reasonable number n of pixels."""
    m = (a + n * p) / a
    return m


def eval_minimum_radius(n: int, p: int, m: int) -> int:
    """Evaluate the minimum disk radius required to obtain a focal spot image involving a reasonable number n of pixels."""
    r = (1 + n**2) * p / (2 * m)
    return r


def crop_square_roi(
    img: np.ndarray,
    center: tuple[float, float],
    radius: float,
    width_factor: float = 1.5,
    output_path: str = None,
) -> np.ndarray:

    cx, cy = center
    half_w = int(radius * width_factor)

    x0 = max(cx - half_w, 0)
    x1 = min(cx + half_w, img.shape[1])
    y0 = max(cy - half_w, 0)
    y1 = min(cy + half_w, img.shape[0])

    cropped = img[y0:y1, x0:x1]
    if output_path is not None:
        plt.imsave(
            os.path.join(output_path, "cropped.png"),
            cropped.astype(np.uint16),
            cmap="gray"
        )
        print(fr"Saved cropped image to '{output_path}\cropped.png'")
    return cropped