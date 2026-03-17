import numpy as np
from skimage import color


def _match_mean_std(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src_mean = src.reshape(-1, 3).mean(axis=0)
    src_std = src.reshape(-1, 3).std(axis=0) + 1e-8

    ref_mean = ref.reshape(-1, 3).mean(axis=0)
    ref_std = ref.reshape(-1, 3).std(axis=0) + 1e-8

    out = (src - src_mean) / src_std
    out = out * ref_std + ref_mean
    return out


def mean_std_color_transfer(source_rgb: np.ndarray, reference_rgb: np.ndarray, colorspace: str = "lab") -> np.ndarray:
    """
    source_rgb, reference_rgb: float arrays in [0, 1], shape (H, W, 3)
    """
    if colorspace.lower() == "lab":
        src = color.rgb2lab(source_rgb)
        ref = color.rgb2lab(reference_rgb)
        out = _match_mean_std(src, ref)
        out = color.lab2rgb(out)
    elif colorspace.lower() == "rgb":
        out = _match_mean_std(source_rgb, reference_rgb)
    else:
        raise ValueError(f"Unsupported colorspace: {colorspace}")

    return np.clip(out, 0.0, 1.0)