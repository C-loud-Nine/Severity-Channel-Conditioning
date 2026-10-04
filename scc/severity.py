"""Severity map: s = clip[(1 - g(x)/max g) * gamma(m_isp), 0, 1].

g(x): Gaussian-smoothed (15x15, sigma = OpenCV default 2.6) Sobel (3x3) gradient magnitude of the
grayscale blurred image. gamma(m_isp) = 1.5 - m~, with m~ = isp_diff min-max normalized over all
released metadata, range [-3830.9, -33.6]. isp_diff = high-frequency energy of the blurred frame
minus that of the sharp frame (negative when blurring removes high-frequency energy).
"""
import cv2
import numpy as np

ISP_MIN = -3830.879   # min of isp_diff over all 7,400 released samples
ISP_MAX = -33.649     # max of isp_diff over all 7,400 released samples
EPS = 1e-6


def gradient_blur_map(img_rgb: np.ndarray) -> np.ndarray:
    """High in low-gradient regions (blurred or naturally smooth)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx ** 2 + gy ** 2), (15, 15), 0)
    return 1.0 - grad / (grad.max() + EPS)


def isp_gate(isp_diff: float) -> float:
    """gamma(m_isp) = 1.5 - normalized statistic, in [0.5, 1.5]."""
    norm = (isp_diff - ISP_MIN) / (ISP_MAX - ISP_MIN + EPS)
    return float(1.5 - norm)


def build_severity(img_rgb: np.ndarray, isp_diff: float, out_size: int = 256) -> np.ndarray:
    """Computed at full resolution, stored at out_size x out_size (area), upsampled bilinearly before use."""
    sev = gradient_blur_map(img_rgb) * isp_gate(isp_diff)
    sev = cv2.resize(sev, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return np.clip(sev, 0, 1).astype(np.float32)


def build_variant(img_rgb, isp_diff, variant="severity", out_size=256):
    """Fourth-channel maps for the descriptor controls. severity: full map; gradient_only: gamma = 1;
    scale_only: gradient term = 1 (spatially uniform clip(gamma))."""
    if variant == "severity":
        return build_severity(img_rgb, isp_diff, out_size)
    if variant == "gradient_only":
        sev = gradient_blur_map(img_rgb)
    elif variant == "scale_only":
        sev = np.full(img_rgb.shape[:2], isp_gate(isp_diff), dtype=np.float32)
    else:
        raise ValueError(variant)
    sev = cv2.resize(sev, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return np.clip(sev, 0, 1).astype(np.float32)
