import numpy as np
from PIL import Image


def preprocess(image: Image.Image) -> np.ndarray:
    """
    1. Convert to RGB (handles RGBA, grayscale, palette modes).
    2. Resize to (64, 64) using LANCZOS resampling.
    3. Convert to float32 and divide by 255.0.
    Returns: np.ndarray of shape (64, 64, 3), dtype float32, values in [0, 1].
    """
    image = image.convert("RGB")
    image = image.resize((64, 64), Image.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    return arr
