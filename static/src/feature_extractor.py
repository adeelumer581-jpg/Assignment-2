import numpy as np
from skimage.feature import hog


def extract_flat(image_array: np.ndarray) -> np.ndarray:
    """
    Flattens (64, 64, 3) array to shape (12288,).
    """
    return image_array.flatten()


def extract_hog(image_array: np.ndarray) -> np.ndarray:
    """
    Computes HOG descriptor using skimage.feature.hog with:
      orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
      channel_axis=-1.
    Returns a fixed-length 1D float array (~1568 for 64x64 input).
    """
    return hog(
        image_array,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1,
    )


def build_feature_matrix(images: list, method: str = "flat") -> np.ndarray:
    """
    Applies the chosen extraction method to all images.
    method: "flat" | "hog"
    Returns: np.ndarray of shape (N, feature_length).
    """
    if method == "hog":
        extractor = extract_hog
    else:
        extractor = extract_flat

    features = [extractor(img) for img in images]
    return np.array(features)
