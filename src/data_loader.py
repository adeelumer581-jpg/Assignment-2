import os
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}


def get_image_files(folder: str) -> list:
    """
    Returns all file paths in folder whose extension is in
    {.jpg, .jpeg, .png, .gif} (case-insensitive).
    Does NOT recurse into subdirectories.
    """
    result = []
    try:
        entries = os.listdir(folder)
    except OSError:
        return result

    for entry in entries:
        full_path = os.path.join(folder, entry)
        if os.path.isfile(full_path):
            ext = os.path.splitext(entry)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                result.append(full_path)

    return result


def load_dataset(root_dir: str):
    """
    Scans root_dir/men/ (label 0) and root_dir/women/ (label 1).
    Opens each file with Pillow. Skips undecodable files silently.
    Returns (images_list, y_array) where images_list is a list of
    PIL Image objects and y_array is a numpy int array.
    Prints per-class counts after loading.
    """
    class_dirs = [
        (os.path.join(root_dir, "men"), 0),
        (os.path.join(root_dir, "women"), 1),
    ]

    images = []
    labels = []

    class_counts = {0: 0, 1: 0}

    for folder, label in class_dirs:
        file_paths = get_image_files(folder)
        for path in file_paths:
            try:
                img = Image.open(path)
                img.load()  # force decode to catch corrupt files
                images.append(img)
                labels.append(label)
                class_counts[label] += 1
            except Exception:
                # silently skip undecodable files
                pass

    print(f"Loaded {class_counts[0]} men images (label 0)")
    print(f"Loaded {class_counts[1]} women images (label 1)")

    y_array = np.array(labels, dtype=int)
    return images, y_array
