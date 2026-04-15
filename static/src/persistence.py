import os
import json
import joblib

_MODEL_FILES = {
    "knn": "knn_model.joblib",
    "dt": "dt_model.joblib",
    "nb": "nb_model.joblib",
    "best": "best_model.joblib",
}
_CONFIG_FILE = "extractor_config.json"


def save_artifacts(models: dict, extractor_method: str, output_dir: str) -> None:
    """Serializes each model with joblib and writes extractor_config.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for key, filename in _MODEL_FILES.items():
        joblib.dump(models[key], os.path.join(output_dir, filename))

    config_path = os.path.join(output_dir, _CONFIG_FILE)
    with open(config_path, "w") as f:
        json.dump({"method": extractor_method}, f)


def load_artifacts(model_dir: str) -> dict:
    """Loads all model artifacts and extractor config from model_dir.

    Raises FileNotFoundError with a descriptive message if any required file is missing.
    Returns dict with keys: "knn", "dt", "kmeans", "best", "extractor_method".
    """
    result = {}

    for key, filename in _MODEL_FILES.items():
        path = os.path.join(model_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required model artifact not found: '{path}'. "
                f"Run the training pipeline first to generate model files."
            )
        result[key] = joblib.load(path)

    config_path = os.path.join(model_dir, _CONFIG_FILE)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Required config file not found: '{config_path}'. "
            f"Run the training pipeline first to generate model files."
        )
    with open(config_path, "r") as f:
        config = json.load(f)
    result["extractor_method"] = config["method"]

    return result
