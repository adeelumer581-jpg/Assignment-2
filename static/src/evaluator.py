import numpy as np
from sklearn.metrics import confusion_matrix


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns accuracy as a float percentage rounded to 2 decimal places.

    Raises ValueError if either array is empty.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Cannot compute accuracy on empty arrays.")

    accuracy = np.mean(y_true == y_pred) * 100.0
    return round(float(accuracy), 2)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns a 2×2 confusion matrix as np.ndarray.

    Uses labels=[0, 1] to ensure consistent shape even if one class is absent.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compare_models(results: dict) -> dict:
    """Compares model accuracies and returns a summary dict.

    Args:
        results: {"KNN": 72.50, "DecisionTree": 68.30, "KMeans": 55.10}

    Returns:
        {
            "table": [{"model": name, "accuracy": acc}, ...],
            "best": [model_name, ...]  # all models tied at max accuracy
        }
    """
    table = [{"model": name, "accuracy": acc} for name, acc in results.items()]
    max_acc = max(results.values())
    best = [name for name, acc in results.items() if acc == max_acc]
    return {"table": table, "best": best}


def print_results(model_name: str, accuracy: float, cm: np.ndarray) -> None:
    """Prints a formatted summary of model evaluation results."""
    print(f"\n{'=' * 40}")
    print(f"  Model: {model_name}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Man   Woman")
    print(f"  Actual Man  {cm[0, 0]:5d} {cm[0, 1]:5d}")
    print(f"  Actual Woman{cm[1, 0]:5d} {cm[1, 1]:5d}")
    print(f"{'=' * 40}\n")
