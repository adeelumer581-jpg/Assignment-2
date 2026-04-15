"""
train.py — Gender Classification Training Pipeline
"""

import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from data_loader import load_dataset
from preprocessor import preprocess
from feature_extractor import build_feature_matrix
from models import KNNModel, DecisionTreeModel, NaiveBayesModel
from evaluator import compute_accuracy, compute_confusion_matrix, print_results, compare_models
from persistence import save_artifacts

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def save_confusion_matrix_image(cm, model_name, output_path):
    """Save a styled confusion matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Man", "Woman"],
        yticklabels=["Man", "Woman"],
        linewidths=0.5, linecolor="gray",
        ax=ax, annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
    ax.set_ylabel("Actual", fontsize=12, labelpad=8)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_comparison_chart(results, best_names, output_path):
    """Save a bar chart comparing model accuracies."""
    models = list(results.keys())
    accs   = list(results.values())
    colors = ["#4f46e5" if m in best_names else "#93c5fd" for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, accs, color=colors, edgecolor="white", linewidth=1.5, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)

    best_patch = mpatches.Patch(color="#4f46e5", label=f"Best: {', '.join(best_names)}")
    ax.legend(handles=[best_patch], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    train_root = os.path.join(os.path.expanduser("~"), "Downloads", "DATA", "traindata", "traindata")
    test_root  = os.path.join(os.path.expanduser("~"), "Downloads", "DATA", "testdata",  "testdata")

    print("=" * 50)
    print("[1/7] Loading training data...")
    train_images, y_train = load_dataset(train_root)
    print(f"      Total training samples: {len(train_images)}")

    print("\n[2/7] Loading test data...")
    test_images, y_test = load_dataset(test_root)
    print(f"      Total test samples: {len(test_images)}")

    print("\n[3/7] Preprocessing images...")
    preprocessed_train = [preprocess(img) for img in train_images]
    preprocessed_test  = [preprocess(img) for img in test_images]
    print(f"      Done. Train: {len(preprocessed_train)}, Test: {len(preprocessed_test)}")

    print("\n[4/7] Extracting features (method=flat)...")
    X_train = build_feature_matrix(preprocessed_train, method="flat")
    X_test  = build_feature_matrix(preprocessed_test,  method="flat")
    print(f"      X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    print("\n[5/7] Training models...")
    print("      Training KNN (k=5)...")
    knn_model = KNNModel(k=5)
    knn_model.fit(X_train, y_train)

    print("      Training Decision Tree...")
    dt_model = DecisionTreeModel()
    dt_model.fit(X_train, y_train)

    print("      Training Naive Bayes...")
    nb_model = NaiveBayesModel()
    nb_model.fit(X_train, y_train)
    print("      All models trained.")

    print("\n[6/7] Evaluating models on test set...")
    knn_preds    = knn_model.predict(X_test)
    knn_acc      = compute_accuracy(y_test, knn_preds)
    knn_cm       = compute_confusion_matrix(y_test, knn_preds)
    print_results("KNN", knn_acc, knn_cm)

    dt_preds     = dt_model.predict(X_test)
    dt_acc       = compute_accuracy(y_test, dt_preds)
    dt_cm        = compute_confusion_matrix(y_test, dt_preds)
    print_results("Decision Tree", dt_acc, dt_cm)

    nb_preds  = nb_model.predict(X_test)
    nb_acc    = compute_accuracy(y_test, nb_preds)
    nb_cm     = compute_confusion_matrix(y_test, nb_preds)
    print_results("Naive Bayes", nb_acc, nb_cm)

    results    = {"KNN": knn_acc, "DecisionTree": dt_acc, "NaiveBayes": nb_acc}
    comparison = compare_models(results)
    best_names = comparison["best"]

    print("\n--- Model Comparison ---")
    print(f"{'Model':<15} {'Accuracy':>10}")
    print("-" * 27)
    for row in comparison["table"]:
        print(f"{row['model']:<15} {row['accuracy']:>9.2f}%")
    print("-" * 27)
    print(f"Best model(s): {', '.join(best_names)}")

    model_map      = {"KNN": knn_model, "DecisionTree": dt_model, "NaiveBayes": nb_model}
    best_model_obj = model_map[best_names[0]]

    # Save artifacts
    models_dir  = os.path.join(_HERE, "models")
    static_dir  = os.path.join(_HERE, "static")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    print(f"\n[7/7] Saving artifacts and charts...")

    # Confusion matrix images
    cm_data = {
        "KNN":          (knn_cm, "cm_knn.png"),
        "DecisionTree": (dt_cm,  "cm_dt.png"),
        "NaiveBayes":   (nb_cm,  "cm_nb.png"),
    }
    for name, (cm, fname) in cm_data.items():
        save_confusion_matrix_image(cm, name, os.path.join(static_dir, fname))

    # Comparison bar chart
    save_comparison_chart(results, best_names, os.path.join(static_dir, "comparison_chart.png"))

    # Save JSON report for Flask to read
    report = {
        "models": [
            {
                "name": "KNN",
                "full_name": "K-Nearest Neighbors",
                "accuracy": knn_acc,
                "cm": knn_cm.tolist(),
                "cm_image": "cm_knn.png",
                "is_best": "KNN" in best_names,
                "description": "Classifies by majority vote among the 5 nearest training samples."
            },
            {
                "name": "DecisionTree",
                "full_name": "Decision Tree",
                "accuracy": dt_acc,
                "cm": dt_cm.tolist(),
                "cm_image": "cm_dt.png",
                "is_best": "DecisionTree" in best_names,
                "description": "Splits data using feature thresholds to build an interpretable tree."
            },
            {
                "name": "NaiveBayes",
                "full_name": "Naive Bayes",
                "accuracy": nb_acc,
                "cm": nb_cm.tolist(),
                "cm_image": "cm_nb.png",
                "is_best": "NaiveBayes" in best_names,
                "description": "Applies Bayes' theorem with Gaussian likelihood, assuming feature independence."
            },
        ],
        "best": best_names,
        "train_samples": len(train_images),
        "test_samples": len(test_images),
    }
    with open(os.path.join(models_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    save_artifacts(
        {"knn": knn_model, "dt": dt_model, "nb": nb_model, "best": best_model_obj},
        "flat",
        models_dir,
    )
    print("      All artifacts saved successfully.")
    print("=" * 50)
    print("Training pipeline complete.")


if __name__ == "__main__":
    main()
