"""
generate_charts.py — Regenerates confusion matrix images and comparison chart
from the saved report.json without retraining.
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

models_dir = os.path.join(_HERE, "models")
static_dir = os.path.join(_HERE, "static")
os.makedirs(static_dir, exist_ok=True)

with open(os.path.join(models_dir, "report.json")) as f:
    report = json.load(f)


def save_cm(cm_list, model_name, output_path):
    cm = np.array(cm_list)
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
    print(f"  ✅ Saved {output_path}")


def save_chart(report, output_path):
    models     = [m["full_name"] for m in report["models"]]  # Use full names for display
    accs       = [m["accuracy"] for m in report["models"]]
    best_names = report["best"]
    colors     = ["#4f46e5" if m in best_names else "#93c5fd" for m in [m["name"] for m in report["models"]]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accs, color=colors, edgecolor="white", linewidth=2, width=0.6)
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    best_patch = mpatches.Patch(color="#4f46e5", label=f"Best: {', '.join(best_names)}")
    ax.legend(handles=[best_patch], fontsize=11, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved {output_path}")


print("🔄 Generating charts from report.json...")
print("\n📊 Confusion Matrices:")
for m in report["models"]:
    save_cm(m["cm"], m["full_name"], os.path.join(static_dir, m["cm_image"]))

print("\n📈 Comparison Chart:")
save_chart(report, os.path.join(static_dir, "comparison_chart.png"))
print("\n✅ Done! Charts regenerated successfully.")
