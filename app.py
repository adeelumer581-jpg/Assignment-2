import os
import sys
import json
import logging

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from flask import Flask, render_template, request, send_from_directory
from PIL import Image

from preprocessor import preprocess
from feature_extractor import build_feature_matrix
from persistence import load_artifacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}

# Load model artifacts at startup
models_dir = os.path.join(_HERE, "models")
try:
    artifacts = load_artifacts(models_dir)
    logger.info("Model artifacts loaded successfully.")
except FileNotFoundError as e:
    logger.error("Failed to load model artifacts: %s", e)
    sys.exit(1)

# Load comparison report
report_path = os.path.join(models_dir, "report.json")
try:
    with open(report_path) as f:
        report = json.load(f)
except FileNotFoundError:
    report = None


@app.route("/")
def index():
    return render_template("index.html", report=report)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", report=report,
                               error="Please select a file.")

    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return render_template("index.html", report=report,
                               error="Unsupported format. Upload a .jpg, .jpeg, .png, or .gif file.")

    try:
        image        = Image.open(file.stream)
        preprocessed = preprocess(image)
        X            = build_feature_matrix([preprocessed], method=artifacts["extractor_method"])
        prediction   = artifacts["best"].predict(X)[0]
        label        = "Woman" if prediction == 1 else "Man"
        best_name    = report["best"][0] if report else "Best Model"
        return render_template("result.html", prediction=label,
                               best_model=best_name, report=report)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return render_template("index.html", report=report,
                               error="An error occurred during prediction. Please try again.")


@app.route("/comparison")
def comparison():
    return render_template("comparison.html", report=report)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(_HERE, "static"), filename)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
