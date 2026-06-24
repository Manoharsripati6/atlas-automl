"""
Flask frontend for the existing data science pipeline.

The analytical logic lives in main.py and tools/. This file only handles
uploads, calls run_pipeline(file_path), and renders the returned artifacts.
"""
import os
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from main import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
ALLOWED_EXTENSIONS = {".csv"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

latest_run = {
    "filename": None,
    "file_path": None,
    "rows": None,
    "columns": None,
    "result": None,
    "error": None,
}


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def current_result():
    return latest_run.get("result")


def output_url(path):
    if not path:
        return None

    path = Path(path)
    if path.is_absolute():
        try:
            rel_path = path.relative_to(OUTPUT_DIR)
        except ValueError:
            return None
    else:
        parts = path.parts
        rel_path = Path(*parts[1:]) if parts and parts[0] == "outputs" else path

    return url_for("outputs", filename=rel_path.as_posix())


@app.context_processor
def inject_pipeline_state():
    return {
        "latest_run": latest_run,
        "has_results": current_result() is not None,
        "output_url": output_url,
    }


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ---------- Public ----------
@app.route("/")
def landing():
    return render_template("landing.html")


# ---------- Auth ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect(url_for("dashboard"))
    return render_template("auth/login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        return redirect(url_for("dashboard"))
    return render_template("auth/signup.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    return render_template("auth/forgot.html")


@app.route("/logout", methods=["POST", "GET"])
def logout():
    return redirect(url_for("landing"))


# ---------- App ----------
@app.route("/dashboard")
def dashboard():
    result = current_result()
    return render_template("app/dashboard.html", active="overview", result=result)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            return jsonify({"status": "error", "message": "Choose a CSV file."}), 400

        if not allowed_file(uploaded.filename):
            return jsonify({"status": "error", "message": "Only CSV files are supported by the current pipeline."}), 400

        filename = secure_filename(uploaded.filename)
        file_path = DATASET_DIR / filename
        uploaded.save(file_path)

        try:
            preview = pd.read_csv(file_path)
            latest_run.update({
                "filename": filename,
                "file_path": str(file_path),
                "rows": int(preview.shape[0]),
                "columns": int(preview.shape[1]),
                "result": None,
                "error": None,
            })
        except Exception as exc:
            latest_run.update({"error": str(exc), "result": None})
            return jsonify({"status": "error", "message": f"Unable to read CSV: {exc}"}), 400

        try:
            result = run_pipeline(str(file_path))
            latest_run.update({"result": result, "error": None})
        except Exception as exc:
            latest_run.update({"error": str(exc), "result": None})
            return jsonify({
                "status": "error",
                "message": f"Pipeline failed: {exc}",
                "rows": latest_run["rows"],
                "columns": latest_run["columns"],
            }), 500

        return jsonify({
            "status": "ok",
            "filename": filename,
            "rows": latest_run["rows"],
            "columns": latest_run["columns"],
            "redirect": url_for("analyze"),
        })

    return render_template("app/upload.html", active="datasets")


@app.route("/analyze")
def analyze():
    return render_template("app/pipeline.html", active="analysis", result=current_result())


@app.route("/eda")
def eda():
    result = current_result()
    return render_template("app/eda.html", active="analysis", eda=(result or {}).get("eda_report"))


@app.route("/insights")
def insights():
    result = current_result()
    return render_template("app/insights.html", active="analysis", insights=(result or {}).get("insights", []))


@app.route("/visualizations")
def visualizations():
    result = current_result() or {}
    return render_template(
        "app/visualizations.html",
        active="visualizations",
        plots=result.get("plots", []),
        ml_plots=result.get("ml_plots", []),
    )


@app.route("/ml-results")
def ml_results():
    result = current_result()
    return render_template("app/ml.html", active="ml", ml=(result or {}).get("ml_summary"))


@app.route("/feature-importance")
def feature_importance():
    result = current_result() or {}
    ml_plots = result.get("ml_plots", [])
    feature_plot = next((plot for plot in ml_plots if "feature_importance" in str(plot)), None)
    return render_template("app/features.html", active="ml", feature_plot=feature_plot)


@app.route("/report")
def report():
    result = current_result()
    return render_template("app/report.html", active="reports", report_text=(result or {}).get("report"))


@app.route("/profile", methods=["GET", "PUT"])
def profile():
    return render_template("app/settings.html", active="settings", tab="profile")


@app.route("/billing")
def billing():
    return render_template("app/billing.html", active="billing")


@app.route("/api-keys")
def api_keys():
    return render_template("app/api_keys.html", active="api")


@app.route("/team")
def team():
    return render_template("app/team.html", active="team")


@app.route("/admin")
def admin():
    return render_template("app/admin.html", active="admin")


@app.route("/settings")
def settings():
    return render_template("app/settings.html", active="settings", tab="profile")


if __name__ == "__main__":
    app.run(debug=True)
