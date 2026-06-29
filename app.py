"""
Flask frontend for the existing data science pipeline.

The analytical logic lives in main.py and tools/. This file handles auth,
uploads, calls run_pipeline(file_path), and renders returned artifacts.
"""
import os
import sqlite3
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from main import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = BASE_DIR / "atlas.db"
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


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            company TEXT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            plan TEXT NOT NULL DEFAULT 'free',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS login_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            email TEXT NOT NULL,
            success INTEGER NOT NULL,
            ip_address TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )

    admin_email = os.environ.get("ATLAS_ADMIN_EMAIL", "admin@atlas.local").lower()
    admin_password = os.environ.get("ATLAS_ADMIN_PASSWORD", "admin12345")
    existing = db.execute(
        "SELECT id FROM users WHERE email = ?",
        (admin_email,),
    ).fetchone()

    if existing is None:
        db.execute(
            """
            INSERT INTO users (name, company, email, password_hash, role, plan)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "Atlas Admin",
                "Atlas",
                admin_email,
                generate_password_hash(admin_password),
                "admin",
                "premium",
            ),
        )
        db.commit()

    db.close()


def log_login(email, success, user_id=None):
    get_db().execute(
        """
        INSERT INTO login_events (user_id, email, success, ip_address)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, email.lower(), int(success), request.remote_addr),
    )
    get_db().commit()


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_db().execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            return redirect(url_for("login", next=request.path))
        if user["role"] != "admin":
            flash("Admin access is required.", "error")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)

    return wrapped


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def current_result():
    return latest_run.get("result")


def display_result_key(key, fallback_key=None, default=None):
    result = current_result() or {}
    if key in result:
        return result[key]
    if fallback_key and fallback_key in result:
        return result[fallback_key]
    return default


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
    user = current_user()
    return {
        "latest_run": latest_run,
        "has_results": current_result() is not None,
        "output_url": output_url,
        "current_user": user,
        "is_admin": bool(user and user["role"] == "admin"),
    }


@app.route("/outputs/<path:filename>")
@login_required
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
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            log_login(email, True, user["id"])
            return redirect(request.args.get("next") or url_for("dashboard"))

        log_login(email or "unknown", False)
        flash("Invalid email or password.", "error")

    return render_template("auth/login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        company = request.form.get("company", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("auth/signup.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("auth/signup.html")

        try:
            db = get_db()
            cursor = db.execute(
                """
                INSERT INTO users (name, company, email, password_hash, role, plan)
                VALUES (?, ?, ?, ?, 'user', 'free')
                """,
                (name, company, email, generate_password_hash(password)),
            )
            db.commit()
        except sqlite3.IntegrityError:
            flash("An account already exists for that email.", "error")
            return render_template("auth/signup.html")

        session.clear()
        session["user_id"] = cursor.lastrowid
        return redirect(url_for("dashboard"))

    return render_template("auth/signup.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    return render_template("auth/forgot.html")


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    return redirect(url_for("landing"))


# ---------- App ----------
@app.route("/dashboard")
@login_required
def dashboard():
    result = current_result()
    return render_template("app/dashboard.html", active="overview", result=result)


@app.route("/upload", methods=["GET", "POST"])
@login_required
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
@login_required
def analyze():
    return render_template("app/pipeline.html", active="analysis", result=current_result())


@app.route("/eda")
@login_required
def eda():
    return render_template(
        "app/eda.html",
        active="analysis",
        eda=display_result_key("display_eda_report", "eda_report"),
    )


@app.route("/insights")
@login_required
def insights():
    return render_template(
        "app/insights.html",
        active="analysis",
        insights=display_result_key("display_insights", "insights", []),
    )


@app.route("/visualizations")
@login_required
def visualizations():
    result = current_result() or {}
    return render_template(
        "app/visualizations.html",
        active="visualizations",
        plots=result.get("plots", []),
        ml_plots=result.get("ml_plots", []),
    )


@app.route("/ml-results")
@login_required
def ml_results():
    return render_template(
        "app/ml.html",
        active="ml",
        ml=display_result_key("display_ml_summary", "ml_summary"),
    )


@app.route("/feature-importance")
@login_required
def feature_importance():
    result = current_result() or {}
    ml_plots = result.get("ml_plots", [])
    feature_plot = next((plot for plot in ml_plots if "feature_importance" in str(plot)), None)
    return render_template("app/features.html", active="ml", feature_plot=feature_plot)


@app.route("/report")
@login_required
def report():
    result = current_result()
    return render_template("app/report.html", active="reports", report_text=(result or {}).get("report"))


@app.route("/profile", methods=["GET", "PUT"])
@login_required
def profile():
    return render_template("app/settings.html", active="settings", tab="profile")


@app.route("/billing")
@login_required
def billing():
    return render_template("app/billing.html", active="billing")


@app.route("/api-keys")
@login_required
def api_keys():
    return render_template("app/api_keys.html", active="api")


@app.route("/team")
@login_required
def team():
    return render_template("app/team.html", active="team")


@app.route("/admin")
@admin_required
def admin():
    users = get_db().execute(
        "SELECT id, name, company, email, role, plan, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    login_events = get_db().execute(
        """
        SELECT email, success, ip_address, created_at
        FROM login_events
        ORDER BY created_at DESC
        LIMIT 20
        """
    ).fetchall()
    return render_template(
        "app/admin.html",
        active="admin",
        users=users,
        login_events=login_events,
    )


@app.route("/settings")
@login_required
def settings():
    return render_template("app/settings.html", active="settings", tab="profile")


init_db()


if __name__ == "__main__":
    app.run(debug=True)
