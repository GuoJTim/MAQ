from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, send_file, send_from_directory


WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent
EVALUATION_DIR = REPO_ROOT / "evaluation_results"
DEFAULT_CSV = "agent_performance_door-human-v1.csv"


def create_app(default_csv: str = DEFAULT_CSV) -> Flask:
    app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
    app.config["DEFAULT_CSV"] = default_csv

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/web")
    @app.get("/web/")
    def web_index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/web/<path:filename>")
    def web_static(filename: str):
        return send_from_directory(WEB_DIR, filename)

    @app.get("/dashboard")
    def dashboard_alias():
        return redirect("/")

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/api/evaluation-results")
    def list_evaluation_results():
        if not EVALUATION_DIR.exists():
            return jsonify({"files": []})

        files = []
        for path in sorted(EVALUATION_DIR.glob("*.csv")):
            files.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "is_default": path.name == app.config["DEFAULT_CSV"],
                }
            )
        return jsonify({"files": files})

    @app.get("/api/evaluation-results/default")
    def default_evaluation_result():
        return send_evaluation_csv(app.config["DEFAULT_CSV"])

    @app.get("/api/evaluation-results/<path:filename>")
    def named_evaluation_result(filename: str):
        return send_evaluation_csv(filename)

    return app


def send_evaluation_csv(filename: str):
    if not filename.endswith(".csv"):
        abort(404)

    csv_path = (EVALUATION_DIR / filename).resolve()
    try:
        csv_path.relative_to(EVALUATION_DIR.resolve())
    except ValueError:
        abort(404)

    if not csv_path.is_file():
        abort(404)

    return send_file(csv_path, mimetype="text/csv; charset=utf-8", as_attachment=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the MAQ experiment dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind.")
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Default CSV filename inside evaluation_results/.",
    )
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(default_csv=args.csv)
    app.run(host=args.host, port=args.port, debug=args.debug)
