"""
app.py
------
Flask entry point for the NIDS prediction API.

Routes
------
GET  /health          – Liveness probe
GET  /ui              – Browser UI (served from templates/index.html)
POST /predict         – Accept a TrafficRecord JSON body, return a prediction
GET  /schema          – Return the expected JSON input schema

Run (development)
-----------------
    python app.py

Run (production)
----------------
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
"""

from __future__ import annotations

import csv
import io
import logging
import os
from typing import Any

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from model_client import ModelClient
from schema import TrafficRecord

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialise the Watson ML client once at startup.
# The app will fail fast with a clear error if env vars are not set.
_model_client: ModelClient = ModelClient()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# The deployed model expects a full KDD-style feature vector, so we materialise
# the complete feature list and fill any missing values with safe defaults.
_FEATURE_ORDER = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Categorical → integer encoding (must match the encoding used during training)
_PROTOCOL_ENCODING = {"tcp": 0, "udp": 1, "icmp": 2}
_SERVICE_ENCODING = {
    "http": 0,
    "ftp": 1,
    "smtp": 2,
    "ssh": 3,
    "dns": 4,
    "telnet": 5,
    "https": 6,
    "pop3": 7,
    "ftp_data": 8,
    "other": 9,
}
_FLAG_ENCODING = {
    "sf": 0,
    "s0": 1,
    "rej": 2,
    "rsto": 3,
    "sh": 4,
    "s1": 5,
    "s2": 6,
    "s3": 7,
    "oth": 8,
}


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_category(value: Any, encoding: dict[str, int], default: str) -> int:
    if value is None:
        return encoding[default]
    text = str(value).strip().lower()
    return encoding.get(text, encoding.get(default, 0))


def _pick(payload: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in payload and payload[name] not in (None, ""):
            return payload[name]
    return default


def _build_feature_vector(payload: dict[str, Any]) -> list[Any]:
    normalized = {str(key).strip().lower(): value for key, value in payload.items()}

    protocol_value = _pick(normalized, "protocol_type", "protocol", default="tcp")
    service_value = _pick(normalized, "service", default="other")
    flag_value = _pick(normalized, "flag", default="SF")

    return [
        _coerce_int(_pick(normalized, "duration", default=0), 0),
        _coerce_category(protocol_value, _PROTOCOL_ENCODING, "tcp"),
        _coerce_category(service_value, _SERVICE_ENCODING, "other"),
        _coerce_category(flag_value, _FLAG_ENCODING, "sf"),
        _coerce_int(_pick(normalized, "src_bytes", default=0), 0),
        _coerce_int(_pick(normalized, "dst_bytes", default=0), 0),
        _coerce_int(_pick(normalized, "land", default=0), 0),
        _coerce_int(_pick(normalized, "wrong_fragment", default=0), 0),
        _coerce_int(_pick(normalized, "urgent", default=0), 0),
        _coerce_int(_pick(normalized, "hot", default=0), 0),
        _coerce_int(_pick(normalized, "num_failed_logins", default=0), 0),
        _coerce_int(_pick(normalized, "logged_in", default=0), 0),
        _coerce_int(_pick(normalized, "num_compromised", default=0), 0),
        _coerce_int(_pick(normalized, "root_shell", default=0), 0),
        _coerce_int(_pick(normalized, "su_attempted", default=0), 0),
        _coerce_int(_pick(normalized, "num_root", default=0), 0),
        _coerce_int(_pick(normalized, "num_file_creations", default=0), 0),
        _coerce_int(_pick(normalized, "num_shells", default=0), 0),
        _coerce_int(_pick(normalized, "num_access_files", default=0), 0),
        _coerce_int(_pick(normalized, "num_outbound_cmds", default=0), 0),
        _coerce_int(_pick(normalized, "is_host_login", default=0), 0),
        _coerce_int(_pick(normalized, "is_guest_login", default=0), 0),
        _coerce_int(_pick(normalized, "count", default=0), 0),
        _coerce_int(_pick(normalized, "srv_count", default=0), 0),
        _coerce_float(_pick(normalized, "serror_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "srv_serror_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "rerror_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "srv_rerror_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "same_srv_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "diff_srv_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "srv_diff_host_rate", default=0.0), 0.0),
        _coerce_int(_pick(normalized, "dst_host_count", default=0), 0),
        _coerce_int(_pick(normalized, "dst_host_srv_count", default=0), 0),
        _coerce_float(_pick(normalized, "dst_host_same_srv_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "dst_host_diff_srv_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "dst_host_same_src_port_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "dst_host_srv_diff_host_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "dst_host_serror_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "dst_host_srv_serror_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "dst_host_rerror_rate", default=0.0), 0.0),
        _coerce_float(_pick(normalized, "dst_host_srv_rerror_rate", default=0.0), 0.0),
    ]


def _predict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    vector = _build_feature_vector(payload)
    try:
        return _model_client.get_prediction(vector)
    except RuntimeError as exc:
        logger.error("Inference error: %s", exc)
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during inference")
        raise RuntimeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/ui")
def ui():
    """Serve the browser front-end."""
    return render_template("index.html")


@app.get("/health")
def health():
    """Liveness probe — returns 200 when the service is running."""
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    """Classify a single network traffic record as Normal or Anomaly."""
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400

    if not isinstance(body, dict):
        return jsonify({"error": "Expected a JSON object with traffic fields"}), 400

    try:
        result = _predict_payload(body)
    except RuntimeError as exc:
        return jsonify({"error": "Model inference failed", "details": str(exc)}), 502

    return jsonify(result), 200


@app.post("/predict/csv")
def predict_csv():
    """Classify a batch of rows uploaded as CSV."""
    if "file" not in request.files:
        return jsonify({"error": "Please upload a CSV file"}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Please choose a CSV file"}), 400

    stream = io.StringIO(uploaded.stream.read().decode("utf-8-sig"))
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        return jsonify({"error": "CSV file is empty or has no header row"}), 400

    rows = []
    for row in reader:
        cleaned = {}
        for key, value in row.items():
            if key is None:
                continue
            cleaned[str(key).strip().lower()] = value
        rows.append(cleaned)

    if not rows:
        return jsonify({"error": "No rows found in CSV file"}), 400

    results = []
    for row in rows:
        try:
            results.append(_predict_payload(row))
        except RuntimeError as exc:
            results.append({"error": str(exc), "row": row})

    return jsonify({"count": len(results), "results": results}), 200


@app.get("/schema")
def get_schema():
    """Return the JSON Schema for the expected TrafficRecord input."""
    return jsonify(TrafficRecord.model_json_schema()), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
