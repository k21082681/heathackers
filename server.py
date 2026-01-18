#!/usr/bin/env python3
"""
FastAPI Server for PCM Heat Recovery ML System
Loads trained models and serves predictions

UPDATED:
- /predict now accepts EITHER:
  A) {"features": [40 floats]}  (ordered)
  B) {"Tin": ..., "Tout": ..., ...} (named dict; server orders by feature_names.json)
- /health returns model_type consistent with winner.txt
"""

import json
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import time

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(
    title="PCM Heat Recovery ML Server",
    description="ML predictions for heat recovery monitoring",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"

FEATURE_NAMES: Optional[List[str]] = None
RESIDUAL_STDS: Optional[Dict[str, float]] = None
MODELS: Dict[str, Any] = {}
WINNER: Optional[str] = None
STARTUP_TIME = time.time()

TARGETS = ["yQ_kWh_next_window", "time_to_x95_min", "x_next", "Tout_next"]

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictRequest(BaseModel):
    """
    Accept either:
      - features: ordered list
      - or arbitrary named keys (extra fields) which we will map to FEATURE_NAMES
    """
    features: Optional[List[float]] = None

    class Config:
        extra = "allow"  # allow named features like Tin, Tout, ...

class PredictResponse(BaseModel):
    yQ: float
    yTcharge_min: float
    x_next: float
    Tout_next: float
    qBands: dict
    confidence: float
    model_type: str
    latency_ms: float

# ============================================================================
# STARTUP - LOAD MODELS
# ============================================================================

@app.on_event("startup")
async def load_models():
    global FEATURE_NAMES, RESIDUAL_STDS, MODELS, WINNER

    print("\n" + "=" * 70)
    print("🚀 PCM HEAT RECOVERY ML SERVER STARTUP")
    print("=" * 70)

    feature_file = ARTIFACTS_DIR / "feature_names.json"
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing {feature_file}")

    with open(feature_file, "r") as f:
        FEATURE_NAMES = json.load(f)
    print(f"✓ Loaded feature_names.json ({len(FEATURE_NAMES)} features)")

    residual_file = ARTIFACTS_DIR / "residual_std.json"
    if not residual_file.exists():
        raise FileNotFoundError(f"Missing {residual_file}")

    with open(residual_file, "r") as f:
        RESIDUAL_STDS = json.load(f)
    print("✓ Loaded residual_std.json")

    winner_file = ARTIFACTS_DIR / "winner.txt"
    if not winner_file.exists():
        raise FileNotFoundError(f"Missing {winner_file}")

    with open(winner_file, "r") as f:
        WINNER = f.read().strip()
    print(f"✓ Model winner: {WINNER.upper()}")

    for target in TARGETS:
        model_path = MODELS_DIR / f"{WINNER}_{target}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}")

        with open(model_path, "rb") as f:
            MODELS[target] = pickle.load(f)
        print(f"  ✓ {target}")

    print("\n✅ Server ready on http://localhost:8000")
    print("   POST /predict — ML inference")
    print("   GET  /health  — Server status")
    print("   GET  /config  — Model configuration")
    print("=" * 70 + "\n")

# ============================================================================
# HELPERS
# ============================================================================

def _to_ordered_vector(req: PredictRequest) -> np.ndarray:
    """
    Convert request into a 2D numpy array shape (1, num_features)
    Supports:
      - req.features (ordered list)
      - or named feature fields in request body
    """
    if FEATURE_NAMES is None:
        raise RuntimeError("FEATURE_NAMES not loaded")

    # Case A: already ordered list
    if req.features is not None:
        if len(req.features) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(req.features)}")
        X = np.array([req.features], dtype=np.float32)
        return X

    # Case B: named fields
    payload = req.dict()
    payload.pop("features", None)

    missing = [name for name in FEATURE_NAMES if name not in payload]
    if missing:
        # You can choose to fill missing with 0 instead of error, but error is safer.
        # If you want "fill with 0" behavior, replace this raise with fill logic.
        raise ValueError(f"Missing feature(s): {missing[:12]}{'...' if len(missing) > 12 else ''}")

    ordered = [float(payload[name]) for name in FEATURE_NAMES]
    X = np.array([ordered], dtype=np.float32)
    return X

def _residual_key(model_name: str, target: str) -> str:
    # train scripts commonly store like: "rf_yQ_kWh_next_window" or "xgb_yQ_kWh_next_window"
    return f"{model_name}_{target}"

# ============================================================================
# API ENDPOINTS
# ============================================================================
from fastapi.responses import FileResponse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
def root():
    return FileResponse(BASE_DIR / "index.html")



@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.time()

    try:
        if WINNER is None or RESIDUAL_STDS is None:
            raise RuntimeError("Server not fully initialized")

        X = _to_ordered_vector(request)

        yQ = float(MODELS["yQ_kWh_next_window"].predict(X)[0])
        yTcharge = float(MODELS["time_to_x95_min"].predict(X)[0])
        x_next = float(MODELS["x_next"].predict(X)[0])
        Tout_next = float(MODELS["Tout_next"].predict(X)[0])

        # Physical clamps
        yQ = max(0.0, yQ)
        x_next = float(np.clip(x_next, 0.0, 1.0))

        # Uncertainty bands using residual std (P10/P50/P90)
        std_key = _residual_key(WINNER, "yQ_kWh_next_window")
        std_yQ = float(RESIDUAL_STDS.get(std_key, 0.04))

        qBands = {
            "p10": max(0.0, yQ - 1.28 * std_yQ),
            "p50": yQ,
            "p90": yQ + 1.28 * std_yQ,
        }

        confidence = 1.0 / (1.0 + std_yQ)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        latency_ms = (time.time() - start_time) * 1000.0

        return PredictResponse(
            yQ=yQ,
            yTcharge_min=yTcharge,
            x_next=x_next,
            Tout_next=Tout_next,
            qBands=qBands,
            confidence=confidence,
            model_type=WINNER,
            latency_ms=latency_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_seconds": time.time() - STARTUP_TIME,
        "num_features":  0 if FEATURE_NAMES is None else len(FEATURE_NAMES),
        "targets": list(MODELS.keys()),
        "model_type": WINNER or "unknown",
    }

@app.get("/config")
async def config():
    return {
        "feature_names": FEATURE_NAMES,
        "num_features": 0 if FEATURE_NAMES is None else len(FEATURE_NAMES),
        "targets": list(MODELS.keys()),
        "model_type": WINNER or "unknown",
        "uncertainty_method": "residual_std_p10_p50_p90",
        "predict_input_formats": [
            {"features": "[ordered list length num_features]"},
            {"<feature_name>": "value, ... (server orders using feature_names.json)"},
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
