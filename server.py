#!/usr/bin/env python3
"""
FastAPI Server for PCM Heat Recovery ML System
Loads trained RandomForest models and serves predictions
"""

import json
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(
    title="PCM Heat Recovery ML Server",
    description="RandomForest predictions for heat recovery monitoring",
    version="1.0.0"
)

# Enable CORS (allow HTML dashboard to call this server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE (loaded once at startup)
# ============================================================================

ARTIFACTS_DIR = Path('artifacts')
MODELS_DIR = ARTIFACTS_DIR / 'models'

# Loaded at startup
FEATURE_NAMES = None
RESIDUAL_STDS = None
MODELS = {}
STARTUP_TIME = time.time()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictRequest(BaseModel):
    """Feature vector (40 values in correct order)"""
    features: list  # 40 floats

class PredictResponse(BaseModel):
    """ML prediction output"""
    yQ: float  # Heat recovery next window [kWh]
    yTcharge_min: float  # Time to 95% melt [min]
    x_next: float  # Next melt fraction [0-1]
    Tout_next: float  # Next outlet temp [°C]
    qBands: dict  # {"p10": float, "p50": float, "p90": float}
    confidence: float  # [0-1]
    model_type: str  # "rf"
    latency_ms: float

# ============================================================================
# STARTUP - LOAD MODELS
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load trained models and metadata at server startup"""
    global FEATURE_NAMES, RESIDUAL_STDS, MODELS
    
    print("\n" + "="*70)
    print("🚀 PCM HEAT RECOVERY ML SERVER STARTUP")
    print("="*70)
    
    # Load feature names
    feature_file = ARTIFACTS_DIR / 'feature_names.json'
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing {feature_file}")
    
    with open(feature_file, 'r') as f:
        FEATURE_NAMES = json.load(f)
    print(f"✓ Loaded feature_names.json ({len(FEATURE_NAMES)} features)")
    
    # Load residual stds (for uncertainty quantification)
    residual_file = ARTIFACTS_DIR / 'residual_std.json'
    if not residual_file.exists():
        raise FileNotFoundError(f"Missing {residual_file}")
    
    with open(residual_file, 'r') as f:
        RESIDUAL_STDS = json.load(f)
    print(f"✓ Loaded residual_std.json")
    
    # Load winner
    winner_file = ARTIFACTS_DIR / 'winner.txt'
    if not winner_file.exists():
        raise FileNotFoundError(f"Missing {winner_file}")
    
    with open(winner_file, 'r') as f:
        winner = f.read().strip()
    print(f"✓ Model winner: {winner.upper()}")
    
    # Load RF models (only loading winner models now)
    targets = ['yQ_kWh_next_window', 'time_to_x95_min', 'x_next', 'Tout_next']
    for target in targets:
        model_path = MODELS_DIR / f'{winner}_{target}.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}")
        
        with open(model_path, 'rb') as f:
            MODELS[target] = pickle.load(f)
        print(f"  ✓ {target}")
    
    print(f"\n✅ Server ready on http://localhost:8000")
    print(f"   POST /predict — ML inference (5-20ms)")
    print(f"   GET  /health  — Server status")
    print(f"   GET  /config  — Model configuration")
    print("="*70 + "\n")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make predictions given 40-feature vector
    
    Input: 40 features in exact order (see FEATURE_NAMES)
    Output: yQ, yTcharge_min, x_next, Tout_next + uncertainty bands
    """
    start_time = time.time()
    
    try:
        # Validate input
        if len(request.features) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, got {len(request.features)}"
            )
        
        # Convert to numpy array
        X = np.array([request.features], dtype=np.float32)
        
        # Make predictions
        yQ = float(MODELS['yQ_kWh_next_window'].predict(X)[0])
        yTcharge = float(MODELS['time_to_x95_min'].predict(X)[0])
        x_next = float(MODELS['x_next'].predict(X)[0])
        Tout_next = float(MODELS['Tout_next'].predict(X)[0])
        
        # Clamp to physical bounds
        yQ = max(0, yQ)
        x_next = np.clip(x_next, 0, 1)
        
        # Compute uncertainty bands (P10/P50/P90)
        # P10 = median - 1.28 * std
        # P50 = median (our prediction)
        # P90 = median + 1.28 * std
        
        std_yQ = RESIDUAL_STDS.get('rf_yQ_kWh_next_window', 0.04)
        
        qBands = {
            'p10': max(0, yQ - 1.28 * std_yQ),
            'p50': yQ,
            'p90': yQ + 1.28 * std_yQ,
        }
        
        # Confidence = 1 / (1 + residual_std)
        confidence = 1.0 / (1.0 + std_yQ)
        confidence = np.clip(confidence, 0, 1)
        
        # Latency
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictResponse(
            yQ=yQ,
            yTcharge_min=yTcharge,
            x_next=x_next,
            Tout_next=Tout_next,
            qBands=qBands,
            confidence=confidence,
            model_type="rf",
            latency_ms=latency_ms,
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    """Server health check"""
    return {
        "status": "ok",
        "uptime_seconds": time.time() - STARTUP_TIME,
        "models_loaded": len(MODELS),
        "model_type": "RandomForest",
    }

@app.get("/config")
async def config():
    """Return model configuration"""
    return {
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
        "targets": list(MODELS.keys()),
        "model_type": "RandomForest",
        "uncertainty_method": "residual_std_p10_p50_p90",
    }

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
