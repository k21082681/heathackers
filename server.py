#!/usr/bin/env python3
"""
PCM Heat Recovery: FastAPI Server with Safety Features
Loads trained models and serves predictions with real-time safety monitoring
"""

import json
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import time

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(
    title="PCM Heat Recovery ML Server",
    description="ML predictions for heat recovery monitoring with safety features",
    version="2.0.0"
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

# Safety thresholds
SAFETY_THRESHOLDS = {
    'ua_deviation_max': 0.15,
    'temp_gradient_max': 10.0,
    'health_score_warning': 0.7,
    'health_score_critical': 0.5,
    'anomaly_consecutive_max': 3,
    'energy_balance_tolerance': 0.20
}

BASELINE_UA = 1040.0
SHUTDOWN_STATE = {"in_progress": False, "step": 0, "start_time": None}
ANOMALY_HISTORY = []
CRITICAL_WARNING_STATE = {"active": False, "start_time": None, "acknowledged": False}
# Critical warning countdown state
CRITICAL_COUNTDOWN_SECONDS = 30



# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictRequest(BaseModel):
    features: Optional[List[float]] = None
    class Config:
        extra = "allow"

class PredictResponse(BaseModel):
    yQ: float
    yTcharge_min: float
    x_next: float
    Tout_next: float
    qBands: dict
    confidence: float
    model_type: str
    latency_ms: float
    safety_status: str
    requires_manual_intervention: bool
    safe_shutdown_recommended: bool
    health_score: float
    critical_countdown_active: bool
    critical_countdown_remaining: Optional[int]

class SafetyCheckResponse(BaseModel):
    safe_to_operate: bool
    safety_status: str
    issues: List[str]
    recommendation: str
    health_score: float
    timestamp: float
    critical_countdown_active: bool
    critical_countdown_remaining: Optional[int]

class VerificationResponse(BaseModel):
    health_status: str
    can_proceed: bool
    warnings: List[str]
    checks_passed: Dict[str, bool]
    timestamp: float

class EmergencyShutdownResponse(BaseModel):
    initiated: bool
    shutdown_steps: List[str]
    estimated_time_seconds: int
    manual_actions_required: List[str]
    current_step: int
    timestamp: float

class AcknowledgeCriticalRequest(BaseModel):
    acknowledged: bool

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
    print("  POST /predict — ML inference with safety")
    print("  POST /verify_pcm_health — Pre-operation verification")
    print("  POST /safety_check — Continuous safety monitoring")
    print("  POST /emergency_shutdown — Safe shutdown procedure")
    print("  GET /health — Server status")
    print("=" * 70 + "\n")

# ============================================================================
# SAFETY HELPER FUNCTIONS
# ============================================================================

def check_ua_health(current_ua: float, baseline_ua: float) -> tuple:
    deviation = abs(current_ua - baseline_ua) / baseline_ua
    max_dev = SAFETY_THRESHOLDS['ua_deviation_max']
    if deviation > max_dev:
        return False, f"UA deviation {deviation:.1%} exceeds threshold {max_dev:.1%}"
    return True, "UA within normal range"

def check_temperature_consistency(Tpcm_top: float, Tpcm_mid: float, Tpcm_bot: float) -> tuple:
    temps = [Tpcm_top, Tpcm_mid, Tpcm_bot]
    gradient = max(temps) - min(temps)
    max_grad = SAFETY_THRESHOLDS['temp_gradient_max']
    if gradient > max_grad:
        return False, f"Temperature gradient {gradient:.1f}°C exceeds {max_grad}°C"
    return True, "Temperature gradient acceptable"

def check_sensor_health(feature_dict: dict) -> tuple:
    required_sensors = ['Tin', 'Tout', 'mdot', 'dp', 'Tpcm_top', 'Tpcm_mid', 'Tpcm_bot']
    for sensor in required_sensors:
        if sensor not in feature_dict:
            return False, f"Missing sensor: {sensor}"
        value = feature_dict[sensor]
        if np.isnan(value) or np.isinf(value):
            return False, f"Invalid reading from {sensor}: {value}"
        if sensor.startswith('T') and (value < -50 or value > 150):
            return False, f"{sensor} out of physical range: {value}°C"
        if sensor == 'mdot' and (value < 0 or value > 10):
            return False, f"Flow rate out of range: {value} kg/s"
    return True, "All sensors healthy"

def check_recent_anomalies() -> tuple:
    global ANOMALY_HISTORY
    if len(ANOMALY_HISTORY) < 3:
        return True, "Insufficient history for anomaly check"
    recent = ANOMALY_HISTORY[-SAFETY_THRESHOLDS['anomaly_consecutive_max']:]
    consecutive_anomalies = sum(recent)
    if consecutive_anomalies >= SAFETY_THRESHOLDS['anomaly_consecutive_max']:
        return False, f"Detected {consecutive_anomalies} consecutive anomalies"
    return True, "No recent anomaly pattern detected"

def get_safety_status(health_score: float) -> str:
    if health_score >= SAFETY_THRESHOLDS['health_score_warning']:
        return "SAFE"
    elif health_score >= SAFETY_THRESHOLDS['health_score_critical']:
        return "WARNING"
    else:
        return "CRITICAL"

def manage_critical_countdown(safety_status: str) -> tuple:
    """Manage 30-second countdown when entering CRITICAL state"""
    global CRITICAL_WARNING_STATE
    
    if safety_status == "CRITICAL":
        if not CRITICAL_WARNING_STATE["active"]:
            CRITICAL_WARNING_STATE["active"] = True
            CRITICAL_WARNING_STATE["start_time"] = time.time()
            CRITICAL_WARNING_STATE["acknowledged"] = False
        
        elapsed = time.time() - CRITICAL_WARNING_STATE["start_time"]
        remaining = max(0, 30 - int(elapsed))
        
        return True, remaining
    else:
        if CRITICAL_WARNING_STATE["active"]:
            CRITICAL_WARNING_STATE = {"active": False, "start_time": None, "acknowledged": False}
        return False, None

# ============================================================================
# HELPERS
# ============================================================================

def _to_ordered_vector(req: PredictRequest) -> np.ndarray:
    if FEATURE_NAMES is None:
        raise RuntimeError("FEATURE_NAMES not loaded")
    
    if req.features is not None:
        if len(req.features) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(req.features)}")
        X = np.array([req.features], dtype=np.float32)
        return X
    
    payload = req.dict()
    payload.pop("features", None)
    missing = [name for name in FEATURE_NAMES if name not in payload]
    if missing:
        raise ValueError(f"Missing feature(s): {missing[:12]}{'...' if len(missing) > 12 else ''}")
    ordered = [float(payload[name]) for name in FEATURE_NAMES]
    X = np.array([ordered], dtype=np.float32)
    return X

def _residual_key(model_name: str, target: str) -> str:
    return f"{model_name}_{target}"

# ============================================================================
# API ENDPOINTS
# ============================================================================

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
        feature_dict = dict(zip(FEATURE_NAMES, X[0]))
        
        yQ = float(MODELS["yQ_kWh_next_window"].predict(X)[0])
        yTcharge = float(MODELS["time_to_x95_min"].predict(X)[0])
        x_next = float(MODELS["x_next"].predict(X)[0])
        Tout_next = float(MODELS["Tout_next"].predict(X)[0])
        
        yQ = max(0.0, yQ)
        x_next = float(np.clip(x_next, 0.0, 1.0))
        
        std_key = _residual_key(WINNER, "yQ_kWh_next_window")
        std_yQ = float(RESIDUAL_STDS.get(std_key, 0.04))
        
        qBands = {
            "p10": max(0.0, yQ - 1.28 * std_yQ),
            "p50": yQ,
            "p90": yQ + 1.28 * std_yQ,
        }
        
        confidence = 1.0 / (1.0 + std_yQ)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Safety check
        health_score = 1.0
        ua_ok, _ = check_ua_health(feature_dict.get('keff', BASELINE_UA), BASELINE_UA)
        if not ua_ok:
            health_score *= 0.7
        
        temp_ok, _ = check_temperature_consistency(
            feature_dict.get('Tpcm_top', 56),
            feature_dict.get('Tpcm_mid', 56),
            feature_dict.get('Tpcm_bot', 56)
        )
        if not temp_ok:
            health_score *= 0.6
        
        sensor_ok, _ = check_sensor_health(feature_dict)
        if not sensor_ok:
            health_score *= 0.3
        
        safety_status = get_safety_status(health_score)
        countdown_active, countdown_remaining = manage_critical_countdown(safety_status)
        
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
            safety_status=safety_status,
            requires_manual_intervention=(safety_status == "CRITICAL"),
            safe_shutdown_recommended=(health_score < 0.5),
            health_score=health_score,
            critical_countdown_active=countdown_active,
            critical_countdown_remaining=countdown_remaining
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/verify_pcm_health", response_model=VerificationResponse)
async def verify_pcm_health(request: PredictRequest):
    try:
        X = _to_ordered_vector(request)
        feature_dict = dict(zip(FEATURE_NAMES, X[0]))
        
        checks = {}
        warnings = []
        
        sensor_ok, sensor_msg = check_sensor_health(feature_dict)
        checks['sensor_health'] = sensor_ok
        if not sensor_ok:
            warnings.append(sensor_msg)
        
        current_ua = feature_dict.get('keff', BASELINE_UA)
        ua_ok, ua_msg = check_ua_health(current_ua, BASELINE_UA)
        checks['ua_health'] = ua_ok
        if not ua_ok:
            warnings.append(ua_msg)
        
        temp_ok, temp_msg = check_temperature_consistency(
            feature_dict['Tpcm_top'],
            feature_dict['Tpcm_mid'],
            feature_dict['Tpcm_bot']
        )
        checks['temperature_consistency'] = temp_ok
        if not temp_ok:
            warnings.append(temp_msg)
        
        anomaly_ok, anomaly_msg = check_recent_anomalies()
        checks['anomaly_history'] = anomaly_ok
        if not anomaly_ok:
            warnings.append(anomaly_msg)
        
        all_passed = all(checks.values())
        critical_passed = checks['sensor_health'] and checks['temperature_consistency']
        
        if all_passed:
            health_status = "HEALTHY"
            can_proceed = True
        elif critical_passed:
            health_status = "DEGRADED"
            can_proceed = True
            warnings.append("System degraded but operational - monitor closely")
        else:
            health_status = "UNHEALTHY"
            can_proceed = False
            warnings.append("CANNOT PROCEED - manual inspection required")
        
        return VerificationResponse(
            health_status=health_status,
            can_proceed=can_proceed,
            warnings=warnings,
            checks_passed=checks,
            timestamp=time.time()
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Verification failed: {str(e)}")

@app.post("/safety_check", response_model=SafetyCheckResponse)
async def safety_check(request: PredictRequest):
    try:
        X = _to_ordered_vector(request)
        feature_dict = dict(zip(FEATURE_NAMES, X[0]))
        
        health_score = 1.0
        issues = []
        
        ua_ok, ua_msg = check_ua_health(feature_dict.get('keff', BASELINE_UA), BASELINE_UA)
        if not ua_ok:
            health_score *= 0.7
            issues.append(ua_msg)
        
        temp_ok, temp_msg = check_temperature_consistency(
            feature_dict.get('Tpcm_top', 56),
            feature_dict.get('Tpcm_mid', 56),
            feature_dict.get('Tpcm_bot', 56)
        )
        if not temp_ok:
            health_score *= 0.6
            issues.append(temp_msg)
        
        sensor_ok, sensor_msg = check_sensor_health(feature_dict)
        if not sensor_ok:
            health_score *= 0.3
            issues.append(sensor_msg)
        
        safety_status = get_safety_status(health_score)
        safe_to_operate = safety_status != "CRITICAL"
        countdown_active, countdown_remaining = manage_critical_countdown(safety_status)
        
        if safety_status == "CRITICAL":
            recommendation = "IMMEDIATE ACTION REQUIRED: Initiate safe shutdown and request manual intervention"
        elif safety_status == "WARNING":
            recommendation = "Elevated monitoring required. Consider scheduling maintenance."
        else:
            recommendation = "System operating normally"
        
        return SafetyCheckResponse(
            safe_to_operate=safe_to_operate,
            safety_status=safety_status,
            issues=issues,
            recommendation=recommendation,
            health_score=health_score,
            timestamp=time.time(),
            critical_countdown_active=countdown_active,
            critical_countdown_remaining=countdown_remaining
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Safety check failed: {str(e)}")

@app.post("/acknowledge_critical")
async def acknowledge_critical(request: AcknowledgeCriticalRequest):
    global CRITICAL_WARNING_STATE
    if request.acknowledged:
        CRITICAL_WARNING_STATE["acknowledged"] = True
    return {"acknowledged": CRITICAL_WARNING_STATE["acknowledged"]}

@app.post("/emergency_shutdown", response_model=EmergencyShutdownResponse)
async def emergency_shutdown(request: Dict[str, Any]):
    global SHUTDOWN_STATE
    
    shutdown_steps = [
        "Step 1: Set operating mode to HOLD (0)",
        "Step 2: Reduce pump speed to 20% over 60 seconds",
        "Step 3: Open bypass valve to 80%",
        "Step 4: Monitor PCM temperatures until gradient < 5°C",
        "Step 5: Shut down pump completely",
        "Step 6: Close all valves",
        "Step 7: System in safe state - ready for manual inspection"
    ]
    
    manual_actions = [
        "1. Verify all temperatures are stable and decreasing",
        "2. Check for any leaks or unusual sounds",
        "3. Inspect PCM module for physical damage",
        "4. Review system logs for error patterns",
        "5. Contact maintenance team: [CONTACT INFO]",
        "6. Do not restart until inspection complete and logged"
    ]
    
    if not SHUTDOWN_STATE["in_progress"]:
        SHUTDOWN_STATE = {
            "in_progress": True,
            "step": 0,
            "start_time": time.time()
        }
    
    estimated_time = 300
    
    return EmergencyShutdownResponse(
        initiated=True,
        shutdown_steps=shutdown_steps,
        estimated_time_seconds=estimated_time,
        manual_actions_required=manual_actions,
        current_step=SHUTDOWN_STATE["step"],
        timestamp=time.time()
    )

@app.get("/shutdown_status")
async def get_shutdown_status():
    return {
        "in_progress": SHUTDOWN_STATE["in_progress"],
        "current_step": SHUTDOWN_STATE["step"],
        "elapsed_time": time.time() - SHUTDOWN_STATE["start_time"] if SHUTDOWN_STATE["start_time"] else 0
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_seconds": time.time() - STARTUP_TIME,
        "num_features": 0 if FEATURE_NAMES is None else len(FEATURE_NAMES),
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
        "safety_thresholds": SAFETY_THRESHOLDS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)