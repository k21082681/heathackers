# PCM Heat Recovery: Complete ML Pipeline Guide

## Overview

This package contains:
- **train.py**: Train XGBoost and Random Forest models with 4 targets each, auto-select winner
- **server.py**: FastAPI endpoint for real-time inference with uncertainty bands
- **edge_model_integration.py**: JavaScript integration guide + Python sensor state manager
- **requirements.txt**: All dependencies

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create `data/pcm.csv` with these columns (minimum):

```
timestamp,Tin,Tout,mdot,dp,Tpcm_top,Tpcm_mid,Tpcm_bot,mode,prevMode,timeInMode,valvePct,pumpPct,bypassFrac
2024-01-01T00:00:00,40.5,35.2,0.45,5.2,48.1,54.6,51.3,1,-1,120,75,85,0.15
2024-01-01T00:01:00,40.8,35.4,0.46,5.1,48.5,55.0,51.8,1,1,180,75,85,0.15
...
```

**Note**: Target columns (yQ_kWh_next_window, time_to_x95_min, x_next, Tout_next) are optional.
If missing, train.py will compute them from shifted windows.

#### Column Descriptions

| Column | Unit | Description |
|--------|------|-------------|
| timestamp | ISO 8601 | Timestamp (optional) |
| Tin | °C | Inlet temperature to PCM |
| Tout | °C | Outlet temperature from PCM |
| mdot | kg/s | Heat transfer fluid mass flow rate |
| dp | kPa | Pressure drop across PCM |
| Tpcm_top | °C | PCM top sensor temperature |
| Tpcm_mid | °C | PCM middle sensor temperature |
| Tpcm_bot | °C | PCM bottom sensor temperature |
| mode | -1/0/1 | Operating mode (discharge/hold/charge) |
| prevMode | -1/0/1 | Previous mode |
| timeInMode | s | Seconds in current mode |
| valvePct | 0-100 | Valve position setpoint |
| pumpPct | 0-100 | Pump speed setpoint |
| bypassFrac | 0-1 | Fraction of flow bypassing PCM |
| yQ_kWh_next_window | kWh | [Optional] Heat recovered in next window |
| time_to_x95_min | min | [Optional] Time to reach 95% melt |
| x_next | 0-1 | [Optional] Melt fraction next step |
| Tout_next | °C | [Optional] Outlet temp next step |

### 3. Adjust PCM Constants (Optional)

Edit `train.py` to match your system:

```python
# Line ~40: PCM_VOLUME_M3 = 0.05  # Change to your volume [m³]
# PCM mass will be auto-calculated
```

If your CSV has a `mass_kg` column, modify `compute_enthalpy_observer()` to use it.

### 4. Train Models

```bash
python train.py
```

**Output**:
- `artifacts/feature_names.json` → Feature order for predictions
- `artifacts/metrics.json` → MAE/RMSE for all model combinations
- `artifacts/residual_std.json` → Std dev of residuals (for P10/P50/P90)
- `artifacts/winner.txt` → Selected model (xgb or rf)
- `artifacts/models/` → Pickle files for each target

Example output:
```
================================================================================
PCM HEAT RECOVERY: XGBoost vs Random Forest Training Pipeline
================================================================================

[1] Loading data/pcm.csv...
    Loaded 10000 rows, 14 columns

[2] Engineering features...
[3] Computing targets (if missing)...
    After dropping NaNs: 9940 rows

[4] Preparing feature matrices...
    Feature matrix shape: (9940, 40)

[5] Splitting data (70% train / 15% val / 15% test)...
    Train: 6958, Val: 1491, Test: 1491

[6] Training models...

    Target: yQ_kWh_next_window
      Training XGBoost...
      Training Random Forest...
      Evaluating on test set...
        XGBoost  - MAE: 0.1234, RMSE: 0.1567
        RF       - MAE: 0.1189, RMSE: 0.1456

    Target: time_to_x95_min
      ...

[7] Comparing models...
    WINNER: XGBoost (normalized MAE: 0.9234 vs 0.9876)

[8] Saving artifacts...
    ✓ feature_names.json
    ✓ metrics.json
    ✓ residual_std.json
    ✓ winner.txt
    ✓ xgb_yQ_kWh_next_window.pkl
    ✓ xgb_time_to_x95_min.pkl
    ✓ xgb_x_next.pkl
    ✓ xgb_Tout_next.pkl

Training complete! Ready to run server.py
================================================================================
```

### 5. Start the Server

```bash
python server.py
```

Should see:
```
Starting PCM Edge Model Server on http://localhost:8000
Health check: http://localhost:8000/health
API docs: http://localhost:8000/docs
```

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "model": "xgb",
  "targets": ["yQ_kWh_next_window", "time_to_x95_min", "x_next", "Tout_next"],
  "features": 40
}
```

---

## API Usage

### POST /predict

**Request** (JSON):
```json
{
  "Tin": 40.5,
  "Tout": 35.2,
  "mdot": 0.45,
  "dp": 5.2,
  "Tpcm_top": 48.1,
  "Tpcm_mid": 54.6,
  "Tpcm_bot": 51.3,
  "Tpcm_avg": 51.33,
  "dT": 5.3,
  "Qdot": 9.87,
  "Qpcm_signed": 8.39,
  "melt_fraction_x": 0.45,
  "Erem_kWh": 65.2,
  "plateauFlag": 0.0,
  "keff": 0.32,
  "mode_charge": 1.0,
  "mode_discharge": 0.0,
  "mode_hold": 0.0,
  "prevMode_charge": 0.0,
  "prevMode_discharge": 1.0,
  "prevMode_hold": 0.0,
  "timeInMode": 180.0,
  "valvePct": 75.0,
  "pumpPct": 85.0,
  "bypassFrac": 0.15,
  "Qdot_lag1": 10.2,
  "Qdot_lag2": 9.8,
  "Qdot_lag3": 9.5,
  "x_lag1": 0.43,
  "x_lag2": 0.41,
  "x_lag3": 0.39,
  "Tin_lag1": 39.8,
  "Tin_lag2": 39.5,
  "Tin_lag3": 39.2,
  "TpcmAvg_lag1": 52.0,
  "TpcmAvg_lag2": 51.5,
  "TpcmAvg_lag3": 51.0
}
```

**Response** (JSON):
```json
{
  "yQ": 0.245,
  "yTcharge_min": 28.5,
  "x_next": 0.456,
  "Tout_next": 35.4,
  "qBands": {
    "p10": 0.201,
    "p50": 0.245,
    "p90": 0.289
  },
  "chargeBands": {
    "p10": 24.1,
    "p50": 28.5,
    "p90": 32.9
  },
  "confidence": 0.892,
  "model_type": "xgb",
  "latency_ms": 12.34
}
```

### GET /health

Returns status and configuration:
```json
{
  "status": "ok",
  "model": "xgb",
  "targets": ["yQ_kWh_next_window", "time_to_x95_min", "x_next", "Tout_next"],
  "features": 40
}
```

### GET /config

Returns model configuration for client-side code:
```json
{
  "model_type": "xgb",
  "targets": ["yQ_kWh_next_window", "time_to_x95_min", "x_next", "Tout_next"],
  "feature_order": ["Tin", "Tout", "mdot", ...],
  "residual_stds": {"xgb_yQ_kWh_next_window": 0.0456, ...}
}
```

---

## HTML/JavaScript Integration

### Add This to Your HTML

Place in the `<head>` or before closing `</body>`:

```html
<script>
// Global sensor state (update from your data source)
const sensorState = {
    Tin: 40.0,
    Tout: 35.0,
    mdot: 0.5,
    dp: 5.0,
    Tpcm_top: 50.0,
    Tpcm_mid: 55.0,
    Tpcm_bot: 52.0,
    Tpcm_avg: 52.33,
    dT: 5.0,
    Qdot: 10.45,
    Qpcm_signed: 8.36,
    melt_fraction_x: 0.45,
    Erem_kWh: 65.2,
    plateauFlag: 0.0,
    keff: 0.32,
    mode: 1.0,
    prevMode: 1.0,
    timeInMode: 180.0,
    valvePct: 75.0,
    pumpPct: 85.0,
    bypassFrac: 0.15,
    Qdot_lag1: 10.2,
    Qdot_lag2: 9.8,
    Qdot_lag3: 9.5,
    x_lag1: 0.43,
    x_lag2: 0.41,
    x_lag3: 0.39,
    Tin_lag1: 39.8,
    Tin_lag2: 39.5,
    Tin_lag3: 39.2,
    TpcmAvg_lag1: 52.0,
    TpcmAvg_lag2: 51.5,
    TpcmAvg_lag3: 51.0,
};

// Update mode one-hot encoding
function updateModeOneHot() {
    const mode = sensorState.mode;
    sensorState.mode_charge = mode === 1.0 ? 1.0 : 0.0;
    sensorState.mode_hold = mode === 0.0 ? 1.0 : 0.0;
    sensorState.mode_discharge = mode === -1.0 ? 1.0 : 0.0;
    
    const prevMode = sensorState.prevMode;
    sensorState.prevMode_charge = prevMode === 1.0 ? 1.0 : 0.0;
    sensorState.prevMode_hold = prevMode === 0.0 ? 1.0 : 0.0;
    sensorState.prevMode_discharge = prevMode === -1.0 ? 1.0 : 0.0;
}

// Main inference function - call every second
async function edgeModelInfer() {
    updateModeOneHot();
    
    const t0 = performance.now();
    
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(sensorState),
        });
        
        if (!response.ok) {
            console.error(`Inference failed: HTTP ${response.status}`);
            document.getElementById('latency').textContent = 'ERR';
            return;
        }
        
        const result = await response.json();
        const latency = performance.now() - t0;
        
        // Update DOM with predictions
        document.getElementById('yQ').textContent = result.yQ.toFixed(2);
        document.getElementById('yTcharge').textContent = result.yTcharge_min.toFixed(1);
        document.getElementById('xNext').textContent = result.x_next.toFixed(3);
        document.getElementById('ToutNext').textContent = result.Tout_next.toFixed(1);
        
        // Update bands (P10/P50/P90)
        document.getElementById('qBands').textContent =
            `P10: ${result.qBands.p10.toFixed(2)} | ` +
            `P50: ${result.qBands.p50.toFixed(2)} | ` +
            `P90: ${result.qBands.p90.toFixed(2)}`;
        
        // Update latency
        document.getElementById('latency').textContent = latency.toFixed(1) + ' ms';
        
    } catch (error) {
        console.error('Inference error:', error);
        document.getElementById('latency').textContent = 'OFFLINE';
    }
}

// Call every second
setInterval(edgeModelInfer, 1000);
</script>
```

### Required DOM IDs in Your HTML

Ensure these elements exist in your dashboard:

```html
<span id="yQ">-</span>          <!-- Recovered heat [kWh/window] -->
<span id="yTcharge">-</span>    <!-- Time to x >= 0.95 [min] -->
<span id="xNext">-</span>       <!-- Melt fraction next step -->
<span id="ToutNext">-</span>    <!-- Outlet temp next step [°C] -->
<span id="qBands">-</span>      <!-- P10/P50/P90 for yQ -->
<span id="latency">-</span>     <!-- Inference latency [ms] -->
```

### Updating Sensor Values

When you receive new sensor data (e.g., from MQTT, WebSocket, or REST API):

```javascript
// Example: Update from sensor reading
function onSensorUpdate(data) {
    sensorState.Tin = data.inlet_temp;
    sensorState.Tout = data.outlet_temp;
    sensorState.mdot = data.mass_flow;
    sensorState.Tpcm_top = data.pcm_top_temp;
    sensorState.Tpcm_mid = data.pcm_mid_temp;
    sensorState.Tpcm_bot = data.pcm_bot_temp;
    sensorState.mode = data.operating_mode;  // 1, 0, or -1
    // ... other fields
    
    // Update lags (shift previous values)
    sensorState.Qdot_lag3 = sensorState.Qdot_lag2;
    sensorState.Qdot_lag2 = sensorState.Qdot_lag1;
    sensorState.Qdot_lag1 = sensorState.Qdot;
    
    // Next edgeModelInfer() call will use updated values
}
```

---

## Directory Structure

```
project/
├── data/
│   └── pcm.csv                 # Your training data
├── artifacts/
│   ├── feature_names.json      # Feature order (auto-generated)
│   ├── metrics.json            # Training metrics (auto-generated)
│   ├── residual_std.json       # Uncertainty bands (auto-generated)
│   ├── winner.txt              # Selected model name (auto-generated)
│   └── models/
│       ├── xgb_yQ_kWh_next_window.pkl
│       ├── xgb_time_to_x95_min.pkl
│       ├── xgb_x_next.pkl
│       └── xgb_Tout_next.pkl
├── train.py                    # Training script
├── server.py                   # FastAPI server
├── edge_model_integration.py   # Python sensor state manager (optional)
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── web2.html                   # Your dashboard (updated with JS)
```

---

## Troubleshooting

### ImportError: No module named 'xgboost'

```bash
pip install -r requirements.txt
```

### FileNotFoundError: data/pcm.csv not found

Create the CSV file with your sensor data:
```bash
mkdir -p data
# Add your CSV to data/pcm.csv
```

### Connection refused when calling /predict

Ensure server.py is running:
```bash
python server.py  # Should print "Starting PCM Edge Model Server..."
```

### High latency (>500ms)

- Check server CPU usage
- Reduce feature dimensions (optional engineering)
- Use batch prediction if available

### Predictions seem off

1. Check feature scaling: Are your inputs in reasonable ranges?
2. Review training data: Does test set cover the operating regime?
3. Check residual_std.json: High std = high uncertainty

---

## Advanced Usage

### Custom Model Parameters

Edit `train.py` lines ~280-290:

```python
xgb_model = train_model(
    XGBRegressor,
    X_train, y_train,
    n_estimators=200,        # Increase for better fit
    max_depth=8,             # Increase for more complexity
    learning_rate=0.05,      # Lower = slower training, better generalization
    random_state=42,
)
```

### Custom Feature Selection

Edit `FEATURE_ORDER` in `train.py` to include/exclude features:

```python
FEATURE_ORDER = [
    'Tin', 'Tout', 'mdot', 'dp', 'Tpcm_top', 'Tpcm_mid', 'Tpcm_bot',
    # ... only include features that matter for your system
]
```

### Using Different Time Windows

Edit `OBSERVER_DT_SEC` and window sizes:

```python
OBSERVER_DT_SEC = 30          # 30-second timesteps
window_sec = 120              # 2-minute prediction window
```

### A/B Testing Old vs New Models

Before deploying, train on new data and compare metrics:

```bash
# Keep old model
cp artifacts/models artifacts/models_v1

# Train new version
python train.py

# Compare artifacts/metrics.json (v1 vs current)
```

---

## Contact & Support

- Questions on feature engineering? → See `train.py` comments
- API issues? → Check server.py error logs
- Data format? → See "Prepare Your Data" section
- JavaScript integration? → See HTML Integration section

---

## License

This pipeline is provided as-is for PCM Heat Recovery research.

