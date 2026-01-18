# PCM Edge ML Pipeline: Complete Runnable System

## What You Have

A **production-ready machine learning pipeline** for PCM (Phase Change Material) heat recovery systems that:

1. ✅ **Trains** XGBoost and Random Forest models on your sensor data
2. ✅ **Compares** performance automatically and selects the winner
3. ✅ **Deploys** via FastAPI for real-time inference (1-second updates)
4. ✅ **Predicts** 4 targets: heat recovery, charge time, melt fraction, outlet temp
5. ✅ **Quantifies** uncertainty using P10/P50/P90 confidence bands
6. ✅ **Integrates** directly into your HTML dashboard via JavaScript

---

## 5-Minute Quick Start

### Step 1: Install (2 minutes)
```bash
pip install -r requirements.txt
```

### Step 2: Get Data (1 minute)
Either:
- **Option A**: Use your own `data/pcm.csv` (see README.md for format)
- **Option B**: Generate synthetic data for testing:
  ```bash
  python generate_data.py  # Creates data/pcm.csv automatically
  ```

### Step 3: Train (1 minute)
```bash
python train.py
```
Output: `artifacts/` folder with trained models + metrics

### Step 4: Run Server (1 minute)
```bash
python server.py
```
Server starts on `http://localhost:8000`

### Step 5: Connect Dashboard (Add to your HTML `<head>` or before `</body>`)
```html
<script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
<script>
const sensorState = {
    Tin: 40.0, Tout: 35.0, mdot: 0.5, dp: 5.0,
    Tpcm_top: 50.0, Tpcm_mid: 55.0, Tpcm_bot: 52.0,
    Tpcm_avg: 52.33, dT: 5.0, Qdot: 10.45, Qpcm_signed: 8.36,
    melt_fraction_x: 0.45, Erem_kWh: 65.2, plateauFlag: 0.0, keff: 0.32,
    mode: 1.0, prevMode: 1.0, timeInMode: 180.0,
    valvePct: 75.0, pumpPct: 85.0, bypassFrac: 0.15,
    Qdot_lag1: 10.2, Qdot_lag2: 9.8, Qdot_lag3: 9.5,
    x_lag1: 0.43, x_lag2: 0.41, x_lag3: 0.39,
    Tin_lag1: 39.8, Tin_lag2: 39.5, Tin_lag3: 39.2,
    TpcmAvg_lag1: 52.0, TpcmAvg_lag2: 51.5, TpcmAvg_lag3: 51.0,
};

function updateModeOneHot() {
    sensorState.mode_charge = sensorState.mode === 1.0 ? 1.0 : 0.0;
    sensorState.mode_hold = sensorState.mode === 0.0 ? 1.0 : 0.0;
    sensorState.mode_discharge = sensorState.mode === -1.0 ? 1.0 : 0.0;
    sensorState.prevMode_charge = sensorState.prevMode === 1.0 ? 1.0 : 0.0;
    sensorState.prevMode_hold = sensorState.prevMode === 0.0 ? 1.0 : 0.0;
    sensorState.prevMode_discharge = sensorState.prevMode === -1.0 ? 1.0 : 0.0;
}

async function edgeModelInfer() {
    updateModeOneHot();
    const t0 = performance.now();
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sensorState),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        document.getElementById('yQ').textContent = result.yQ.toFixed(2);
        document.getElementById('yTcharge').textContent = result.yTcharge_min.toFixed(1);
        document.getElementById('xNext').textContent = result.x_next.toFixed(3);
        document.getElementById('ToutNext').textContent = result.Tout_next.toFixed(1);
        document.getElementById('qBands').textContent = 
            `P10: ${result.qBands.p10.toFixed(2)} | P50: ${result.qBands.p50.toFixed(2)} | P90: ${result.qBands.p90.toFixed(2)}`;
        document.getElementById('latency').textContent = (performance.now() - t0).toFixed(1) + ' ms';
    } catch (e) {
        console.error('Inference error:', e);
        document.getElementById('latency').textContent = 'OFFLINE';
    }
}
setInterval(edgeModelInfer, 1000);
</script>
```

**Ensure your HTML has these DOM IDs:**
```html
<span id="yQ">-</span>
<span id="yTcharge">-</span>
<span id="xNext">-</span>
<span id="ToutNext">-</span>
<span id="qBands">-</span>
<span id="latency">-</span>
```

---

## Files Provided

| File | Purpose |
|------|---------|
| `train.py` | Feature engineering + XGBoost/RF training + auto-selection |
| `server.py` | FastAPI inference server with /predict endpoint |
| `generate_data.py` | Create synthetic PCM data for testing |
| `edge_model_integration.py` | Python classes for sensor state management |
| `requirements.txt` | All pip dependencies |
| `README.md` | Full documentation |
| `QUICKSTART.md` | This file |

---

## How It Works (High-Level)

```
┌─────────────────┐
│  Sensor Data    │
│  (1 Hz stream)  │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Feature Engineering (train.py)                              │
│  - Raw features (Tin, Tout, mdot, Tpcm_*, mode, control)    │
│  - Derived (Tpcm_avg, dT, Qdot, Qpcm_signed)                │
│  - Enthalpy Observer: compute melt_fraction_x                │
│  - Computed: Erem_kWh, plateauFlag, keff                    │
│  - One-hot: mode (charge/hold/discharge)                    │
│  - Lags: Qdot_lag1/2/3, x_lag1/2/3, Tin_lag1/2/3, etc.      │
│  Result: 40 features per timestep                            │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Training (train.py)                                         │
│  - Time-based split: 70% train, 15% val, 15% test           │
│  - 4 targets × 2 algorithms = 8 models total                │
│  ├─ XGBoost: yQ, time_to_x95, x_next, Tout_next            │
│  └─ Random Forest: yQ, time_to_x95, x_next, Tout_next      │
│  - Evaluate on test set: MAE, RMSE per target               │
│  - Pick winner (XGBoost or RF) by normalized score          │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Artifacts (artifacts/)                                      │
│  ├─ feature_names.json (40 features in order)               │
│  ├─ metrics.json (MAE/RMSE for all models)                  │
│  ├─ residual_std.json (uncertainty quantification)          │
│  ├─ winner.txt (xgb or rf)                                  │
│  └─ models/                                                  │
│      ├─ {winner}_yQ_kWh_next_window.pkl                     │
│      ├─ {winner}_time_to_x95_min.pkl                        │
│      ├─ {winner}_x_next.pkl                                 │
│      └─ {winner}_Tout_next.pkl                              │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI Server (server.py)                                  │
│  POST /predict (JSON with 40 features)                       │
│  ├─ Load models from artifacts/                             │
│  ├─ Reorder features per feature_names.json                 │
│  ├─ Run 4 inference passes (one per target)                 │
│  ├─ Compute P10/P50/P90 using residual_std                  │
│  ├─ Calculate confidence = 1/(1+avg_std)                    │
│  └─ Return JSON:                                             │
│      {                                                        │
│        yQ, yTcharge_min, x_next, Tout_next,                │
│        qBands: {p10, p50, p90},                             │
│        chargeBands: {p10, p50, p90},                        │
│        confidence, model_type, latency_ms                   │
│      }                                                        │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  HTML Dashboard (your web page)                              │
│  JavaScript fetch() calls /predict every 1 second            │
│  Updates DOM elements:                                       │
│  - yQ: recoverable heat [kWh/window]                        │
│  - yTcharge: time to x >= 0.95 [min]                        │
│  - xNext: melt fraction next step                           │
│  - ToutNext: outlet temp next step [°C]                     │
│  - qBands: uncertainty bands                                 │
│  - latency: inference time [ms]                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Why 40 Features?
- **7 raw sensors** (Tin, Tout, mdot, dp, Tpcm_top/mid/bot)
- **7 derived** (Tpcm_avg, dT, Qdot, Qpcm_signed, melt_fraction_x, Erem_kWh, plateauFlag, keff)
- **3 one-hot modes** (charge/hold/discharge)
- **3 one-hot prev modes** (charge/hold/discharge)
- **4 control** (timeInMode, valvePct, pumpPct, bypassFrac)
- **12 lags** (Qdot_1/2/3, x_1/2/3, Tin_1/2/3, TpcmAvg_1/2/3)

### Why Enthalpy Observer?
- **Physical model** to compute melt_fraction_x from temperatures
- **Integrates** Qpcm_signed accounting for losses
- **Clamps** to [0, 1] for physical validity
- **More robust** than direct temperature mapping

### Why Two Models?
- **XGBoost**: Typically better for structured data, faster inference
- **Random Forest**: More interpretable, sometimes better on noisy data
- **Automatic winner selection** by normalized MAE across all targets

### Why P10/P50/P90?
- **Quantifies uncertainty** around predictions
- Uses **residual standard deviation** from training
- Assumes **Gaussian residuals** (reasonable for most regressors)
- Helps **control systems** know when to trust the prediction

---

## What Happens In Training (train.py)

1. **Load CSV** → 10,000 rows of sensor data
2. **Engineer features** → From 14 columns to 40 features
   - Compute Qdot = mdot × Cp × ΔT
   - Enthalpy balance: ∫(Qpcm_signed - losses) → melt_fraction_x
   - Erem = m × L × (1 - x) / 3600 [kWh]
   - One-hot encoding for mode
   - Lag features (t-1, t-2, t-3)
3. **Drop NaNs** → From lags and look-ahead
4. **Time-based split**:
   - Training: rows 0–6,958
   - Validation: rows 6,958–8,449
   - Test: rows 8,449–9,940
5. **Train 8 models** (4 targets × 2 algorithms):
   - XGBoost with n_estimators=100, max_depth=6
   - Random Forest with n_estimators=100, max_depth=10
6. **Evaluate on test set**:
   - Compute MAE and RMSE per target per model
   - Track residual standard deviation
7. **Select winner**:
   - Normalize MAE by target
   - Average across targets
   - Pick XGBoost or RF (whichever is lower)
8. **Save artifacts**:
   - 4 pickle files for winner
   - feature_names.json (order matters!)
   - metrics.json (for logging)
   - residual_std.json (for confidence bands)

---

## What Happens In Inference (server.py)

1. **POST /predict** with JSON containing 40 features
2. **Reorder features** using feature_names.json
3. **Run 4 predictions** (one per target model)
4. **Compute uncertainty bands**:
   ```
   p10 = prediction - 1.28 × residual_std  (~10th percentile)
   p50 = prediction                         (median)
   p90 = prediction + 1.28 × residual_std  (~90th percentile)
   ```
5. **Calculate confidence**: 1 / (1 + average_residual_std)
6. **Return JSON** with:
   - Point predictions (p50)
   - Confidence bands (p10/p50/p90)
   - Inference latency
   - Model type used

---

## Integration Checklist

- [ ] Install requirements.txt
- [ ] Prepare data/pcm.csv or run generate_data.py
- [ ] Run train.py (creates artifacts/)
- [ ] Run server.py (starts on localhost:8000)
- [ ] Add JavaScript to your HTML
- [ ] Verify DOM IDs exist (yQ, yTcharge, xNext, ToutNext, qBands, latency)
- [ ] Test: Open browser console (F12) and check for errors
- [ ] Check latency element updates every second
- [ ] Celebrate! 🎉

---

## Troubleshooting

### "ImportError: No module named 'xgboost'"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: data/pcm.csv"
```bash
python generate_data.py
```

### Server won't start ("Address already in use")
```bash
# Kill existing process on port 8000
lsof -ti:8000 | xargs kill -9
python server.py
```

### Predictions not updating on dashboard
1. Open browser F12 Console
2. Check for network errors (red 🔴)
3. Verify server is running: `curl http://localhost:8000/health`
4. Check that sensorState values are realistic (not all zeros)

### High latency (>100 ms)
- Your computer is busy. Model inference should be <20 ms.
- Check CPU usage while running
- Reduce model complexity: edit train.py max_depth, n_estimators

### Confidence is always low
- Check residual_std.json values
- High std = hard prediction problem
- Consider more/better features or more training data

---

## Customization

### Different sampling rate?
Edit `train.py`:
```python
OBSERVER_DT_SEC = 30  # Instead of 60 if you sample faster
```

### Different PCM volume?
Edit `train.py`:
```python
PCM_VOLUME_M3 = 0.10  # Your system volume in m³
```

### Different targets?
Edit `train.py`:
```python
TARGETS = ['yQ_kWh_next_window', 'time_to_x95_min']  # Remove x_next, Tout_next if you don't need them
```

### Better hyperparameters?
Edit `train.py` around line 280:
```python
xgb_model = train_model(
    XGBRegressor,
    X_train, y_train,
    n_estimators=200,      # More trees
    max_depth=8,           # Deeper trees
    learning_rate=0.05,    # Slower learning, better generalization
    random_state=42,
)
```

---

## Performance Notes

- **Training time**: ~10-30 seconds on typical laptop (10K rows)
- **Inference time**: ~5-20 ms per prediction
- **Memory**: ~100 MB for all 4 models in memory
- **Model size**: Each .pkl is ~1-5 MB

---

## Next Steps

1. **Monitor drift**: Save predictions + actuals, check residuals over time
2. **Retrain monthly**: Collect new data, run train.py again, replace models
3. **A/B test**: Keep old models, compare new vs old on same test data
4. **Feature importance**: Use xgb_model.feature_importances_ to understand what matters
5. **Production deployment**: Move server.py to cloud (AWS Lambda, Heroku, etc.)

---

## Questions?

See README.md for full documentation, API examples, and troubleshooting.

**You're ready to deploy!** 🚀
