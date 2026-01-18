# PCM HEAT RECOVERY ML PIPELINE - FILE MANIFEST

Complete production-ready system for training XGBoost vs Random Forest models and deploying via FastAPI.

---

## 📋 Quick File Guide

### 🟢 START HERE
1. **SOLUTION_SUMMARY.md** - Executive overview (5 min read)
2. **QUICKSTART.md** - Get running in 5 minutes

### 🔴 CORE FILES (Required)
3. **train.py** - Training pipeline with model comparison
   - Feature engineering (40 features)
   - Enthalpy observer for melt_fraction_x
   - Time-based split & auto winner selection
   - Run: `python train.py`

4. **server.py** - FastAPI inference server
   - POST /predict endpoint
   - Uncertainty bands (P10/P50/P90)
   - CORS enabled for browser calls
   - Run: `python server.py`

5. **requirements.txt** - All dependencies
   - Install: `pip install -r requirements.txt`

### 🟡 DATA & INTEGRATION
6. **generate_data.py** - Create synthetic PCM data
   - 10,000 rows of realistic sensor time series
   - Run once: `python generate_data.py`

7. **JAVASCRIPT_INTEGRATION.html** - Browser integration code
   - Copy/paste into your HTML dashboard
   - sensorState object + edgeModelInfer() function
   - Updates: yQ, yTcharge, xNext, ToutNext, qBands, latency

### 📖 DOCUMENTATION
8. **README.md** - Full documentation (400 lines)
   - Setup instructions
   - Column mappings
   - API reference
   - Troubleshooting

9. **INTEGRATION_EXAMPLES.md** - 10 real-world patterns
   - WebSocket, REST polling, MQTT, SSE, database logging, etc.
   - Control system integration
   - Drift detection
   - A/B testing

### 🛠️ OPTIONAL
10. **edge_model_integration.py** - Python helper classes
    - SensorState class
    - EdgeModelClient async wrapper

---

## 📁 Directory Structure (After Running)

```
project/
├── data/
│   └── pcm.csv                      # Training data (you provide or generate)
│
├── artifacts/                       # Created by train.py
│   ├── feature_names.json           # 40 feature names in order
│   ├── metrics.json                 # MAE/RMSE per model
│   ├── residual_std.json            # Uncertainty values
│   ├── winner.txt                   # "xgb" or "rf"
│   └── models/
│       ├── {winner}_yQ_kWh_next_window.pkl
│       ├── {winner}_time_to_x95_min.pkl
│       ├── {winner}_x_next.pkl
│       └── {winner}_Tout_next.pkl
│
├── Core Files
│   ├── train.py                     # Training script
│   ├── server.py                    # FastAPI server
│   ├── generate_data.py             # Synthetic data generator
│   ├── edge_model_integration.py    # Python helpers (optional)
│   ├── JAVASCRIPT_INTEGRATION.html  # Browser code
│   └── requirements.txt             # Dependencies
│
└── Documentation
    ├── SOLUTION_SUMMARY.md          # This overview
    ├── QUICKSTART.md                # 5-minute setup
    ├── README.md                    # Full documentation
    └── INTEGRATION_EXAMPLES.md      # 10 usage patterns
```

---

## 🚀 Execution Flow

```
1. pip install -r requirements.txt
   ↓
2. python generate_data.py              (optional, if no data)
   ↓
3. python train.py                      (creates artifacts/)
   ├─ Loads data/pcm.csv
   ├─ Engineers 40 features
   ├─ Trains 8 models (4 targets × 2 algorithms)
   ├─ Compares XGBoost vs Random Forest
   └─ Saves winner + metrics
   ↓
4. python server.py                     (starts on localhost:8000)
   ├─ Loads winner models from artifacts/
   └─ Ready for /predict requests
   ↓
5. Add JAVASCRIPT_INTEGRATION.html to your web2.html
   ├─ Manages sensorState
   └─ Calls /predict every 1 second
   ↓
6. Update sensorState from your data source
   └─ WebSocket, REST API, MQTT, form input, etc.
   ↓
7. Watch predictions update in DOM: yQ, yTcharge, xNext, ToutNext, qBands, latency
```

---

## 📊 What Gets Trained

### Features (40 Total)
- **Raw sensors (7)**: Tin, Tout, mdot, dp, Tpcm_top/mid/bot
- **Derived (8)**: Tpcm_avg, dT, Qdot, Qpcm_signed, melt_fraction_x, Erem_kWh, plateauFlag, keff
- **Control (4)**: timeInMode, valvePct, pumpPct, bypassFrac
- **Modes (6)**: mode (charge/hold/discharge) + prevMode (charge/hold/discharge) one-hot
- **Lags (12)**: Qdot_lag1/2/3, x_lag1/2/3, Tin_lag1/2/3, TpcmAvg_lag1/2/3

### Targets (4)
1. **yQ_kWh_next_window** - Heat recovered in next 60-second window [kWh]
2. **time_to_x95_min** - Time to reach 95% melt [minutes]
3. **x_next** - Melt fraction at next timestep [0-1]
4. **Tout_next** - Outlet temperature at next timestep [°C]

### Models (8 Total)
- **XGBoost**: yQ, time_to_x95, x_next, Tout_next (4 models)
- **Random Forest**: yQ, time_to_x95, x_next, Tout_next (4 models)
- **Winner**: Selected by normalized MAE across all targets

---

## 🔌 API Endpoints

### POST /predict
- Input: 40-feature JSON
- Output: Predictions + P10/P50/P90 bands + confidence + latency

### GET /health
- Status check
- Returns model type, targets, feature count

### GET /config
- Fetch model configuration
- Returns feature order, residual stds

---

## 🎯 Key Metrics

| Metric | Typical Value |
|--------|---------------|
| Training time | 10-30 seconds |
| Inference latency | 5-20 milliseconds |
| Model memory | ~100 MB |
| Model size | 1-5 MB per pickle |
| Feature count | 40 |
| Target count | 4 |
| Data split | 70% train, 15% val, 15% test |

---

## 🔧 Customization Quick Links

See file for details:

| Change | File | Line |
|--------|------|------|
| PCM volume | train.py | ~30 |
| Inference frequency | JAVASCRIPT_INTEGRATION.html | ~165 |
| Model hyperparameters | train.py | ~280 |
| Sampling rate | train.py | ~65 |
| Feature selection | train.py | ~85-90 |
| Server port | server.py | bottom |
| Confidence bands | server.py | ~150+ |

---

## 📚 Documentation Map

```
Want to...                          → Read...
────────────────────────────────────────────────────────
Get started in 5 minutes            → QUICKSTART.md
Understand the whole system         → SOLUTION_SUMMARY.md
Set up data format                  → README.md (Prepare Your Data section)
Learn the API                       → README.md (API Usage section)
Add to your dashboard               → JAVASCRIPT_INTEGRATION.html
See integration patterns            → INTEGRATION_EXAMPLES.md
Troubleshoot issues                 → README.md (Troubleshooting section)
Customize features                  → train.py (comments)
Adjust model parameters             → train.py (comments)
Monitor predictions                 → INTEGRATION_EXAMPLES.md (Pattern 6)
Set up control loop                 → INTEGRATION_EXAMPLES.md (Pattern 10)
```

---

## ✅ Verification Checklist

- [ ] All files downloaded/created
- [ ] `pip install -r requirements.txt` succeeded
- [ ] `python generate_data.py` created data/pcm.csv (or you provided data)
- [ ] `python train.py` completed successfully (created artifacts/)
- [ ] `python server.py` started on localhost:8000
- [ ] Browser console shows "✓ Inference #1, #2, ..." messages
- [ ] Latency element updates every 1 second
- [ ] Predictions are reasonable numbers (not all zeros or NaN)
- [ ] JAVASCRIPT_INTEGRATION.html code added to your dashboard
- [ ] DOM elements exist: yQ, yTcharge, xNext, ToutNext, qBands, latency

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| ImportError: xgboost | `pip install -r requirements.txt` |
| data/pcm.csv not found | `python generate_data.py` |
| Server won't start | Check port 8000: `lsof -ti:8000` |
| Predictions show OFFLINE | Verify server.py is running |
| High latency (>500ms) | Check CPU usage |
| Always low confidence | Check residual_std.json values |

Full troubleshooting: See README.md

---

## 📞 Support Resources

1. **Python/ML Issues** → Check train.py comments (well documented)
2. **FastAPI Issues** → Check server.py + API reference in README.md
3. **JavaScript Issues** → Check JAVASCRIPT_INTEGRATION.html + browser F12 console
4. **Integration Help** → See INTEGRATION_EXAMPLES.md for 10 patterns
5. **Data Format** → See README.md "Prepare Your Data" section

---

## 🎓 What You're Learning

By working through this system, you'll understand:

1. **Time Series Feature Engineering**
   - Raw sensor processing
   - Domain-specific derived features (Qdot, melt_fraction_x, etc.)
   - Enthalpy balance for physics-informed models
   - Lag features for temporal dynamics
   - One-hot encoding for categorical variables

2. **Model Comparison**
   - Time-based splitting (crucial for time series!)
   - Training multiple algorithms
   - Normalized metric scoring
   - Automatic winner selection

3. **ML Deployment**
   - Production-grade API with FastAPI
   - Input validation and error handling
   - Uncertainty quantification (P10/P50/P90)
   - CORS and real-time integration

4. **Real-Time Dashboard Integration**
   - Browser JavaScript/fetch API
   - Async sensor updates
   - DOM manipulation
   - Error handling and fallbacks

---

## 🚀 Next Steps After Setup

1. **Use real data**: Replace synthetic data with your actual CSV
2. **Monitor performance**: Log predictions and compare to actuals
3. **Retrain regularly**: Weekly/monthly with new data
4. **Implement drift detection**: Alert when model accuracy drops
5. **Control integration**: Use predictions for closed-loop control
6. **Scale up**: Deploy server to cloud (AWS Lambda, Heroku, etc.)
7. **A/B test**: Compare new models before full deployment

---

## 📝 License & Citation

This pipeline was created for PCM heat recovery research and development.

**Key Technologies:**
- XGBoost (Chen & Guestrin, 2016)
- Random Forest (Breiman, 2001)
- FastAPI (Ramirez, 2019)
- scikit-learn

---

## Final Checklist for Deployment

- [ ] Trained on representative data (full charge/discharge cycles)
- [ ] Tested on held-out test set
- [ ] Metrics logged and reviewed
- [ ] Winner model selected by objective criteria
- [ ] Server running and healthy
- [ ] JavaScript integrated into dashboard
- [ ] Sensor data source connected
- [ ] DOM elements updating every 1 second
- [ ] Error handling in place (offline detection, etc.)
- [ ] Backup of trained models created
- [ ] Retraining schedule planned (weekly/monthly)
- [ ] Monitoring/alerting configured (drift detection)

**You're ready to deploy!** 🎉

---

**Questions?** See the documentation files in this package. Everything is well-commented and documented.

Happy modeling! 🚀
