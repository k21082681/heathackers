🔥 HeatHackers — PCM Heat Recovery ML System
Hackathon Competition Submission

🚀 LIVE DASHBOARD
Open Dashboard Here → https://heathackers.onrender.com

📋 Overview
HeatHackers is a real-time machine learning system for optimizing Phase-Change Material (PCM) thermal energy storage. Using RandomForest models, we predict heat recovery, melt state, and outlet temperatures in 5-20ms per inference, enabling autonomous thermal control at the edge.

Competition: Energy Optimization Hackathon 2026
Team: HeatHackers
- Shahad Alhamazani  
- Najla Albassam
- Bayan Alfallty
- Maryam AlQaed
- Raneem Alolayan

  ![WhatsApp Image 2026-02-09 at 8 27 27 PM](https://github.com/user-attachments/assets/2b37051d-8248-4bb9-9b98-6606b3b2aea5)


Challenge: Reduce energy loss in thermal storage systems via predictive ML



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

## 📊 Model Performance Statistics

### **Overall Accuracy Comparison: RandomForest vs XGBoost**

| **Target** | **Metric** | **RandomForest** | **XGBoost** | **Winner** |
|:-----------|:-----------|:---------------:|:----------:|:---------:|
| **Heat Recovery (yQ)** | MAE (kWh) | **0.01341** | 0.01341 | TIE ✓ |
| | RMSE (kWh) | 0.0411 | **0.0351** | XGB |
| | Confidence | 96% | 96% | TIE ✓ |
| **Charge Time (min)** | MAE | **0.0** | 0.0 | TIE ✓ |
| | RMSE | 0.0 | 0.0 | TIE ✓ |
| | Confidence | 100% | 100% | TIE ✓ |
| **Melt Fraction (x)** | MAE | 0.00170 | **0.00165** | XGB |
| | RMSE | 0.00883 | **0.00828** | XGB |
| | Confidence | 99% | 99% | TIE ✓ |
| **Outlet Temp (Tout)** | MAE (°C) | **0.449** | 0.478 | **RF ✓** |
| | RMSE (°C) | **0.608** | 0.655 | **RF ✓** |
| | Confidence | 62% | 62% | TIE ✓ |

### **Why RandomForest Was Selected**

| Criterion | RandomForest | XGBoost | Decision |
|:----------|:-------------|:--------|:--------:|
| **Outlet Temperature Accuracy** | 0.449°C (best) | 0.478°C | **RF ✓** |
| **Inference Speed** | ~8ms | ~12ms | **RF ✓** |
| **Model Size** | 16 MB | 24 MB | **RF ✓** |
| **Overfitting Risk** | Low (robust) | High (risky) | **RF ✓** |
| **Production Readiness** | ✅ Simple, stable | ⚠️ Complex | **RF ✓** |
| **Real-Time Control** | Optimal | Suboptimal | **RF ✓** |

**Decision:** RandomForest excels in **outlet temperature prediction (critical for thermal control)**, **inference speed**, and **production robustness**. For a 36-hour hackathon requiring real-time edge ML, RF's simplicity and reliability make it the clear winner.

---



📁 File Structure & Description
text
├── index.html                      # Beautiful, responsive monitoring dashboard
│                                    # - Live KPIs, predictions, uncertainty bands
│                                    # - Drift detection, control setpoints
│                                    # - Real-time charts (120s history)
│
├── server.py                        # FastAPI backend (230 lines)
│                                    # - Loads 4 RandomForest models
│                                    # - Serves /predict endpoint (5-20ms)
│                                    # - CORS-enabled for safe cross-origin requests
│                                    # - /health, /config endpoints
│
├── data/
│   └── pcm.csv                      # Synthetic training data (10,000 rows)
│                                    # - 40 features: sensors, derived, lags, encoding
│                                    # - 4 targets: yQ, yTcharge, x_next, Tout_next
│
├── artifacts/
│   ├── models/
│   │   ├── rf_yQ_kWh_next_window.pkl
│   │   ├── rf_time_to_x95_min.pkl
│   │   ├── rf_x_next.pkl
│   │   └── rf_Tout_next.pkl
│   │
│   ├── feature_names.json           # 40 feature names (ordered)
│   ├── metrics.json                 # Mod
