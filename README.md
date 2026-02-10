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
