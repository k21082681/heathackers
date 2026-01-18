# Usage Examples: Real-World Integration Patterns

This document shows common ways to integrate the edge model into your system.

---

## Pattern 1: WebSocket Real-Time Stream (e.g., from IoT device)

```javascript
// In your HTML/dashboard JavaScript

const ws = new WebSocket('ws://your-device-ip:8080/stream');

ws.onmessage = function(event) {
    const sensorData = JSON.parse(event.data);
    
    // Update sensor state from WebSocket message
    updateSensors({
        Tin: sensorData.inlet_temp,
        Tout: sensorData.outlet_temp,
        mdot: sensorData.mass_flow,
        dp: sensorData.pressure_drop,
        Tpcm_top: sensorData.pcm_top,
        Tpcm_mid: sensorData.pcm_mid,
        Tpcm_bot: sensorData.pcm_bot,
        mode: sensorData.mode,  // 1, 0, or -1
        prevMode: sensorData.prev_mode,
        timeInMode: sensorData.time_in_mode,
        valvePct: sensorData.valve_pct,
        pumpPct: sensorData.pump_pct,
        bypassFrac: sensorData.bypass_frac,
        melt_fraction_x: sensorData.estimated_x,
        Erem_kWh: sensorData.energy_remaining,
        plateauFlag: sensorData.is_melting ? 1.0 : 0.0,
        keff: sensorData.effective_k,
    });
    
    // edgeModelInfer() will use these new values next iteration
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};
```

---

## Pattern 2: REST API Polling (Every 1 second)

```javascript
// Polling alternative if WebSocket not available

async function pollSensors() {
    try {
        const response = await fetch('http://your-device-ip:5000/api/sensors', {
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        updateSensors({
            Tin: data.Tin,
            Tout: data.Tout,
            mdot: data.mdot,
            // ... all other fields
        });
        
    } catch (error) {
        console.error('Sensor polling error:', error);
    }
}

// Poll every 500ms
setInterval(pollSensors, 500);
```

---

## Pattern 3: HTML Form Input (Manual testing)

```html
<!-- For testing without real sensors -->

<form id="sensor-form">
    <fieldset>
        <legend>Manual Sensor Input</legend>
        
        <label>Tin [°C]
            <input type="number" id="input-Tin" value="40" step="0.1" />
        </label>
        
        <label>Tout [°C]
            <input type="number" id="input-Tout" value="35" step="0.1" />
        </label>
        
        <label>mdot [kg/s]
            <input type="number" id="input-mdot" value="0.5" step="0.01" />
        </label>
        
        <label>Tpcm_mid [°C]
            <input type="number" id="input-Tpcm_mid" value="55" step="0.1" />
        </label>
        
        <label>Mode
            <select id="input-mode">
                <option value="-1">Discharge (-1)</option>
                <option value="0">Hold (0)</option>
                <option value="1" selected>Charge (1)</option>
            </select>
        </label>
        
        <button type="button" onclick="updateFromForm()">Update Sensors</button>
    </fieldset>
</form>

<script>
    function updateFromForm() {
        updateSensors({
            Tin: parseFloat(document.getElementById('input-Tin').value),
            Tout: parseFloat(document.getElementById('input-Tout').value),
            mdot: parseFloat(document.getElementById('input-mdot').value),
            Tpcm_mid: parseFloat(document.getElementById('input-Tpcm_mid').value),
            mode: parseFloat(document.getElementById('input-mode').value),
        });
        console.log('Sensors updated from form');
    }
</script>
```

---

## Pattern 4: MQTT Broker Integration

```javascript
// Using Paho MQTT library (include in HTML)
// <script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js"></script>

const mqtt = {
    client: null,
    
    connect: function(brokerUrl, clientId) {
        this.client = new Paho.MQTT.Client(brokerUrl, 8001, clientId);
        
        this.client.onMessageArrived = (message) => {
            const sensorData = JSON.parse(message.payloadString);
            updateSensors(sensorData);
        };
        
        this.client.onConnectionLost = (error) => {
            console.error('MQTT connection lost:', error);
        };
        
        this.client.connect({
            onSuccess: () => {
                console.log('MQTT connected');
                // Subscribe to sensor topic
                this.client.subscribe('pcm/sensors');
            }
        });
    }
};

mqtt.connect('mqtt.example.com', 'pcm-dashboard-client');
```

---

## Pattern 5: Python Backend with Server-Sent Events (SSE)

```html
<!-- Client-side: Listen to server-sent events -->

<script>
    const eventSource = new EventSource('http://your-backend:5000/stream-sensors');
    
    eventSource.onmessage = function(event) {
        const sensorData = JSON.parse(event.data);
        updateSensors(sensorData);
    };
    
    eventSource.onerror = function(error) {
        console.error('SSE error:', error);
    };
</script>
```

```python
# Server-side (Python Flask/FastAPI)
from flask import Flask, Response
import json
import time

app = Flask(__name__)

@app.route('/stream-sensors')
def stream_sensors():
    def generate():
        while True:
            # Read from your sensor API/database
            sensor_data = {
                'Tin': 40.5,
                'Tout': 35.2,
                'mdot': 0.45,
                # ... all fields
            }
            
            # Send as server-sent event
            yield f'data: {json.dumps(sensor_data)}\n\n'
            time.sleep(0.5)  # Every 500ms
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=5000)
```

---

## Pattern 6: Logging Predictions to Database

```javascript
// Store predictions in database for analysis

let predictionHistory = [];

async function edgeModelInfer() {
    // ... existing inference code ...
    
    const result = await response.json();
    const latency = performance.now() - t0;
    
    // Create record
    const prediction = {
        timestamp: new Date().toISOString(),
        yQ: result.yQ,
        yTcharge_min: result.yTcharge_min,
        x_next: result.x_next,
        Tout_next: result.Tout_next,
        confidence: result.confidence,
        latency_ms: latency,
        sensor_state: { ...sensorState }  // For debugging
    };
    
    // Store locally
    predictionHistory.push(prediction);
    
    // Send to backend (optional)
    if (predictionHistory.length >= 60) {  // Every minute
        logPredictions(predictionHistory);
        predictionHistory = [];
    }
}

async function logPredictions(batch) {
    try {
        const response = await fetch('http://your-backend:5000/api/predictions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ predictions: batch })
        });
        
        if (response.ok) {
            console.log(`Logged ${batch.length} predictions`);
        }
    } catch (error) {
        console.error('Failed to log predictions:', error);
    }
}
```

---

## Pattern 7: Model Retraining Pipeline

```bash
#!/bin/bash
# Automatic daily retraining

# 1. Collect data from last 24 hours
python /path/to/export_data.py --hours 24 --output data/pcm_daily.csv

# 2. Append to historical dataset
cat data/pcm_daily.csv >> data/pcm_historical.csv

# 3. Retrain models
python /path/to/train.py

# 4. Compare metrics
python << 'EOF'
import json

with open('artifacts/metrics.json') as f:
    metrics = json.load(f)

avg_mae = sum(m['mae'] for m in metrics.values()) / len(metrics)
print(f"New model avg MAE: {avg_mae:.4f}")

# Check if significantly better than previous...
EOF

# 5. Optionally backup old models
cp -r artifacts artifacts_backup_$(date +%Y%m%d)

# 6. Restart server to load new models
# pkill -f server.py
# sleep 2
# python server.py &
```

---

## Pattern 8: A/B Testing Old vs New Model

```python
# server.py modification for A/B testing

import random

MODELS_V1 = {}  # Old models
MODELS_V2 = {}  # New models

def predict_with_ab_test(features):
    """Randomly split traffic between old and new models"""
    if random.random() < 0.5:
        version = 'v1'
        models = MODELS_V1
    else:
        version = 'v2'
        models = MODELS_V2
    
    predictions = {}
    for target in TARGETS:
        predictions[target] = float(models[target].predict(features)[0])
    
    return predictions, version

# In /predict endpoint:
# predictions, version = predict_with_ab_test(X)
# return {..., "model_version": version}
```

---

## Pattern 9: Anomaly Detection on Residuals

```javascript
// Detect when predictions are far from actual (drift detection)

let residualBuffer = [];
const RESIDUAL_WINDOW = 60;  // Last 60 predictions

async function edgeModelInfer() {
    // ... existing code ...
    
    // Store actual outcome (when available)
    const actual = getActualOutcome();  // Your function
    const residual = actual - result.yQ;
    
    residualBuffer.push(residual);
    if (residualBuffer.length > RESIDUAL_WINDOW) {
        residualBuffer.shift();
    }
    
    // Calculate drift
    const mean = residualBuffer.reduce((a, b) => a + b, 0) / residualBuffer.length;
    const std = Math.sqrt(
        residualBuffer.reduce((sum, x) => sum + (x - mean) ** 2, 0) / residualBuffer.length
    );
    
    // Flag if drift detected
    if (Math.abs(mean) > 2 * std) {
        console.warn('⚠️ DRIFT DETECTED: Model may need retraining');
        // alert('Model drift detected - consider retraining');
    }
}
```

---

## Pattern 10: Control System Integration

```javascript
// Use model predictions to set control setpoints

async function edgeModelInfer() {
    const result = await response.json();
    
    // === CONTROL LOGIC ===
    
    // If heat recovery drops below target, increase charge time
    if (result.yQ < 0.2) {  // Target: 0.2 kWh per window
        setControlSetpoint({
            mode: 1.0,          // Charge
            valvePct: 90,       // Increase valve
            pumpPct: 100,       // Max pump
            bypassFrac: 0.05,   // Minimize bypass
        });
        console.log('⬆️ Increased charging');
    }
    
    // If almost fully charged, reduce pump speed
    else if (result.x_next > 0.90) {
        setControlSetpoint({
            mode: 0.0,          // Hold
            valvePct: 0,        // Close valve
            pumpPct: 10,        // Minimal pump
            bypassFrac: 1.0,    // Bypass all flow
        });
        console.log('⬇️ Charge complete - holding');
    }
    
    // Normal operation: maintain setpoints
    else {
        console.log('→ Maintaining current setpoints');
    }
}

function setControlSetpoint(setpoints) {
    // Send to PLC/controller
    fetch('http://your-controller:8080/setpoints', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(setpoints)
    }).catch(err => console.error('Setpoint error:', err));
}
```

---

## Summary: Choose Your Pattern

| Pattern | Use Case | Latency |
|---------|----------|---------|
| WebSocket | Real-time streams | <100ms |
| REST polling | Medium-frequency | 1-5s |
| HTML form | Testing/debugging | Manual |
| MQTT | IoT devices | <1s |
| SSE | Web-friendly streaming | <1s |
| Database logging | Historical analysis | Async |
| Retraining | Model updates | Daily |
| A/B testing | Validation | Runtime |
| Drift detection | Model monitoring | Runtime |
| Control integration | Closed-loop automation | Real-time |

Pick one or combine multiple patterns based on your system architecture!

---

## Next: Integration Steps

1. Choose appropriate pattern(s) for your system
2. Update `updateSensors()` calls in JAVASCRIPT_INTEGRATION.html
3. Test with your actual sensor data
4. Log predictions to monitor model performance
5. Plan periodic retraining (weekly/monthly)
6. Consider A/B testing before full deployment
7. Implement drift detection for alerting

Good luck! 🚀
