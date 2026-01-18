#!/usr/bin/env python3
"""
PCM Heat Recovery: Sensor State Management & Edge Model Integration
This module provides the interface between your HTML dashboard and the FastAPI backend.
"""

import json
import asyncio
from typing import Dict, Optional, List

# ============================================================================
# SENSOR STATE MANAGEMENT (IN-MEMORY OR DB)
# ============================================================================

class SensorState:
    """Maintain current sensor readings."""
    def __init__(self):
        self.data = {
            'Tin': 40.0,
            'Tout': 35.0,
            'mdot': 0.5,
            'dp': 5.0,
            'Tpcm_top': 50.0,
            'Tpcm_mid': 55.0,
            'Tpcm_bot': 52.0,
            'Tpcm_avg': 52.33,
            'dT': 5.0,
            'Qdot': 10.45,
            'Qpcm_signed': 8.36,
            'melt_fraction_x': 0.45,
            'Erem_kWh': 65.2,
            'plateauFlag': 0.0,
            'keff': 0.32,
            'mode': 1.0,  # 1=charge, 0=hold, -1=discharge
            'prevMode': 1.0,
            'timeInMode': 180.0,
            'valvePct': 75.0,
            'pumpPct': 85.0,
            'bypassFrac': 0.15,
            # Lags (initialize to 0 or use buffer)
            'Qdot_lag1': 10.2,
            'Qdot_lag2': 9.8,
            'Qdot_lag3': 9.5,
            'x_lag1': 0.43,
            'x_lag2': 0.41,
            'x_lag3': 0.39,
            'Tin_lag1': 39.8,
            'Tin_lag2': 39.5,
            'Tin_lag3': 39.2,
            'TpcmAvg_lag1': 52.0,
            'TpcmAvg_lag2': 51.5,
            'TpcmAvg_lag3': 51.0,
        }
        # One-hot mode (derived)
        self._update_mode_onehot()
    
    def update(self, **kwargs):
        """Update sensor readings."""
        for key, val in kwargs.items():
            if key in self.data:
                self.data[key] = val
        self._update_mode_onehot()
    
    def _update_mode_onehot(self):
        """Compute one-hot encoding from mode."""
        mode = self.data.get('mode', 0.0)
        self.data['mode_charge'] = 1.0 if mode == 1.0 else 0.0
        self.data['mode_hold'] = 1.0 if mode == 0.0 else 0.0
        self.data['mode_discharge'] = 1.0 if mode == -1.0 else 0.0
        
        prevMode = self.data.get('prevMode', 0.0)
        self.data['prevMode_charge'] = 1.0 if prevMode == 1.0 else 0.0
        self.data['prevMode_hold'] = 1.0 if prevMode == 0.0 else 0.0
        self.data['prevMode_discharge'] = 1.0 if prevMode == -1.0 else 0.0
    
    def get_features_dict(self) -> Dict:
        """Return features in correct order for API."""
        return self.data.copy()

# ============================================================================
# EDGE MODEL CLIENT
# ============================================================================

class EdgeModelClient:
    """Interface to FastAPI backend."""
    def __init__(self, base_url: str = 'http://localhost:8000'):
        self.base_url = base_url
        self.last_response = None
        self.last_error = None
    
    async def predict(self, features: Dict) -> Optional[Dict]:
        """
        Send prediction request to FastAPI backend.
        Returns parsed JSON response or None on error.
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.base_url}/predict',
                    json=features,
                    timeout=aiohttp.ClientTimeout(total=2.0)
                ) as response:
                    if response.status == 200:
                        self.last_response = await response.json()
                        self.last_error = None
                        return self.last_response
                    else:
                        self.last_error = f"HTTP {response.status}: {await response.text()}"
                        return None
        except Exception as e:
            self.last_error = str(e)
            return None

# ============================================================================
# INTEGRATION EXAMPLE FOR JAVASCRIPT
# ============================================================================

"""
=== JAVASCRIPT CODE TO ADD TO YOUR HTML ===

Replace or augment the existing edgeModelInfer() function with:

```javascript
// Global state
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
        
        // Optionally: update confidence or other displays
        console.log('Inference:', {
            yQ: result.yQ,
            yTcharge_min: result.yTcharge_min,
            x_next: result.x_next,
            Tout_next: result.Tout_next,
            confidence: result.confidence,
            latency_ms: latency,
        });
        
    } catch (error) {
        console.error('Inference error:', error);
        document.getElementById('latency').textContent = 'OFFLINE';
    }
}

// Call every second (adjust interval as needed)
setInterval(edgeModelInfer, 1000);

// When sensors update (from WebSocket, polling, or manual input):
// sensorState.Tin = newValue;
// sensorState.Tout = newValue;
// ... etc
// edgeModelInfer() will use the latest values next cycle

```

=== KEY INTEGRATION POINTS ===

1. **Sensor Update Loop**: Your existing sensor data pipeline should update `sensorState` object.
   
2. **Feature Engineering**: The server handles feature computation server-side. You only send raw sensor
   values (Tin, Tout, mdot, etc.) and basic control state (mode, valvePct, etc.).
   
3. **Lag Management**: Initialize lags to 0 or to historical values. For best results, maintain a 
   rolling buffer of past 3 timesteps and update before each inference call.
   
4. **DOM Updates**: The prediction results go directly to your Dashboard DOM elements:
   - yQ: Recoverable heat prediction [kWh/window]
   - yTcharge: Time to x >= 0.95 [minutes]
   - xNext: Melt fraction next step
   - ToutNext: Outlet temperature next step [°C]
   - qBands: P10/P50/P90 for heat recovery
   - latency: Inference time in milliseconds
   
5. **Error Handling**: If the server is down, 'latency' shows 'OFFLINE'. Check browser console for errors.

"""
