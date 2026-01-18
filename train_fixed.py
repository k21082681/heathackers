#!/usr/bin/env python3
"""
PCM Heat Recovery: Train XGBoost vs Random Forest Models
- Loads data/pcm.csv
- Engineers features including enthalpy observer for melt_fraction_x
- Trains 4 target models per algorithm type (8 total)
- Evaluates and selects winner
- Saves to artifacts/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Hexacosane (PCM) Physical Constants
PCM_CONSTANTS = {
    'Tm': 56.6,          # Melting temperature [°C]
    'L': 166.6,          # Latent heat [kJ/kg]
    'Cp_s': 1.80,        # Solid heat capacity [kJ/kg·K]
    'Cp_l': 2.37,        # Liquid heat capacity [kJ/kg·K]
    'rho': 779,          # Density [kg/m³]
    'k_s': 0.38,         # Solid thermal conductivity [W/m·K]
    'k_l': 0.26,         # Liquid thermal conductivity [W/m·K]
    'Tref': 25.0,        # Reference temperature [°C]
}

# PCM Volume and mass
PCM_VOLUME_M3 = 0.05     # Cubic meters (adjust to your system)
PCM_MASS_KG = PCM_VOLUME_M3 * PCM_CONSTANTS['rho']

# HTF (Heat Transfer Fluid) properties - assume water
HTF_CP = 4.18            # kJ/kg·K

# ENTHALPY OBSERVER
OBSERVER_LOSS_FRACTION = 0.02  # Assume 2% loss to ambient per timestep
OBSERVER_DT_SEC = 60           # Time step in seconds

# CSV COLUMN MAPPING
# Modify these if your CSV has different column names
COLUMN_MAPPING = {
    'timestamp': 'timestamp',      # or None if not present
    'Tin': 'Tin',                  # Inlet temperature [°C]
    'Tout': 'Tout',                # Outlet temperature [°C]
    'mdot': 'mdot',                # Mass flow rate [kg/s]
    'dp': 'dp',                    # Pressure drop [kPa]
    'Tpcm_top': 'Tpcm_top',        # PCM top temperature [°C]
    'Tpcm_mid': 'Tpcm_mid',        # PCM mid temperature [°C]
    'Tpcm_bot': 'Tpcm_bot',        # PCM bot temperature [°C]
    'mode': 'mode',                # Mode: -1=discharge, 0=hold, 1=charge
    'prevMode': 'prevMode',        # Previous mode
    'timeInMode': 'timeInMode',    # Time in current mode [s]
    'valvePct': 'valvePct',        # Valve position [%]
    'pumpPct': 'pumpPct',          # Pump speed [%]
    'bypassFrac': 'bypassFrac',    # Bypass fraction [0-1]
    # Targets (optional - will be computed if missing):
    'yQ_kWh_next_window': 'yQ_kWh_next_window',
    'time_to_x95_min': 'time_to_x95_min',
    'x_next': 'x_next',
    'Tout_next': 'Tout_next',
}

FEATURE_ORDER = [
    # Raw sensors
    'Tin', 'Tout', 'mdot', 'dp', 'Tpcm_top', 'Tpcm_mid', 'Tpcm_bot',
    # Derived
    'Tpcm_avg', 'dT', 'Qdot', 'Qpcm_signed',
    'melt_fraction_x', 'Erem_kWh', 'plateauFlag', 'keff',
    # Control + mode (one-hot later)
    'mode_charge', 'mode_discharge', 'mode_hold',
    'prevMode_charge', 'prevMode_discharge', 'prevMode_hold',
    'timeInMode', 'valvePct', 'pumpPct', 'bypassFrac',
    # Lags
    'Qdot_lag1', 'Qdot_lag2', 'Qdot_lag3',
    'x_lag1', 'x_lag2', 'x_lag3',
    'Tin_lag1', 'Tin_lag2', 'Tin_lag3',
    'TpcmAvg_lag1', 'TpcmAvg_lag2', 'TpcmAvg_lag3',
]

TARGETS = ['yQ_kWh_next_window', 'time_to_x95_min', 'x_next', 'Tout_next']

SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_get_column(df, key, mapping):
    """Safely get column from df using mapping dict."""
    col_name = mapping.get(key)
    if col_name is None or col_name not in df.columns:
        return None
    return df[col_name].copy()

def compute_enthalpy_observer(df, pcm_mass, pcm_constants, observer_loss_frac, dt_sec):
    """
    Compute melt_fraction_x using enthalpy balance observer.
    
    Energy balance:
    dH/dt = Qpcm_signed - loss
    
    H = m * [Cp_s * (T - Tref) + L*x] for T <= Tm
      = m * [Cp_l * (T - Tref) + L]   for T >= Tm
    
    Solve for x given current enthalpy and temperature.
    """
    Tm = pcm_constants['Tm']
    L = pcm_constants['L']
    Cp_s = pcm_constants['Cp_s']
    Cp_l = pcm_constants['Cp_l']
    Tref = pcm_constants['Tref']
    
    # Initialize H
    H = np.zeros(len(df))
    x = np.zeros(len(df))
    
    # Initial condition: assume x[0] from temperature
    T_avg_0 = df['Tpcm_avg'].iloc[0]
    if T_avg_0 >= Tm:
        H[0] = pcm_mass * (Cp_l * (T_avg_0 - Tref) + L)
        x[0] = 1.0
    else:
        H[0] = pcm_mass * Cp_s * (T_avg_0 - Tref)
        x[0] = 0.0
    
    # Time-step integration
    for i in range(1, len(df)):
        Qpcm = df['Qpcm_signed'].iloc[i]  # kJ/s
        loss = observer_loss_frac * H[i-1]  # Simple loss model
        dt = dt_sec / 3600.0  # Convert to hours for energy balance
        
        dH = (Qpcm - loss) * dt
        H[i] = H[i-1] + dH
        H[i] = max(0, min(H[i], pcm_mass * (Cp_l * (Tm - Tref) + L)))  # Clamp
        
        # Solve for x from H and current T_avg
        T_avg = df['Tpcm_avg'].iloc[i]
        
        if T_avg >= Tm:
            # Liquid phase
            x[i] = 1.0
        else:
            # Two-phase region: H = m * [Cp_s*(T-Tref) + L*x]
            H_sensible = pcm_mass * Cp_s * (T_avg - Tref)
            if H[i] > H_sensible:
                x[i] = (H[i] - H_sensible) / (pcm_mass * L)
                x[i] = np.clip(x[i], 0, 1)
            else:
                x[i] = 0.0
    
    return x

def engineer_features(df, pcm_mass, pcm_constants, mapping, window_sec=60):
    """
    Engineer all features for training.
    Assumes df has columns: Tin, Tout, mdot, dp, Tpcm_top/mid/bot, mode, prevMode, 
                            timeInMode, valvePct, pumpPct, bypassFrac
    """
    df = df.copy()
    
    # Extract raw columns with mapping
    for key in ['Tin', 'Tout', 'mdot', 'dp', 'Tpcm_top', 'Tpcm_mid', 'Tpcm_bot',
                'mode', 'prevMode', 'timeInMode', 'valvePct', 'pumpPct', 'bypassFrac']:
        col = safe_get_column(df, key, mapping)
        if col is not None:
            df[key] = col
    
    # === DERIVED FEATURES ===
    
    # PCM average temperature
    df['Tpcm_avg'] = (df['Tpcm_top'] + df['Tpcm_mid'] + df['Tpcm_bot']) / 3.0
    
    # Temperature difference
    df['dT'] = df['Tin'] - df['Tout']
    
    # Heat duty [kW]
    df['Qdot'] = df['mdot'] * HTF_CP * df['dT']  # kJ/s = kW
    
    # Signed PCM heat [kW] - positive = charge, negative = discharge
    df['Qpcm_signed'] = np.sign(df['mode']) * df['Qdot'] * (1.0 - df['bypassFrac'])
    
    # === ENTHALPY OBSERVER FOR MELT FRACTION ===
    df['melt_fraction_x'] = compute_enthalpy_observer(df, pcm_mass, pcm_constants, 
                                                       OBSERVER_LOSS_FRACTION, OBSERVER_DT_SEC)
    
    # Energy remaining [kWh] = m * L * (1-x) / 3600
    df['Erem_kWh'] = pcm_mass * pcm_constants['L'] * (1.0 - df['melt_fraction_x']) / 3600.0
    
    # Plateau flag: |Tpcm_avg - Tm| < eps
    eps = 2.0  # 2°C tolerance
    df['plateauFlag'] = (np.abs(df['Tpcm_avg'] - pcm_constants['Tm']) < eps).astype(float)
    
    # Effective conductivity [W/m·K]
    x = df['melt_fraction_x']
    df['keff'] = x * pcm_constants['k_l'] + (1 - x) * pcm_constants['k_s']
    
    # === ONE-HOT ENCODING FOR MODE ===
    for mode_val, mode_name in [(-1, 'discharge'), (0, 'hold'), (1, 'charge')]:
        df[f'mode_{mode_name}'] = (df['mode'] == mode_val).astype(float)
        df[f'prevMode_{mode_name}'] = (df['prevMode'] == mode_val).astype(float)
    
    # === LAGS ===
    for col in ['Qdot', 'melt_fraction_x', 'Tin']:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    df['TpcmAvg_lag1'] = df['Tpcm_avg'].shift(1)
    df['TpcmAvg_lag2'] = df['Tpcm_avg'].shift(2)
    df['TpcmAvg_lag3'] = df['Tpcm_avg'].shift(3)
    
    return df

def compute_targets_if_missing(df, window_sec=60):
    """
    Compute target variables if missing.
    Assumes 1 Hz sampling (1 second per row).
    """
    df = df.copy()
    window_rows = int(window_sec / 1.0)  # 60 rows for 60-second window
    
    # === yQ_kWh_next_window ===
    if 'yQ_kWh_next_window' not in df.columns or df['yQ_kWh_next_window'].isna().all():
        # Sum of Qdot (in kW) over next window, convert to kWh
        df['yQ_kWh_next_window'] = df['Qdot'].rolling(window=window_rows, min_periods=1).sum() / 3600.0
        df['yQ_kWh_next_window'] = df['yQ_kWh_next_window'].shift(-window_rows)  # Look-ahead
    
    # === time_to_x95_min ===
    if 'time_to_x95_min' not in df.columns or df['time_to_x95_min'].isna().all():
        # Find rows where x >= 0.95, compute time to reach
        time_to_x95 = []
        for i in range(len(df)):
            future_x = df['melt_fraction_x'].iloc[i:min(i+window_rows, len(df))]
            if (future_x >= 0.95).any():
                idx = future_x[future_x >= 0.95].index[0]
                time_min = (idx - i) / 60.0  # Convert seconds to minutes
                time_to_x95.append(time_min)
            else:
                time_to_x95.append(window_rows / 60.0)
        df['time_to_x95_min'] = time_to_x95
    
    # === x_next ===
    if 'x_next' not in df.columns or df['x_next'].isna().all():
        df['x_next'] = df['melt_fraction_x'].shift(-1)
    
    # === Tout_next ===
    if 'Tout_next' not in df.columns or df['Tout_next'].isna().all():
        df['Tout_next'] = df['Tout'].shift(-1)
    
    return df

def create_train_val_test_split(df, split_ratios):
    """Time-based split (no random shuffle)."""
    n = len(df)
    n_train = int(n * split_ratios['train'])
    n_val = int(n * split_ratios['val'])
    
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train+n_val]
    test = df.iloc[n_train+n_val:]
    
    return train, val, test

def train_model(model_class, X_train, y_train, **kwargs):
    """Train a single regressor."""
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_name):
    """Compute MAE and RMSE."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    residuals = y_test.values - y_pred
    residual_std = np.std(residuals)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'residual_std': residual_std,
        'y_pred': y_pred,
        'residuals': residuals,
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("PCM HEAT RECOVERY: XGBoost vs Random Forest Training Pipeline")
    print("=" * 80)
    
    # Create artifacts directory
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    models_dir = artifacts_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # === LOAD DATA ===
    print("\n[1] Loading data/pcm.csv...")
    if not Path('data/pcm.csv').exists():
        raise FileNotFoundError("data/pcm.csv not found. Please provide training data.")
    
    df = pd.read_csv('data/pcm.csv')
    print(f"    Loaded {len(df)} rows, {df.shape[1]} columns")
    
    # === FEATURE ENGINEERING ===
    print("\n[2] Engineering features...")
    df = engineer_features(df, PCM_MASS_KG, PCM_CONSTANTS, COLUMN_MAPPING)
    
    print("\n[3] Computing targets (if missing)...")
    df = compute_targets_if_missing(df, window_sec=60)
    
    # Drop rows with NaN AFTER ALL FEATURES AND TARGETS ARE COMPUTED
    print("\n[4] Dropping rows with NaN values...")
    initial_rows = len(df)
    df = df.dropna()
    print(f"    Dropped {initial_rows - len(df)} rows with NaN")
    print(f"    After dropping NaNs: {len(df)} rows")
    
    # === PREPARE FEATURE & TARGET MATRICES ===
    print("\n[5] Preparing feature matrices...")
    X = df[FEATURE_ORDER].copy()
    
    # Check for any remaining NaNs
    if X.isna().any().any():
        print("    WARNING: NaNs found in features, dropping...")
        valid_idx = ~X.isna().any(axis=1)
        X = X[valid_idx]
        df = df[valid_idx]
    
    print(f"    Feature matrix shape: {X.shape}")
    print(f"    Features: {len(FEATURE_ORDER)} total")
    
    # === TIME-BASED SPLIT ===
    print("\n[6] Splitting data (70% train / 15% val / 15% test)...")
    train_idx = int(len(df) * SPLIT_RATIOS['train'])
    val_idx = train_idx + int(len(df) * SPLIT_RATIOS['val'])
    
    X_train = X.iloc[:train_idx]
    X_val = X.iloc[train_idx:val_idx]
    X_test = X.iloc[val_idx:]
    
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # === TRAIN MODELS ===
    print("\n[7] Training models...")
    
    models_xgb = {}
    models_rf = {}
    metrics = {}
    residual_stds = {}
    
    for target in TARGETS:
        print(f"\n    Target: {target}")
        y_train = df[target].iloc[:train_idx]
        y_val = df[target].iloc[train_idx:val_idx]
        y_test = df[target].iloc[val_idx:]
        
        # XGBoost
        print(f"      Training XGBoost...")
        xgb_model = train_model(
            XGBRegressor,
            X_train, y_train,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )
        models_xgb[target] = xgb_model
        
        # Random Forest
        print(f"      Training Random Forest...")
        rf_model = train_model(
            RandomForestRegressor,
            X_train, y_train,
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        models_rf[target] = rf_model
        
        # Evaluate on test set
        print(f"      Evaluating on test set...")
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, target)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, target)
        
        metrics[f'xgb_{target}'] = {
            'mae': float(xgb_metrics['mae']),
            'rmse': float(xgb_metrics['rmse']),
            'residual_std': float(xgb_metrics['residual_std']),
        }
        metrics[f'rf_{target}'] = {
            'mae': float(rf_metrics['mae']),
            'rmse': float(rf_metrics['rmse']),
            'residual_std': float(rf_metrics['residual_std']),
        }
        
        residual_stds[f'xgb_{target}'] = float(xgb_metrics['residual_std'])
        residual_stds[f'rf_{target}'] = float(rf_metrics['residual_std'])
        
        print(f"        XGBoost  - MAE: {xgb_metrics['mae']:.4f}, RMSE: {xgb_metrics['rmse']:.4f}")
        print(f"        RF       - MAE: {rf_metrics['mae']:.4f}, RMSE: {rf_metrics['rmse']:.4f}")
    
    # === SELECT WINNER ===
    print("\n[8] Comparing models...")
    
    # Compute normalized score (lower is better)
    xgb_score = 0
    rf_score = 0
    
    for target in TARGETS:
        xgb_mae = metrics[f'xgb_{target}']['mae']
        rf_mae = metrics[f'rf_{target}']['mae']
        
        # Normalize by mean
        mean_mae = (xgb_mae + rf_mae) / 2
        if mean_mae > 0:
            xgb_score += xgb_mae / mean_mae
            rf_score += rf_mae / mean_mae
    
    xgb_score /= len(TARGETS)
    rf_score /= len(TARGETS)
    
    if xgb_score < rf_score:
        winner = 'xgb'
        winner_models = models_xgb
        print(f"\n    WINNER: XGBoost (normalized MAE: {xgb_score:.4f} vs {rf_score:.4f})")
    else:
        winner = 'rf'
        winner_models = models_rf
        print(f"\n    WINNER: Random Forest (normalized MAE: {rf_score:.4f} vs {xgb_score:.4f})")
    
    # === SAVE ARTIFACTS ===
    print("\n[9] Saving artifacts...")
    
    # Feature names
    with open(artifacts_dir / 'feature_names.json', 'w') as f:
        json.dump(FEATURE_ORDER, f, indent=2)
    print(f"    ✓ feature_names.json")
    
    # Metrics
    with open(artifacts_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    ✓ metrics.json")
    
    # Residual STDs
    with open(artifacts_dir / 'residual_std.json', 'w') as f:
        json.dump(residual_stds, f, indent=2)
    print(f"    ✓ residual_std.json")
    
    # Winner name
    with open(artifacts_dir / 'winner.txt', 'w') as f:
        f.write(winner)
    print(f"    ✓ winner.txt")
    
    # Save models
    for target in TARGETS:
        model_path = models_dir / f'{winner}_{target}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(winner_models[target], f)
        print(f"    ✓ {model_path.name}")
    
    print("\n" + "=" * 80)
    print("Training complete! Ready to run server.py")
    print("=" * 80)

if __name__ == '__main__':
    main()
