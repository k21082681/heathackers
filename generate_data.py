#!/usr/bin/env python3
"""
PCM Heat Recovery: Example Data Generator
Generate synthetic training data for testing the pipeline if you don't have real data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_pcm_data(n_samples=10000, sampling_rate_hz=1.0):
    """
    Generate realistic PCM system data.
    
    Args:
        n_samples: Number of timesteps
        sampling_rate_hz: Sampling frequency (1 Hz = 1 second per sample)
    
    Returns:
        pd.DataFrame with synthetic sensor data
    """
    
    # Hexacosane constants
    Tm = 56.6  # Melting temperature [°C]
    L = 166.6  # Latent heat [kJ/kg]
    Cp_s = 1.80  # Solid specific heat [kJ/kg·K]
    Cp_l = 2.37  # Liquid specific heat [kJ/kg·K]
    
    # Time array [seconds]
    dt = 1.0 / sampling_rate_hz
    time_sec = np.arange(n_samples) * dt
    
    # === INLET TEMPERATURE (sinusoidal with noise) ===
    # Cycle between 30-50°C with 2-hour period, plus random walk
    Tin = 40.0 + 10.0 * np.sin(2 * np.pi * time_sec / 7200.0)
    Tin += np.cumsum(np.random.normal(0, 0.05, n_samples)) * 0.1  # Random walk
    Tin = np.clip(Tin, 20, 60)
    
    # === MODE SEQUENCE ===
    # Cycles: charge → hold → discharge → hold → repeat
    cycle_length = 1200  # 20 minutes per cycle
    mode_phase = (time_sec % cycle_length) / cycle_length
    mode = np.zeros(n_samples)
    mode[mode_phase < 0.4] = 1.0   # Charge
    mode[(mode_phase >= 0.4) & (mode_phase < 0.5)] = 0.0   # Hold
    mode[(mode_phase >= 0.5) & (mode_phase < 0.9)] = -1.0  # Discharge
    mode[(mode_phase >= 0.9)] = 0.0  # Hold
    
    # === MASS FLOW RATE ===
    # 0 during hold, 0.3-0.6 kg/s otherwise
    mdot = np.where(mode == 0, 0.0, 0.45 + 0.1 * np.random.normal(0, 1, n_samples))
    mdot = np.clip(mdot, 0, 1.0)
    
    # === OUTLET TEMPERATURE ===
    # Depends on inlet and mode (HTF interaction with PCM)
    dT_charge = 8.0  # Inlet 8°C hotter during charge
    dT_discharge = -6.0  # Inlet 6°C cooler during discharge
    dT = np.where(mode == 1, dT_charge, np.where(mode == -1, dT_discharge, 0.0))
    dT += np.random.normal(0, 0.5, n_samples)
    Tout = Tin + dT
    
    # === PRESSURE DROP ===
    dp = 2.0 + 3.0 * np.abs(mdot) + np.random.normal(0, 0.2, n_samples)
    dp = np.clip(dp, 0, 20)
    
    # === CONTROL VARIABLES ===
    valvePct = np.where(mode == 0, 0, 50 + 50 * np.random.uniform(0, 1, n_samples))
    pumpPct = np.where(mode == 0, 0, 70 + 20 * np.random.uniform(0, 1, n_samples))
    bypassFrac = 0.1 + 0.1 * np.abs(np.sin(2 * np.pi * time_sec / 1800.0))
    
    # === PCM TEMPERATURES (phase-change dynamics) ===
    # During charge: temperatures rise toward melting point
    # During discharge: temperatures fall from melting point
    Tpcm_avg = 40.0 + 15.0 * np.sin(2 * np.pi * time_sec / 3600.0)
    Tpcm_avg += np.cumsum(np.random.normal(0, 0.02, n_samples)) * 0.05
    Tpcm_avg = np.clip(Tpcm_avg, 25, 65)
    
    # Stratification
    Tpcm_top = Tpcm_avg + 2.0 * np.random.normal(0, 1, n_samples) + 1.0
    Tpcm_mid = Tpcm_avg + 0.5 * np.random.normal(0, 1, n_samples)
    Tpcm_bot = Tpcm_avg - 1.5 * np.random.normal(0, 1, n_samples) - 1.0
    
    Tpcm_top = np.clip(Tpcm_top, 25, 65)
    Tpcm_mid = np.clip(Tpcm_mid, 25, 65)
    Tpcm_bot = np.clip(Tpcm_bot, 25, 65)
    
    # === MODE TRACKING ===
    prevMode = np.roll(mode, 1)
    prevMode[0] = mode[0]
    timeInMode = np.zeros(n_samples)
    current_mode = mode[0]
    current_time = 0
    for i in range(n_samples):
        if mode[i] == current_mode:
            current_time += dt
        else:
            current_mode = mode[i]
            current_time = dt
        timeInMode[i] = current_time
    
    # === CREATE DATAFRAME ===
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1S'),
        'Tin': Tin,
        'Tout': Tout,
        'mdot': mdot,
        'dp': dp,
        'Tpcm_top': Tpcm_top,
        'Tpcm_mid': Tpcm_mid,
        'Tpcm_bot': Tpcm_bot,
        'mode': mode,
        'prevMode': prevMode,
        'timeInMode': timeInMode,
        'valvePct': valvePct,
        'pumpPct': pumpPct,
        'bypassFrac': bypassFrac,
    })
    
    # === COMPUTE TARGETS ===
    # Qdot [kW]
    df['Qdot'] = df['mdot'] * 4.18 * (df['Tin'] - df['Tout'])  # Assume water, Cp=4.18 kJ/kg·K
    
    # yQ_kWh_next_window (sum of Qdot over next 60 seconds, convert to kWh)
    window_size = 60  # 60-second window
    df['yQ_kWh_next_window'] = df['Qdot'].rolling(window=window_size, min_periods=1).sum() / 3600.0
    df['yQ_kWh_next_window'] = df['yQ_kWh_next_window'].shift(-window_size)
    
    # x_next (melt fraction, simulated from Tpcm_avg)
    # Simple model: x = 0 if T < Tm-5, x = 1 if T > Tm+5, linear interpolation between
    Tpcm_avg_vals = df['Tpcm_mid'].values  # Use middle sensor
    x = np.where(
        Tpcm_avg_vals < Tm - 5,
        0.0,
        np.where(
            Tpcm_avg_vals > Tm + 5,
            1.0,
            (Tpcm_avg_vals - (Tm - 5)) / 10.0
        )
    )
    df['melt_fraction_x'] = np.clip(x, 0, 1)
    df['x_next'] = df['melt_fraction_x'].shift(-1)
    
    # time_to_x95_min (time to reach 95% melt)
    time_to_x95 = []
    for i in range(len(df)):
        future_x = df['melt_fraction_x'].iloc[i:min(i+window_size, len(df))].values
        if (future_x >= 0.95).any():
            idx = np.where(future_x >= 0.95)[0][0]
            time_min = (i + idx - i) / 60.0
        else:
            time_min = window_size / 60.0
        time_to_x95.append(time_min)
    df['time_to_x95_min'] = time_to_x95
    
    # Tout_next
    df['Tout_next'] = df['Tout'].shift(-1)
    
    # === DERIVED FEATURES (not strictly targets, but useful for inspection) ===
    df['Tpcm_avg'] = (df['Tpcm_top'] + df['Tpcm_mid'] + df['Tpcm_bot']) / 3.0
    df['dT'] = df['Tin'] - df['Tout']
    df['Erem_kWh'] = 0.05 * 779 * 166.6 * (1 - df['melt_fraction_x']) / 3600.0  # Assume 0.05 m³
    
    return df

def main():
    print("Generating synthetic PCM system data...")
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate data
    df = generate_synthetic_pcm_data(n_samples=10000)
    
    # Save to CSV
    csv_path = data_dir / 'pcm.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved {len(df)} rows to {csv_path}")
    
    # Print summary
    print("\nData summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df[['Tin', 'Tout', 'mdot', 'Tpcm_mid', 'melt_fraction_x', 'Qdot']].describe())
    
    print("\nReady for training! Run: python train.py")

if __name__ == '__main__':
    main()
