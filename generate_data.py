"""
generate_data.py
-----------------
Run this ONCE to create a realistic synthetic cocoa_data.csv file.
It simulates 1,000 farm observations with climate and soil variables
typical of Ghana's cocoa-growing regions (Western, Ashanti, Eastern).
"""

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

n_samples = 1000

# ── Realistic ranges for Ghana's cocoa belt ──
rainfall = np.random.normal(1400, 250, n_samples)          # mm/year (1000-1800 typical)
temperature = np.random.normal(26.5, 1.5, n_samples)       # °C (24-30 typical)
solar_radiation = np.random.normal(18, 2.5, n_samples)     # MJ/m²/day
soil_ph = np.random.normal(6.0, 0.6, n_samples)            # slightly acidic soils
phosphorus = np.random.normal(12, 4, n_samples)            # mg/kg (Bray-1 P)
nitrogen = np.random.normal(0.18, 0.05, n_samples)         # % total N
potassium = np.random.normal(0.35, 0.12, n_samples)        # cmol/kg

# ── Yield model (kg/ha) with realistic interactions ──
# Base yield influenced by each factor with diminishing returns
yield_kg_ha = (
    200                                                      # base yield
    + 0.25 * rainfall                                        # more rain → more yield
    - 0.0001 * (rainfall - 1400)**2                          # too much/little rain hurts
    - 8.0 * (temperature - 26)**2                            # optimal around 26°C
    + 5.0 * solar_radiation                                  # more light helps
    - 40.0 * (soil_ph - 6.0)**2                              # optimal pH ~6.0
    + 6.0 * phosphorus                                       # P boosts yield
    + 300.0 * nitrogen                                       # N is critical
    + 80.0 * potassium                                       # K helps pod filling
    + np.random.normal(0, 35, n_samples)                     # natural noise
)

# Clip to realistic range (Ghana avg: 300-800 kg/ha, good farms: up to 1200+)
yield_kg_ha = np.clip(yield_kg_ha, 150, 1500)

# ── Build DataFrame ──
df = pd.DataFrame({
    'rainfall': np.round(rainfall, 1),
    'temperature': np.round(temperature, 1),
    'solar_radiation': np.round(solar_radiation, 1),
    'soil_ph': np.round(soil_ph, 2),
    'phosphorus': np.round(phosphorus, 1),
    'nitrogen': np.round(nitrogen, 3),
    'potassium': np.round(potassium, 3),
    'yield_kg_ha': np.round(yield_kg_ha, 1)
})

# ── Inject ~3% missing values randomly (to test imputation) ──
for col in df.columns:
    mask = np.random.random(n_samples) < 0.03
    df.loc[mask, col] = np.nan

# ── Save ──
df.to_csv('cocoa_data.csv', index=False)
print(f"✅ cocoa_data.csv created with {n_samples} rows and {df.shape[1]} columns.")
print(f"   Missing values per column:\n{df.isnull().sum()}")
print(f"\n   Yield statistics:\n{df['yield_kg_ha'].describe()}")