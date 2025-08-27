#!/usr/bin/env python3
"""
Load the proper dataset with all columns into current_data.pkl
"""

import pickle
import pandas as pd
import os
from datetime import datetime

# Read the main Excel file
excel_path = "../SambaWage_250825.xlsx"
df = pd.read_excel(excel_path)

print("=" * 80)
print("📊 LOADING PROPER DATASET")
print("=" * 80)
print(f"Source: {excel_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# The first row contains the English column names
# Use them to rename the columns
english_names = df.iloc[0].to_dict()

# Remove the first row and rename columns
df = df.iloc[1:].reset_index(drop=True)
df = df.rename(columns=english_names)

# Convert string values to numeric
for col in df.columns:
    if col != 'year':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert year to int
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

print(f"\n✨ After processing:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nTarget columns found:")
print(f"  - wage_increase_bu_sbl: {'✓' if 'wage_increase_bu_sbl' in df.columns else '✗'}")
print(f"  - wage_increase_mi_sbl: {'✓' if 'wage_increase_mi_sbl' in df.columns else '✗'}")

# Save to current_data.pkl
os.makedirs("data", exist_ok=True)
data_dict = {
    'data': df,
    'info': {
        'source': excel_path,
        'rows': len(df),
        'columns': len(df.columns),
        'has_baseup': 'wage_increase_bu_sbl' in df.columns,
        'has_performance': 'wage_increase_mi_sbl' in df.columns
    },
    'timestamp': datetime.now().isoformat()
}

with open("data/current_data.pkl", 'wb') as f:
    pickle.dump(data_dict, f)

print(f"\n✅ Data saved to data/current_data.pkl")
print(f"   - {len(df)} rows")
print(f"   - {len(df.columns)} columns")
print("=" * 80)