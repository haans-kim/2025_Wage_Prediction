#!/usr/bin/env python3
import pickle
import pandas as pd

# master_data.pkl 로드
with open('data/master_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
print('Columns in master data:')
print(data.columns.tolist())
print()
print('wage_increase 관련 컬럼:')
for col in data.columns:
    if 'wage_increase' in col.lower() or 'mi_sbl' in col.lower():
        print(f'  - {col}')
print()
print('Data shape:', data.shape)
print('First few columns:', data.columns[:10].tolist())