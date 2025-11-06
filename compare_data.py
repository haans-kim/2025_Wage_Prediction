import pickle
import pandas as pd

old = pickle.load(open('c:/temp/old_data.pkl', 'rb'))
curr = pickle.load(open('backend/data/master_data.pkl', 'rb'))

old_df = old if isinstance(old, pd.DataFrame) else (old['data'] if isinstance(old, dict) and 'data' in old else pd.DataFrame())
curr_df = curr if isinstance(curr, pd.DataFrame) else (curr['data'] if isinstance(curr, dict) and 'data' in curr else pd.DataFrame())

old_cols = set(old_df.columns)
curr_cols = set(curr_df.columns)

missing_cols = old_cols - curr_cols
new_cols = curr_cols - old_cols

print('=== Missing columns (in old but not in current) ===')
for col in sorted(missing_cols):
    print(f'  - {col}')

print('\n=== New columns (in current but not in old) ===')
for col in sorted(new_cols):
    print(f'  - {col}')

print('\n=== Column count ===')
print(f'Old: {len(old_cols)} columns')
print(f'Current: {len(curr_cols)} columns')
print(f'Difference: {len(old_cols) - len(curr_cols)} columns removed')
