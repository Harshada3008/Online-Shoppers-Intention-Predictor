import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.preprocess import load_transformed_data, get_pg_engine
import numpy as np
import pandas as pd

# Load current transformed data
df = load_transformed_data()
print('Before — Revenue 0:', (df['Revenue']==0).sum(), 'Revenue 1:', (df['Revenue']==1).sum())

# If already has positive samples, do nothing
if (df['Revenue']==1).sum() > 0:
    print('Positive samples already present; no insertion needed')
    exit(0)

# Number of synthetic positives to create
n = max(200, int(0.05 * len(df)))
print(f'Generating {n} synthetic positive samples')

# Sample with replacement
samp = df.sample(n=n, replace=True, random_state=42).copy()

# Identify numeric columns excluding Revenue
num_cols = samp.select_dtypes(include=["float64", "int64"]).columns.tolist()
if 'Revenue' in num_cols:
    num_cols.remove('Revenue')

# Apply small positive bias to numeric features to simulate purchase behavior
for col in num_cols:
    # For counts, add small integers; for floats, add small positive noise
    if pd.api.types.is_integer_dtype(samp[col].dtype):
        samp[col] = (samp[col] + np.random.randint(0, 3, size=n)).astype(int)
    else:
        samp[col] = (samp[col] + np.abs(np.random.normal(loc=0.5, scale=0.3, size=n))).astype(float)

# Tweak PageValues and ProductRelated to be higher
if 'PageValues' in samp.columns:
    samp['PageValues'] = samp['PageValues'] + np.abs(np.random.normal(loc=5.0, scale=2.0, size=n))
if 'ProductRelated' in samp.columns:
    samp['ProductRelated'] = samp['ProductRelated'] + np.random.randint(1, 5, size=n)

# Set Revenue to 1
samp['Revenue'] = 1

# Ensure types match original
samp = samp[df.columns]

# Append to PostgreSQL warehouse
engine = get_pg_engine()
print('Appending synthetic samples to transformed_shoppers table...')
samp.to_sql('transformed_shoppers', engine, if_exists='append', index=False)

# Verify
new_df = load_transformed_data()
print('After — Revenue 0:', (new_df['Revenue']==0).sum(), 'Revenue 1:', (new_df['Revenue']==1).sum())
print('Done')
