import pandas as pd

# 1. Load the data
df = pd.read_csv('detector/data/dataset.txt')

# 2. Manually list the exact columns you want
features_to_use = [
    'AccX', 'AccY', 'AccZ',
    'GyroX', 'GyroY', 'GyroZ',
    'RotX', 'RotY', 'RotZ',
    'MagX', 'MagY', 'MagZ'
]

# 3. Create a new DataFrame with ONLY those columns
df_selected = df[features_to_use]

# 4. Calculate stats
means = df_selected.mean()
stds = df_selected.std()

# 5. Combine and View
summary_df = pd.DataFrame({'Mean': means, 'Std': stds})
print(summary_df)

print({'Mean': means.to_list(), 'Std': stds.to_list()})