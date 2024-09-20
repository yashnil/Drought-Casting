
import pandas as pd
import aggregatessi12

df = aggregatessi12.df

print(df)

# Assuming you have a DataFrame `df` with columns 'streamflow' and 'SSI_12'
# The target is SSI_12 with a 6-month lead time (t+6)

# Create lagged streamflow features (lag 1 to lag 12 for monthly data)
for i in range(1, 13):
    df[f'streamflow_12_month_lag{i}'] = df['streamflow_12_month'].shift(i)

# Create lagged SSI_12 features (lag 1 to lag 12)
for i in range(1, 13):
    df[f'SSI_12_lag{i}'] = df['SSI_12'].shift(i)

# Target: SSI_12 at t+6 (forecasting 12 months ahead with 6-month lead)
df['SSI_12_target'] = df['SSI_12'].shift(-6)

# Drop rows with missing values due to lagging
df.dropna(inplace=True)
df.to_csv('output.txt', sep='\t', index=False)

# Now you can use the lagged features and target to train the model
