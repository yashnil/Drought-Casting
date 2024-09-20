
import dataframe
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

df = dataframe.data

# Assuming you have a 'date' column and 'mean_va' for monthly streamflow
df['date'] = pd.to_datetime(dict(year=df['year_nu'], month=df['month_nu'], day=1))

# Set 'date' as index
df.set_index('date', inplace=True)

# 12-month rolling sum for streamflow (SSI 12)
df['streamflow_12_month'] = df['mean_va'].rolling(window=12).sum()

# 24-month rolling sum for streamflow (SSI 24)
df['streamflow_24_month'] = df['mean_va'].rolling(window=24).sum()

# Fit Gamma distribution to the 12-month streamflow data
gamma_params = stats.gamma.fit(df['streamflow_24_month'].dropna())

# Use the fitted distribution to compute cumulative probabilities
cdf_24 = stats.gamma.cdf(df['streamflow_24_month'].dropna(), *gamma_params)

# Transform cumulative probabilities into the standard normal distribution (SSI 12)
ssi_24 = stats.norm.ppf(cdf_24)

# Store SSI 12 in the data frame
df.loc[df['streamflow_24_month'].notna(), 'SSI_24'] = ssi_24

# Plotting SSI 12
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['SSI_24'], label='SSI 24')
plt.title('Standardized Streamflow Index (SSI 24)')
plt.xlabel('Date')
plt.ylabel('SSI 24')
plt.axhline(0, color='red', linestyle='--')  # Drought threshold
plt.grid(True)
plt.legend()
plt.show()

