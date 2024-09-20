
import dataframe
import matplotlib.pyplot as plt
import seaborn as sns

df = dataframe.data

plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='mean_va', data=df)
plt.title('Monthly Discharge at USGS Station')
plt.xlabel('Date')
plt.ylabel('Discharge (cfs)')
plt.grid(True)
plt.show()