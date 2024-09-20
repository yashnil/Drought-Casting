
import aggregatessi12
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

df = aggregatessi12.df

# Assuming you already have the SSI values computed and stored in 'SSI_12' column
# Example of classification function based on SSI
def classify_drought(ssi):
    if ssi > 2:
        return 'Extremely wet'
    elif 1.5 <= ssi <= 1.99:
        return 'Very wet'
    elif 1 <= ssi <= 1.49:
        return 'Moderately wet'
    elif 0 <= ssi <= 0.99:
        return 'Near normal'
    elif -1 <= ssi <= -0.99:
        return 'Moderate drought'
    elif -1.5 <= ssi <= -1.99:
        return 'Severe drought'
    elif ssi < -2:
        return 'Extreme drought'
    else:
        return np.nan  # Return NaN for missing values or undefined SSI values

# Apply the classification to SSI_12 (or SSI_24)
df['Drought Classification'] = df['SSI_12'].apply(classify_drought)

# Display the classified data
print(df[['SSI_12', 'Drought Classification']])

# Optional: You can save the results to a CSV for further analysis
df.to_csv('ssi_classified.csv', index=False)

# Define color map for drought classification
colors = {
    'Extremely wet': 'blue',
    'Very wet': 'lightblue',
    'Moderately wet': 'green',
    'Near normal': 'gray',
    'Moderate drought': 'yellow',
    'Severe drought': 'orange',
    'Extreme drought': 'red'
}

# Create a 'color' column based on drought classification
df['color'] = df['Drought Classification'].map(colors)

# Ensure no NaN values in the 'SSI_12' or 'color' columns
filtered_data = df.dropna(subset=['SSI_12', 'color'])

# Count the frequency of each drought classification
classification_counts = df['Drought Classification'].value_counts()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=classification_counts.index, y=classification_counts.values, palette=colors)

# Set plot labels and title
plt.title('Frequency of Drought Classifications (SSI 12)', fontsize=16)
plt.xlabel('Drought Classification', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Rotate x-axis labels for clarity
plt.xticks(rotation=45)

# Show the plot
plt.show()


# Create the plot
plt.figure(figsize=(12, 6))

# Scatter plot with colors based on drought classification
plt.scatter(filtered_data.index, filtered_data['SSI_12'], c=filtered_data['color'], s=10)

# Set up labels and titles
plt.title('Drought Classification Over Time (SSI 12)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('SSI 12', fontsize=14)

# Optional: Add a grid for clarity
plt.grid(True)

# Set x-axis to show major ticks by year and format the date labels
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Rotate date labels
plt.xticks(rotation=45)

# Show the plot
plt.show()
