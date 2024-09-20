
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the column names based on the file's metadata
columns = ["agency_cd", "site_no", "parameter_cd", "ts_id", "year_nu", "month_nu", "mean_va"]

# Load the dataset, skipping the commented lines
data = pd.read_csv('monthly.txt', sep='\t', comment='#', names=columns, skiprows=37)

# Preview the first few rows
# print(data.head())

# Ensure 'mean_va' is numeric (if there are any non-numeric entries)
# data['mean_va'] = pd.to_numeric(data['mean_va'], errors='coerce')

# Drop rows where mean_va is NaN
data = data.dropna(subset=['mean_va'])

# Create a 'date' column by combining year and month
data['date'] = pd.to_datetime(dict(year=data['year_nu'], month=data['month_nu'], day=1))