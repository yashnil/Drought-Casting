
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lagfeatures

df = lagfeatures.df

# Assuming df is your DataFrame with lagged features and 'SSI_12_target' as the target variable

# Step 1: Remove non-numeric columns from X
X = df.drop(columns=['SSI_12_target', 'agency_cd', 'site_no', 'parameter_cd', 'ts_id', 'year_nu', 'month_nu'])

# Target column (already correct)
y = df['SSI_12_target']

# Step 2: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardization (Scaling the features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build the MLP model
mlp = MLPRegressor(hidden_layer_sizes=(100, 50),  # Two layers with 100 and 50 neurons
                   max_iter=1000,                 # Maximum iterations for training
                   random_state=42)

# Step 5: Train the model
mlp.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_train_pred = mlp.predict(X_train_scaled)
y_test_pred = mlp.predict(X_test_scaled)

# Step 7: Evaluate the model
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print evaluation metrics
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
print(f"Train R-Squared: {train_r2}, Test R-Squared: {test_r2}")
