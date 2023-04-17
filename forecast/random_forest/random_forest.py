import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('sensor_data.csv', parse_dates=['date'], index_col='date')

# Configure the attribute to forecast
attribute_to_forecast = "Temperature"

# List all attributes and remove the attribute_to_forecast
all_attributes = ['Temperature', 'Humidity', 'CO2']
input_features = [attr for attr in all_attributes if attr != attribute_to_forecast]

# Select features (based on input_features) and target (specified by attribute_to_forecast)
X = data[input_features].values
y = data[attribute_to_forecast].values

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model and measure the training time
start_time = time.time()
rf_regressor.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Calculate the mean squared error and mean absolute error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE) for {attribute_to_forecast}: {mse}")
print(f"Mean Absolute Error (MAE) for {attribute_to_forecast}: {mae}")
print(f"Training time: {training_time} seconds")

# Plot actual vs. forecasted values
plt.figure(figsize=(14, 6))
plt.plot(data.index[-len(y_test):], y_test, label="Actual")
plt.plot(data.index[-len(y_test):], y_pred, label="Forecast")
plt.title(f"{attribute_to_forecast} - Actual vs. Forecast")
plt.xlabel("Date")
plt.ylabel(attribute_to_forecast)
plt.legend()
plt.show()
