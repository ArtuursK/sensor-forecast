import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("sensor_data.csv", parse_dates=["date"], index_col="date")
df = df.drop('Occupancy', axis=1)

# Select CO, temperature, and humidity columns
data = df[['Temperature', 'Humidity', 'CO2']]
attribute_to_forecast = "Temperature"
num_forecasts = 100

# Normalize data
scalers = {}
for col in data.columns:
    scalers[col] = MinMaxScaler()
    data[col] = scalers[col].fit_transform(data[col].values.reshape(-1, 1))

# Determine the optimal lag order for the VAR model
max_lags = 15  # Adjust this value based on the size and characteristics of your dataset
model = VAR(endog=data)
results = model.select_order(max_lags)
lag_order = results.aic
print(f"Optimal lag order (AIC): {lag_order}")

# Fit the VAR model using the optimal lag order
var_model = model.fit(lag_order)

# Generate forecasts
forecast = var_model.forecast(data.values[-lag_order:], num_forecasts)

# Denormalize forecast
forecast = scalers[attribute_to_forecast].inverse_transform(forecast[:, data.columns.get_loc(attribute_to_forecast)].reshape(-1, 1))

# Calculate errors
actual_values = df[attribute_to_forecast].iloc[-num_forecasts:].values
mse = mean_squared_error(actual_values, forecast)
mae = mean_absolute_error(actual_values, forecast)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plot forecast
plt.plot(df.index[-num_forecasts:], actual_values, label="actual")
plt.plot(df.index[-num_forecasts:], forecast, label="forecast")
plt.title(f'Actual vs. VAR Forecast {attribute_to_forecast}')
plt.ylabel(attribute_to_forecast)
plt.xlabel('Date')
plt.legend()
plt.show()
