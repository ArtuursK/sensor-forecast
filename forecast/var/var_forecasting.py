import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time

# Function to perform the Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series)
    return result[1]

# Read the data as a DataFrame and set 'date' as the index
df = pd.read_csv("../../sensordata/sensor_data.csv", parse_dates=["date"], index_col="date")

# Keep only the relevant columns (Temperature, Humidity, CO2)
df = df[['Temperature', 'Humidity', 'CO2']]

# Create a copy of the original DataFrame for later use
df_original = df.copy()

# Check stationarity and difference the dataset if needed
differenced_columns = []
for col in df.columns:
    p_value = adf_test(df[col])
    if p_value > 0.05:
        df[col] = df[col].diff().dropna()
        differenced_columns.append(col)

# Drop the first row with NaN values due to differencing
df = df.dropna()

# Set the attribute to forecast
attribute_to_forecast = "Temperature"

# Configure the number of forecast points
num_forecasts = 100

# Split the data to forecast the last count_of_last_values_to_forecast values
train, test = df[:-num_forecasts], df[-num_forecasts:]

# Create a VAR model and fit it to the training data
model = VAR(train)
lag_order = model.select_order(maxlags=12).aic

# Start measuring the training time
start_time = time.time()

results = model.fit(maxlags=12, ic='aic')

# Calculate the training time
training_time = time.time() - start_time
print(f'Training time: {training_time} seconds')

# Forecast the test dataset
forecast = results.forecast(train.values[-lag_order:], len(test))

# Reverse the differencing operation on the forecasted values
for col in differenced_columns:
    index = df.columns.get_loc(col)
    forecast[:, index] = np.cumsum(forecast[:, index]) + df_original[col].iloc[-num_forecasts-1]

# Get the index of the attribute_to_forecast in the DataFrame
forecast_index = df.columns.get_loc(attribute_to_forecast)

# Calculate the mean squared error
mse = mean_squared_error(test[attribute_to_forecast], forecast[:, forecast_index])
print(f'Mean Squared Error: {mse}')

# Calculate the mean absolute error
mae = mean_absolute_error(test[attribute_to_forecast], forecast[:, forecast_index])
print(f'Mean Absolute Error: {mae}')

# Plot the actual vs. forecast results
plt.figure(figsize=(12, 6))
plt.plot(test.index, df_original[attribute_to_forecast][-num_forecasts:], label='Actual')
plt.plot(test.index, forecast[:, forecast_index], label='Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel(attribute_to_forecast)
plt.title(f'Actual vs. VAR Forecast {attribute_to_forecast}')
plt.legend()
plt.show()