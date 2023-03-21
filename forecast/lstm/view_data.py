import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("sensor_data.csv", parse_dates=["date"], index_col="date")
df = df.drop('Occupancy', axis=1)

# Select CO, temperature, and humidity columns

# Set the attribute to forecast
attribute_to_forecast = "CO2"
data = df[[attribute_to_forecast]]

# Plot forecast
plt.plot(data)
plt.ylabel(attribute_to_forecast)
plt.xlabel('Date')
plt.legend()
plt.show()


