import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('sensor_data.csv', parse_dates=['date'], index_col='date')

# Set attribute_to_forecast and input_features
attribute_to_forecast = 'Temperature'
all_attributes = ['Temperature', 'Humidity', 'CO2']
input_features = [attr for attr in all_attributes if attr != attribute_to_forecast]

# Select features and target
X = data[input_features]
y = data[attribute_to_forecast]

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ANN model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=len(input_features)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train ANN model
start_time = time.time()
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
end_time = time.time()

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Training time: {end_time - start_time} seconds")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plot actual vs forecasted
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Forecast')
plt.legend()
plt.show()
