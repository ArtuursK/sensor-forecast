import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../../sensordata/sensor_data.csv", parse_dates=["date"], index_col="date")
df = df.drop('Occupancy', axis=1)

# Select CO, temperature, and humidity columns
data = df[['Temperature', 'Humidity', 'CO2']]
# Set the attribute to forecast
attribute_to_forecast = "Temperature"

# Configure the number of forecast points
num_forecasts = 50

# Normalize data
scalers = {}
for col in data.columns:
    scalers[col] = MinMaxScaler()
    data[col] = scalers[col].fit_transform(data[col].values.reshape(-1, 1))

# Create input sequences and corresponding target values
seq_length = 23
forecast_col_index = data.columns.get_loc(attribute_to_forecast)
input_cols = [col for col in data.columns if col != attribute_to_forecast]

if len(data) >= seq_length:
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq_input = data[input_cols].iloc[i:i + seq_length - 1].values
        X.append(seq_input)
        y.append(data.iloc[i + seq_length - 1, forecast_col_index])

    X_train = np.array(X)
    y_train = np.array(y)

    # Define 1D CNN model
    model = Sequential()
    model.add(Conv1D(32, 3, activation="relu", input_shape=(22, len(input_cols))))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Train 1D CNN model and measure training time
    start_time = time.time()
    model.fit(X_train, y_train, epochs=10)
    training_duration = time.time() - start_time
    print(f"Training time: {training_duration} seconds")

    # Generate forecast
    forecast = []
    input_seq = X_train[-1]
    for i in range(num_forecasts):
        pred = model.predict(input_seq.reshape(1, 22, len(input_cols)))[0, 0]
        forecast.append(pred)
        input_seq = np.vstack((input_seq[1:], data[input_cols].iloc[-num_forecasts + i].values))

    # Denormalize forecast
    forecast = scalers[attribute_to_forecast].inverse_transform(np.array(forecast).reshape(-1, 1))

    # Calculate errors
    mse = mean_squared_error(df[attribute_to_forecast][-num_forecasts:], forecast)
    mae = mean_absolute_error(df[attribute_to_forecast][-num_forecasts:], forecast)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")


    # Plot forecast
    plt.plot(df.index[-num_forecasts:], df[attribute_to_forecast][-num_forecasts:], label="actual")
    plt.plot(df.index[-num_forecasts:], forecast, label="forecast")
    plt.title(f'Actual vs. CNN Forecast {attribute_to_forecast}')
    plt.ylabel(attribute_to_forecast)
    plt.xlabel('Date')
    plt.legend()
    plt.show()
else:
    print(f"Data length is less than {seq_length}. Cannot generate forecast.")
