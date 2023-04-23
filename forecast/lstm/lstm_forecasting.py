import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("sensor_data.csv", parse_dates=["date"], index_col="date")
df = df.drop('Occupancy', axis=1)

attribute_to_forecast = "Temperature"
# Select CO, temperature, and humidity columns
data = df[['Temperature', 'Humidity', 'CO2']]
num_forecasts = 100
epoch_count = 10

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

    # Define the LSTM model builder function
    def build_lstm_model(hp):
        model = Sequential()
        model.add(LSTM(hp.Int('units_1', min_value=30, max_value=100, step=10),
                       activation="relu", return_sequences=True, input_shape=(22, len(input_cols))))
        model.add(LSTM(hp.Int('units_2', min_value=30, max_value=100, step=10), activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    # Create the Keras Tuner and search for the best hyperparameters
    tuner = RandomSearch(
        build_lstm_model,
        objective='loss',
        max_trials=5,
        executions_per_trial=2,
        project_name='lstm_hyperparameter_tuning'
    )

    tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

    # Get the best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Train the best LSTM model and measure training time
    start_time = time.time()
    best_model.fit(X_train, y_train, epochs=epoch_count)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Generate forecast
    forecast = []
    input_seq = X_train[-1]
    for i in range(num_forecasts):
        pred = best_model.predict(input_seq.reshape(1, 22, len(input_cols)))[0, 0]
        forecast.append(pred)
        input_seq = np.vstack((input_seq[1:], data[input_cols].iloc[-num_forecasts + i].values))

    # Denormalize forecast
    forecast = scalers[attribute_to_forecast].inverse_transform(np.array(forecast).reshape(-1, 1))

    # Calculate errors

    actual_values = df[attribute_to_forecast].iloc[-num_forecasts:].values
    mse = mean_squared_error(actual_values, forecast)
    mae = mean_absolute_error(actual_values, forecast)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # Plot forecast
    plt.plot(df.index[-num_forecasts:], actual_values, label="actual")
    plt.plot(df.index[-num_forecasts:], forecast, label="forecast")
    plt.title(f'Actual vs. LSTM Forecast {attribute_to_forecast}')
    plt.ylabel(attribute_to_forecast)
    plt.xlabel('Date')
    plt.legend()
    plt.show()
else:
    print(f"Data length is less than {seq_length}. Cannot generate forecast.")
