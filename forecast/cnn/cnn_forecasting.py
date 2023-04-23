import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Function to create the CNN model
def create_model(filters=32, kernel_size=3, pool_size=2, dense_units=50):
    model = Sequential()
    model.add(Conv1D(filters, kernel_size, activation="relu", input_shape=(22, len(input_cols))))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# Load data
df = pd.read_csv("sensor_data.csv", parse_dates=["date"], index_col="date")
df = df.drop('Occupancy', axis=1)

# Set the attribute to forecast
attribute_to_forecast = "Temperature"
# Select CO, temperature, and humidity columns
data = df[['Temperature', 'Humidity', 'CO2']]

# Configure the number of forecast points
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

    # Perform hyperparameter optimization
    model = KerasRegressor(build_fn=create_model)
    param_grid = {
        'filters': [16, 32, 64],
        'kernel_size': [2, 3, 4],
        'pool_size': [1, 2, 3],
        'dense_units': [30, 50, 100],
        'epochs': [epoch_count]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=1)
    grid_result = grid.fit(X_train, y_train)
    print("Best parameters found: ", grid_result.best_params_)

    # Train the best CNN model and measure training time
    start_time = time.time()
    best_model = grid_result.best_estimator_.model
    training_duration = time.time() - start_time
    print(f"Training time: {training_duration} seconds")

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