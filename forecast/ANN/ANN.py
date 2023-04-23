import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(2,)))
    model.add(Dense(units=hp.Int('units_hidden', min_value=32, max_value=512, step=32),
                    activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse')
    return model

# Read data
data = pd.read_csv('sensor_data.csv', parse_dates=['date'], index_col='date')

# Set attribute_to_forecast and input_features
attribute_to_forecast = 'Temperature'
all_attributes = ['Temperature', 'Humidity', 'CO2']
input_features = [attr for attr in all_attributes if attr != attribute_to_forecast]

# Select features and target
X = data[input_features]
y = data[attribute_to_forecast]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='ann_tuning',
    project_name='ann_hp_tuning')

tuner.search_space_summary()

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

start_time = time.time()
best_ann = tuner.get_best_models(num_models=1)[0]
end_time = time.time()

# Evaluate model
y_pred = best_ann.predict(X_test)
y_pred_original_scale = scaler_y.inverse_transform(y_pred)
y_test_original_scale = scaler_y.inverse_transform(y_test)
mse = mean_squared_error(y_test_original_scale, y_pred_original_scale)
mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)

print(f"Training time: {end_time - start_time} seconds")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plot actual vs forecasted
plt.plot(y_test_original_scale, label='Actual')
plt.plot(y_pred_original_scale, label='Forecast')
plt.legend()
plt.show()
