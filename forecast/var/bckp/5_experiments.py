import pandas as pd
import numpy as np
import os


MAIN_LOCATION = "../forecast"
TEST_SET_SIZE = "5" # in %
#parameters to adjust sensor data
OFFSET = 2; MAX = 1500; STEP = 50 #1500/50 = 30 experiments

rawInitialData = pd.read_csv("../../../sensordata/sensor_data_export_19_12_2021.csv",
                             parse_dates=['time'],
                             index_col='time')

# Function for evaluating the Forecasts
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)   # minmax
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})


airtemperatureMAE = []
airtemperatureRMSE = []
airhumidityMAE = []
airhumidityRMSE = []

i = 1
for x in range(0, MAX, STEP):
    print("Experiment Nr. ", i)
    if(i % 2 == 0):
        rawInitialData['airtemperature'][x] = rawInitialData['airtemperature'][x] + OFFSET
        rawInitialData['airhumidity'][x] = rawInitialData['airhumidity'][x] + OFFSET
    else:
        rawInitialData['airtemperature'][x] = rawInitialData['airtemperature'][x] - OFFSET
        rawInitialData['airhumidity'][x] = rawInitialData['airhumidity'][x] - OFFSET

    rawInitialData.to_csv(MAIN_LOCATION + "/data/raw_input_data.csv")

    # start experiments
    os.system("python " + MAIN_LOCATION + "/1_preprocessing.py")
    os.system("python " + MAIN_LOCATION + "/2_transformation.py")
    os.system("python " + MAIN_LOCATION + "/3_forecasting.py")

    forecast = pd.read_csv(MAIN_LOCATION + "/data/forecast.csv", parse_dates=['time'], index_col='time')
    originalData = pd.read_csv(MAIN_LOCATION + "/data/preprocessed_data.csv", parse_dates=['time'], index_col='time')

    accuracy_prod = forecast_accuracy(forecast['airtemperature_forecast'].values, originalData['airtemperature'].tail(len(forecast)).values)
    airtemperatureMAE.append(round(accuracy_prod['mae'], 4))
    airtemperatureRMSE.append(round(accuracy_prod['rmse'], 4))

    accuracy_prod = forecast_accuracy(forecast['airhumidity_forecast'].values, originalData['airhumidity'].tail(len(forecast)).values)
    airhumidityMAE.append(round(accuracy_prod['mae'], 4))
    airhumidityRMSE.append(round(accuracy_prod['rmse'], 4))
    i = i + 1


print("airtemperatureMAE: ", airtemperatureMAE)
print("airtemperatureRMSE: ", airtemperatureRMSE)
print("airhumidityMAE: ", airhumidityMAE)
print("airhumidityRMSE: ", airhumidityRMSE)

# save results to file
data = {
    'airtemperatureMAE': airtemperatureMAE,
    'airtemperatureRMSE': airtemperatureRMSE,
    'airhumidityMAE': airhumidityMAE,
    'airhumidityRMSE': airhumidityRMSE,
}
# Create DataFrame
df = pd.DataFrame(data)
df.to_csv(MAIN_LOCATION + "/data/experiment_result_data" + TEST_SET_SIZE + ".csv")



