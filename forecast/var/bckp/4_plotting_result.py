import matplotlib.pyplot as plt
import pandas as pd

MAIN_LOCATION = "../forecast"

forecast = pd.read_csv(MAIN_LOCATION + "/data/forecast.csv", parse_dates=['time'], index_col='time')
originalData = pd.read_csv(MAIN_LOCATION + "/data/preprocessed_data.csv", parse_dates=['time'], index_col='time')

plt.plot(forecast.index.values, forecast['airtemperature_forecast'], label = 'Prognoze')
plt.plot(originalData.index.values, originalData['airtemperature'], label = 'Patiesie rādījumi')
# plt.plot(forecast.index.values, forecast['airhumidity_forecast'], label = 'Prognoze')
# plt.plot(originalData.index.values, originalData['airhumidity'], label = 'Patiesie rādījumi')

plt.xlabel('Laiks (Intervāls starp rādījumiem: 1 stunda)')
plt.ylabel('Rādījums')

plt.legend()
plt.grid()
plt.show()
