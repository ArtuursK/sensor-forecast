import pandas as pd
import matplotlib.pyplot as plt

MAIN_LOCATION = "../forecast"

#1. Time Series Data Preprocessing
# rawData = pd.read_csv(MAIN_LOCATION + "/data/raw_input_data.csv",
#                       parse_dates=['time'],
#                       index_col='time')

rawData = pd.read_csv("../../../sensordata/sensor_data_export_19_12_2021.csv",
                      parse_dates=['time'],
                      index_col='time')

filteredRawData = rawData # copy dataframe to avoid modifying the original
#filter out rows where temperature == 0 or humidity == 0 or soilmoisture == 0
filteredRawData.drop(filteredRawData[filteredRawData['airtemperature'] <= 0].index, inplace = True)
filteredRawData.drop(filteredRawData[filteredRawData['soilmoisture'] <= 0].index, inplace = True)
filteredRawData.drop(filteredRawData[filteredRawData['airhumidity'] <= 0].index, inplace = True)

preprocessedData = filteredRawData.resample('H').mean() # aggregate average readings in each hour
print(preprocessedData.isnull().value_counts()) ## check how many hours (entries) there are with missing readings

preprocessedData.fillna(method='ffill', inplace=True) # take the previous non n.a. value and fill it forward
print('After replacing missing values: ', preprocessedData.isnull().value_counts()) # check how many hours (entries) there are with missing readings


plt.plot(preprocessedData)
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = preprocessedData[preprocessedData.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    ax.set_title(preprocessedData.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
#plt.show()


preprocessedData.to_csv(MAIN_LOCATION + "/data/preprocessed_data.csv")
