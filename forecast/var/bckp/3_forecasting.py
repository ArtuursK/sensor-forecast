
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR


import warnings
warnings.filterwarnings('ignore')

MAIN_LOCATION = "../forecast"
TEST_SET_SIZE = 5 # in % 5, 10, 15


preprocessedData = pd.read_csv(MAIN_LOCATION + "/data/transformed_data.csv", parse_dates=['time'], index_col='time')

print("Total entries: ", len(preprocessedData))


# how many entries in the test set and how many will be forecasted
nobs = round(len(preprocessedData)*(TEST_SET_SIZE*0.01))
print("Test set size: ", nobs)
# split into train and test
df_train, df_test = preprocessedData[0:-nobs], preprocessedData[-nobs:]

# Check size
print("Training data shape: ", df_train.shape)
print("Test data shape: ", df_test.shape)

### FIND BEST VAR ORDER with AIC ###
AIC = {}
best_aic, best_order = np.inf, 0
lagsToTest = 100
for i in tqdm(range(1, lagsToTest)):
    model = VAR(endog=df_train.values)
    model_result = model.fit(maxlags=i)
    AIC[i] = model_result.aic

    if AIC[i] < best_aic:
        best_aic = AIC[i]
        best_order = i

print('BEST ORDER:', best_order, 'BEST AIC:', best_aic)

plt.figure(figsize=(14,5))
plt.plot(range(len(AIC)), list(AIC.values()))
plt.plot([best_order-1], [best_aic], marker='o', markersize=8, color="red")
plt.xticks(range(0, len(AIC), 2), range(1, lagsToTest, 2), rotation=90)
plt.xlabel('Modeļa kārta'); plt.ylabel('Kritērija AIC vērtība')
np.set_printoptions(False)
#plt.show()


### FIT FINAL VAR WITH LAG CORRESPONTING TO THE BEST AIC ###
var = VAR(endog=df_train.values)
var_result = var.fit(maxlags=best_order)
print("var_result.aic: ", var_result.aic)

print(len(df_train.values[-best_order:]))

forecast_input = df_train.values[-best_order:]

# Forecast
fc = var_result.forecast(y = forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=preprocessedData.index[-nobs:], columns=preprocessedData.columns)
#print(df_forecast)


print("Transforming forecast data back from differenced data")
originalData = pd.read_csv(MAIN_LOCATION + "/data/preprocessed_data.csv", parse_dates=['time'], index_col='time')

def invert_transformation(df_train, df_forecast):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(originalData, df_forecast)
print("Saving data")
df_results.to_csv(MAIN_LOCATION + "/data/forecast.csv")

