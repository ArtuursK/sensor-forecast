import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean
from numpy import std
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import ttest_ind

MAIN_LOCATION = "../forecast"
TEST_SET_SIZE = "5" #

evaluatedData = pd.read_csv(MAIN_LOCATION + "/data/experiment_result_data" + TEST_SET_SIZE + ".csv")
print("Dataset length: ", len(evaluatedData))

#data = evaluatedData['airtemperatureRMSE']
#data = evaluatedData['airtemperatureMAE']
#data = evaluatedData['airhumidityRMSE']
data = evaluatedData['airhumidityMAE']
print('data mean=%.3f stdv=%.3f' % (mean(data), std(data)))

plt.hist(data, color = 'blue', edgecolor = 'black')
trainTestSize = str(100-int(TEST_SET_SIZE))
plt.title(trainTestSize + "% apmācības kopai")
plt.xlabel('Kļūda')
plt.ylabel('Biežums')
plt.show()


alpha = 0.05
print('_____ Kolmogorov–Smirnov test _____')
stat, p = kstest(data, 'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
    print('Sample looks Normaly distributed (Gaussian) (failed to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


print('_____ Shapiro-Wilk test for normality _____')
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
    print('Sample looks Normaly distributed (Gaussian) (failed to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


evaluatedData_5 = pd.read_csv(MAIN_LOCATION + "/data/evaluated_result_data_5.csv")
evaluatedData_10 = pd.read_csv(MAIN_LOCATION + "/data/evaluated_result_data_10.csv")
evaluatedData_15 = pd.read_csv(MAIN_LOCATION + "/data/evaluated_result_data_15.csv")
mae_temp_data = {
    'MAE temperature 95%': evaluatedData_5['airtemperatureMAE'],
    'MAE temperature 90%': evaluatedData_10['airtemperatureMAE'],
    'MAE temperature 85%': evaluatedData_15['airtemperatureMAE']
}
rmse_temp_data = {
    'RMSE temperature 95%': evaluatedData_5['airtemperatureRMSE'],
    'RMSE temperature 90%': evaluatedData_10['airtemperatureRMSE'],
    'RMSE temperature 85%': evaluatedData_15['airtemperatureRMSE']
}
mae_humid_data = {
    'MAE humidity 95%': evaluatedData_5['airhumidityMAE'],
    'MAE humidity 90%': evaluatedData_10['airhumidityMAE'],
    'MAE humidity 85%': evaluatedData_15['airhumidityMAE']
}
rmse_humid_data = {
    'RMSE humidity 95%': evaluatedData_5['airhumidityRMSE'],
    'RMSE humidity 90%': evaluatedData_10['airhumidityRMSE'],
    'RMSE humidity 85%': evaluatedData_15['airhumidityRMSE']
}

#boxplot shows minimum, first quartile (Q1), median (Q2), third quartile (Q3) and maximum.
# fig, ax = plt.subplots()
# ax.boxplot(mae_temp_data.values())
# ax.set_xticklabels(mae_temp_data.keys())
# plt.ylabel("Kļūda")
# plt.show()


# T-Test compare
# 95% and 85% air humidity
# 90% and 85% air temperature
#H0: mean values are equal.
# If the Independent t-test results are significant (p-value very very small p<0,05)
# we can reject the null hypothesis in support of the alternative hypothesis (difference is statistically significant)

# HUMIDITY
maeTTestResult = ttest_ind(mae_humid_data['MAE humidity 95%'], mae_humid_data['MAE humidity 85%'])
rmseTTestResult = ttest_ind(rmse_humid_data['RMSE humidity 95%'], rmse_humid_data['RMSE humidity 85%'])
print(maeTTestResult)
if(maeTTestResult.pvalue < 0.05):
    print("MAE airhumidity difference is statistically significant")
else:
    print("MAE airhumidity difference is not statistically significant. Mean values are equal")
print("maeTTestResult.pvalue: ", maeTTestResult.pvalue)

if(rmseTTestResult.pvalue < 0.05):
    print("RMSE airhumidity difference is statistically significant")
else:
    print("RMSE airhumidity difference is not statistically significant. Mean values are equal")
print("rmseTTestResult.pvalue: ", rmseTTestResult.pvalue)


# TEMPERATURE
maeTTestResult = ttest_ind(mae_temp_data['MAE temperature 90%'], mae_temp_data['MAE temperature 85%'])
rmseTTestResult = ttest_ind(rmse_temp_data['RMSE temperature 90%'], rmse_temp_data['RMSE temperature 85%'])
print(maeTTestResult)
if(maeTTestResult.pvalue < 0.05):
    print("MAE airtemperature difference is statistically significant")
else:
    print("MAE airtemperature difference is not statistically significant. Mean values are equal")
print("maeTTestResult.pvalue: ", maeTTestResult.pvalue)

if(rmseTTestResult.pvalue < 0.05):
    print("RMSE airtemperature difference is statistically significant")
else:
    print("RMSE airtemperature difference is not statistically significant. Mean values are equal")
print("rmseTTestResult.pvalue: ", rmseTTestResult.pvalue)

