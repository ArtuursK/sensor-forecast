import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

MAIN_LOCATION = "../forecast"


preprocessedData = pd.read_csv(MAIN_LOCATION + "/data/preprocessed_data.csv", parse_dates=['time'], index_col='time')


# Check if data is stationary
def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)
    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')
    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')
    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")

    # ADF Test on each column
for name, column in preprocessedData.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')



correlation_mat = preprocessedData.corr()
sns.heatmap(correlation_mat, annot = True)
#plt.show()
# do not use soil moisture as it is has low correlation with other attributes
# remove soil moisture from preprocessed csv data file
preprocessedData[['airtemperature', 'airhumidity']].to_csv(MAIN_LOCATION + "/data/preprocessed_data.csv")

# Transforming non-stationary series to make them stationary
# differencing approach

copyOfPreprocessedData = preprocessedData
copyOfPreprocessedData[['airhumidity_1d']] = preprocessedData[['airhumidity']]
copyOfPreprocessedData[['airtemperature_1d']] = preprocessedData[['airtemperature']]
copyOfPreprocessedData = copyOfPreprocessedData.diff().dropna()


plt.plot(copyOfPreprocessedData[['airtemperature_1d']].index.values, copyOfPreprocessedData[['airtemperature_1d']], label='airtemperature')
plt.plot(copyOfPreprocessedData[['airhumidity_1d']].index.values, copyOfPreprocessedData[['airhumidity_1d']], label='airhumidity')
plt.legend()
plt.grid()
#plt.show()

copyOfPreprocessedData[['airtemperature_1d', 'airhumidity_1d']].to_csv(MAIN_LOCATION + "/data/transformed_data.csv")



