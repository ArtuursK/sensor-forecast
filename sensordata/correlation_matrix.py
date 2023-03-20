import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the column names
column_names = ["date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"]

# Load the data from the CSV file into a DataFrame
file_path = "sensor_data.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path, names=column_names, header=0)

# Compute the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

