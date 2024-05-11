import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from sklearn.linear_model import LinearRegression
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("data.csv", sep="\t")

columns_str = df.columns[0]
column_names = columns_str.split(",")[1:]  

years = [int(year) for year in column_names]
    
region_column = df.columns[0]
region_data = df[region_column].str.split(",", expand=True)

# I selected Austria country for my assigment
region = "AUSTRIA"
region_values = region_data[region_data[0].str.strip() == region].iloc[:, 1:].fillna(0).values.flatten().astype(int)

years = np.array(years).reshape(-1, 1)
    
model = LinearRegression()
region_values = region_values[:-1]

model.fit(years, region_values)

future_years = np.array([years[-1] + i for i in range(1, 4)]).reshape(-1, 1)
predicted_values = model.predict(future_years)

plt.scatter(years, region_values, color='blue', label='Original data')
plt.plot(years, model.predict(years), color='red', label='Linear regression line')
plt.scatter(future_years, predicted_values, color='green', label='Predicted values for next 3 years')
plt.xlabel('Year')
plt.ylabel('Region Data')
plt.title(f"Linear Regression Prediction for {region}")
plt.legend()

# I am using linux  pyplot show() is not supported in virtual env so i save it in image nalang.
plt.savefig('assignment.png')  
