import pandas as pd

cars = pd.read_csv('./Categorical Data/data.csv')
ohe_cars = pd.get_dummies(cars[['Car']])

print(ohe_cars.to_string())