import tensorflow as tf
import pandas as pd
import math

data = pd.read_table(
  'auto-mpg.data',
  delim_whitespace = True,
  na_values = '?',
  header = None,
  names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
)

data = data.dropna()
data = data.drop(axis = 'columns', labels = ['name'])

data = pd.get_dummies(data, columns = ['year', 'origin', 'cylinders'])

data = data.sample(frac = 1)

data.to_csv('./shuffled-auto-mpg.csv')
