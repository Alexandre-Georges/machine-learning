#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.datasets import boston_housing
from keras import models
from keras import layers

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalization of the data (centered around 0 within -1 and +1 standard deviations)
# so all values have a similar range
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

# The number of samples is small, the network will also be small so it can be trained quickly
def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation = 'relu', input_shape = (train_data.shape[1],)))
  model.add(layers.Dense(64, activation = 'relu'))
  model.add(layers.Dense(1))

  # MAE : mean absolute value
  model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
  return model
