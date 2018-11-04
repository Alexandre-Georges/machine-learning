#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.datasets import boston_housing
from keras import models
from keras import layers

import numpy as np

import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalization of the data (centered around 0 within -1 and +1 standard deviations)
# so all values have a similar range
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

# The number of samples is small, the network will also be small so we minimize the overfitting
def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation = 'relu', input_shape = (train_data.shape[1],)))
  model.add(layers.Dense(64, activation = 'relu'))
  # No activation so the output can take any value
  model.add(layers.Dense(1))

  # MAE : mean absolute value, good for regressions
  # MSE : mean squared error, good for regressions
  model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
  return model

# K-fold, the model is trained on 3/4 of the dataset and validated with the last 1/2
# this is done 4 times so it runs with the whole dataset and we compute the average
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
  print('processing fold #', i)
  val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
  val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
  partial_train_data = np.concatenate(
    [
      train_data[:i * num_val_samples],
      train_data[(i + 1) * num_val_samples:]
    ],
    axis = 0
  )
  partial_train_targets = np.concatenate(
    [
      train_targets[:i * num_val_samples],
      train_targets[(i + 1) * num_val_samples:]
    ],
    axis = 0
  )
  model = build_model()
  history = model.fit(
    partial_train_data,
    partial_train_targets,
    validation_data = (val_data, val_targets),
    epochs = num_epochs,
    batch_size = 1,
    verbose = 0
  )
  # MAE by epoch
  mae_history = history.history['val_mean_absolute_error']
  all_mae_histories.append(mae_history)

# MAE by epoch for all the folds
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

def smooth_curve(points, factor = 0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.clf()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 80 epochs seems to be optimal
model = build_model()
model.fit(train_data, train_targets, epochs = 80, batch_size = 16, verbose = 0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mse_score)
print(test_mae_score)
