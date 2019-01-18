f = open('jena_climate_2009_2016.csv')
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

import numpy as np

# Initializes the data with 0s
float_data = np.zeros((len(lines), len(header) - 1))

# Parses the data
for i, line in enumerate(lines):
  values = [float(x) for x in line.split(',')[1:]]
  float_data[i, :] = values

# Plots the temperature
from matplotlib import pyplot as plt

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.figure()

# Plots the first 10 days, one timestep is 10 minutes (1440 * 10 / 60 / 24 = 10 days)
plt.plot(range(1440), temp[:1440])

# First we need to prepare the data with some parameters

# We will look at the 5 previous days (720 * 10 / 60 / 24 = 5 days)
lookback = 720

# One data point per hour, we will bundle 6 data points in one (6 times 10 minutes)
step = 6

# This represent the target which is 24h in the future (144 * 10 / 60 / 24 = 1 day)
delay = 144

# The data needs to be normalized as they use different units and their ranges are dramatically different
mean = float_data[:200000].mean(axis = 0)
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std

"""
We will use a generator so we do not process multiple times the same data :
sample 1 includes day 1, 2, 3, 4, 5
sample 2 includes day 2, 3, 4, 5, 6

data : normalized data
lookback : how far we go in the past (5 days) for each label, number of samples for each label (2 : 2 samples for each label)
delay : how many timesteps in the future the target should be (0 : next sample, 1 : two samples away)
min/max_index : delimits the timeframe in the data that we will get data from (useful for delimiting data sets)
shuffle : shuffles the data or keep it in chronological order
batch_size : number of samples per batch, how many series it will return
step : number of timesteps to sample data (6 * 10 minutes = 1 data point per hour), effectively it returns one sample
out of step samples (2 : returns the 1st sample, third sample, fifth sample)
"""
def generator(data, lookback, delay, min_index, max_index, shuffle = False, batch_size = 128, step = 6):
  if max_index is None:
    max_index = len(data) - delay - 1
  i = min_index + lookback
  while 1:
    if shuffle:
      rows = np.random.randint(min_index + lookback, max_index, size = batch_size)
    else:
      if i + batch_size >= max_index:
        i = min_index + lookback
      rows = np.arange(i, min(i + batch_size, max_index))
      i += len(rows)

    samples = np.zeros((
      len(rows),
      lookback // step,
      data.shape[-1],
    ))

    targets = np.zeros((len(rows),))

    for j, row in enumerate(rows):
      indices = range(rows[j] - lookback, rows[j], step)
      samples[j] = data[indices]
      targets[j] = data[rows[j] + delay][1]

    yield samples, targets
