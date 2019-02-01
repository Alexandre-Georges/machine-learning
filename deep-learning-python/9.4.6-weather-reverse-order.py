f = open('jena_climate_2009_2016.csv')
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

import numpy as np

# Initializes the data with 0s
float_data = np.zeros((len(lines), len(header) - 1))

# Parses the data
for i, line in enumerate(lines):
  values = [float(x) for x in line.split(',')[1:]]
  float_data[i, :] = values

# The data needs to be normalized as they use different units and their ranges are dramatically different
mean = float_data[:200000].mean(axis = 0)
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std

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

    yield samples[:, ::-1, :], targets

# Let's define some parameters

# We will look at the 10 previous days (1440 * 10 / 60 / 24 = 10 days)
lookback = 1440

# One data point per hour, we will use one data points out of 6 (6 times 10 minutes)
step = 6

# This represent the target which is 24h in the future (144 * 10 / 60 / 24 = 1 day)
delay = 144

# Number of samples generated each time we call the generator
batch_size = 128

# We create a generator for each data set
train_gen = generator(
  float_data,
  lookback = lookback,
  delay = delay,
  min_index = 0,
  max_index = 200000,
  shuffle = True,
  step = step,
  batch_size = batch_size,
)

val_gen = generator(
  float_data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size,
)

test_gen = generator(
  float_data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = None,
  step = step,
  batch_size = batch_size,
)

# Number of steps to draw to get the dataset
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

"""
We will try a bidirectional RNN later,
they are really good for natural language processing and they are flexible.
They perform well when the data has an order.

They go through the input sequence in one direction and then
the other (chronologically and antichronogically) and merge the results.

Which raise the question of : would the results have been different if the order of samples
had been reversed ?

The generator has been changed to return timesteps in the reverse order (from 9.4.3)
"""
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape = (None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer = RMSprop(), loss = 'mae')

history = model.fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps,
)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

from matplotlib import pyplot as plt

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

"""
It is way worse : the MAE is 0.43.
It means that the chronological order is important for this case, the most recent datapoints are more important.
"""