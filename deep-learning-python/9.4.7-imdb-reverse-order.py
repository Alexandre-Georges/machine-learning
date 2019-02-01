"""
However for language, the order is not that meaningful,
we will try that on the IMDB dataset.
"""

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential

# Number of words (features)
max_features = 10000

# Reviews are limited to 500 words
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

# Reverses the order of the review words
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

# Padding in case the sequence is too short
x_train = sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = sequence.pad_sequences(x_test, maxlen = maxlen)

model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(
  optimizer = 'rmsprop',
  loss = 'binary_crossentropy',
  metrics = ['acc'],
)

history = model.fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2,
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
The performance of this model is almost identical to the other one with the right order.

This means that each system captures different patterns that can be useful. A bi-directionnal RNN
combines the 2 orders to build its model.
"""
