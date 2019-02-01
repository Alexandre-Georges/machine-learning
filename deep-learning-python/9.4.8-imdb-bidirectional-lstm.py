# Finally, the bi-directional RNN, here we will use the LSTM.

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
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
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

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

from matplotlib import pyplot as plt


plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# 89% accuracy but it overfits fairly quickly since
# it has twice as many parameters than a regular RNN.