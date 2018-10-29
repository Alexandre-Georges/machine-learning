#!/usr/bin/python
# -*- coding: utf-8 -*-

# We will try to categorize the samples into multiple topics
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

import numpy as np

import matplotlib.pyplot as plt

import copy

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = NUM_WORDS)

# Decode an article just in case
"""
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
"""

# Vectorization of the data
def vectorize_sequences(sequences, dimension = NUM_WORDS):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# The labels are integers that represent the 46 categories, we'll encode them
"""
def to_one_hot(labels, dimension = 46):
  results = np.zeros((len(labels), dimension))
  for i, label in enumerate(labels):
    results[i, label] = 1.
  return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
"""

# Or with the keras function :
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# In the last example 16 neuron layers were ok but with 46 categories it might be too small, we'll use 64
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (NUM_WORDS,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

# The output is the probabily of the sampe to belong to the given category (softmax)

# The best loss function in this case is categorical_crossentropy so we can measure the distance between
# the output and the truth for each category
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training set
partial_x_train = x_train[1000:]
partial_y_train = one_hot_train_labels[1000:]

# Validation set
x_val = x_train[:1000]
y_val = one_hot_train_labels[:1000]

history = model.fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = (x_val, y_val),
)

# Plot the loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

# Plot the accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 9 epochs is optimal
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (NUM_WORDS,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(
  partial_x_train,
  partial_y_train,
  epochs = 9,
  batch_size = 512,
  validation_data = (x_val, y_val),
)

results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# Accuracy of 80%

# With a random classification it would be about 19%
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(float(np.sum(hits_array)) / len(test_labels))

# Predictions with new data
predictions = model.predict(x_test)

# The sum of the probabilities accross all categories is 1
print(np.sum(predictions[0]))

# To get the class predicted by the system (highest probability)
print(np.argmax(predictions[0]))


# If we don't want to one-hot encode the categories, we could use the following (categories as integers, sparse_categorical_crossentropy)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(
  optimizer = 'rmsprop',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['acc'],
)
model.fit(
  partial_x_train,
  partial_y_train,
  epochs = 9,
  batch_size = 512,
  validation_data = (x_val, y_val),
)

results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# The number of units should be greater or equals to the number of outputs, if not :
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (NUM_WORDS,)))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))
model.compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = ['accuracy'],
)
model.fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 128,
  validation_data = (x_val, y_val),
)

# We end up with a 71% validation accuracy

# To improve, try to :
# - tweak the number of units (32, 128, etc)
# - use a single of 3 layers
# 