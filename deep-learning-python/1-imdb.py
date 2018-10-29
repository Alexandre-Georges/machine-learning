#!/usr/bin/python
# -*- coding: utf-8 -*-

# This dataset has reviews (words) and positive/negative ratings for each review
from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

WORDS_TO_LOAD = 10000

# Loads the data set
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = WORDS_TO_LOAD)

# Reverses the encoding to get the text review
#word_index = imdb.get_word_index()
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# Converts the "word" sequence into params with a one-hot encode
def vectorize_sequences(sequences, dimension = WORDS_TO_LOAD):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Labels get vectorized (not sure why, it goes from float64 to float32)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()

# Relu : f(x) = x for x ∈ [0 ; ∞] ; f(x) = 0 for x ∈ [-∞ ; 0[
# It removes negative values
model.add(layers.Dense(16, activation = 'relu', input_shape = (WORDS_TO_LOAD,)))
model.add(layers.Dense(16, activation = 'relu'))

# Sigmoid : f(x) = 0 when x = -∞ ; f(x) = 0.5 when x = 0 ; f(x) = 1 when x = ∞
# The output ∈ [0 ; 1]
model.add(layers.Dense(1, activation = 'sigmoid'))

# Cross entropy is great for probabilities like here
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = [ 'acc' ])

# Training set
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

# Validation set
x_val = x_train[:10000]
y_val = y_train[:10000]

# 20 iterations over the entire training set by batches of 512 samples
history = model.fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = (x_val, y_val),
)

# history.history.keys() has 4 keys (acc that was asked, loss and the same two for the validation)
# Plot the loss during the training and validation steps
history_dict = history.history

# Losses for the training
loss_values = history_dict['loss']

# Losses for the validation
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1) 

# Bo will be blue dots
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Clear
plt.clf()

# Accuracy
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# With those graphs we see that we overfit very quickly (3 epochs seems optimal)

# Optimal model
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (WORDS_TO_LOAD,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 4, batch_size = 512)
results = model.evaluate(x_test, y_test)

# 88% accuracy, not bad but we can do better
print(results)

# After training the model, we can use it against new samples to predict the values
print(model.predict(x_test))

# The model is confident with some samples and not so much with others (like 0.6, 0.4)

# To improve the model we could :
# - switch from 2 hidden layers to 1 or 3
# - change the number of hidden units (32, 64)
# - use a different loss function (mse)
# - use another activation function (tanh)
