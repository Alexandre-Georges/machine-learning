# Get the data there : http://mng.bz/0tIo

import os

imdb_dir = './aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

# Labels contains the rating (negative/positive)
labels = []

# Texts contains the review texts
texts = []

# The reviews are either positive or negative
for label_type in ['neg', 'pos']:
  dir_name = os.path.join(train_dir, label_type)

  for fname in os.listdir(dir_name):
    if fname[-4:] == '.txt':
      f = open(os.path.join(dir_name, fname))
      texts.append(f.read())
      f.close()

      if label_type == 'neg':
        labels.append(0)
      else:
        labels.append(1)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Only the 100 first words of each review are kept
maxlen = 100

# Only 200 training samples
training_samples = 200
validation_samples = 10000

# Only the top 10k words
max_words = 10000

# Builds a dictionnary/map of words : each word gets a unique index
# { film: 19, the: 1, [...] }
# It keeps only a predefined number of words
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)

# This is the sequence of words, represented by their indexes : [ 1, 19 , [...]]
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Pad the shorter sequences with 0s
data = pad_sequences(sequences, maxlen = maxlen)
print('Shape of data tensor:', data.shape)

# Transforms the list into an np array
labels = np.asarray(labels)
print('Shape of label tensor:', labels.shape)

# Shuffles the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Splits the data into training and validation sets
x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# Download GloVe there : http://nlp.stanford.edu/data/glove.6B.zip

# We will build a map that for a defined word gives its coefficients
glove_dir = './glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

# Each line starts with the word and then a sequence of 100 coefficients (weights)
# We take the 40k most common words, the sequence details how the word is linked to the others
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype = 'float32')
  embeddings_index[word] = coefs

f.close()
print('Found %s word vectors.' % len(embeddings_index))

# Now we will create a matrix of the reviews' words
# We will copy over the coefficients from glove for each word of the reviews
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
  # We keep only the most used words, the other ones will have 0s
  if i < max_words:
    embedding_vector = embeddings_index.get(word)
    # Words that are not found in GloVe will have 0s as well
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector

# Definition of the model
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length = maxlen))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

# Loading GloVe into the first layer
model.layers[0].set_weights([ embedding_matrix ])
model.layers[0].trainable = False

model.compile(
  optimizer = 'rmsprop',
  loss = 'binary_crossentropy',
  metrics = [ 'acc' ],
)

history = model.fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 32,
  validation_data = (x_val, y_val),
)

model.save_weights('pre_trained_glove_model.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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

# Results might differ as reviews are shuffled
# The model quickly overfits, that is expected as the number of training samples is quite low.
# The accuracy is at about 57%.

# Let's compare without GloVe

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length = maxlen))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(
  optimizer = 'rmsprop',
  loss = 'binary_crossentropy',
  metrics = ['acc'],
)

history = model.fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 32,
  validation_data = (x_val, y_val),
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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

# The accuracy is at 52% which is less than with GloVe.
# With more training samples (20k), we can reach respectively 74% and 86%.
# Having a pre-trained model is helpful only with a small number of samples.
# With more samples it is better to start from scratch.

# Now let's test against the test data

# The data needs to be tokenized first
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []

for label_type in ['neg', 'pos']:
  dir_name = os.path.join(test_dir, label_type)
  for fname in sorted(os.listdir(dir_name)):
    if fname[-4:] == '.txt':
      f = open(os.path.join(dir_name, fname))
      texts.append(f.read())
      f.close()
      if label_type == 'neg':
        labels.append(0)
      else:
        labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen = maxlen)
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

# We get an accuracy of 56%.