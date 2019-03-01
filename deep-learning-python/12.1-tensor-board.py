import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
from keras import backend as K

K.clear_session()

# We have a limit of 2 000 words
max_features = 2000

# And another limit of 500 words per comment
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

x_train = sequence.pad_sequences(x_train, maxlen = max_len)
x_test = sequence.pad_sequences(x_test, maxlen = max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length = max_len, name = 'embed'))
model.add(layers.Conv1D(32, 7, activation = 'relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation = 'relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])

callbacks = [
  # This initialize TensorBoad, remove some of the parameters as like this it takes too much time
  keras.callbacks.TensorBoard(
    log_dir = 'my_log_dir',
    # The histogram records every epoch
    histogram_freq = 1,
    write_grads = True,
    write_images = True,
    embeddings_freq = 1,
    embeddings_data = x_test[0:1000],
  ),
]

history = model.fit(
  x_train,
  y_train,
  epochs = 5,
  batch_size = 128,
  validation_split = 0.2,
  callbacks = callbacks,
)
