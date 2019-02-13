"""
In this model we will process social media comments
and we will have multiple outputs (age, income and gender).
"""
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape = (None, ), dtype = 'int32', name = 'posts')
embedded_posts = layers.Embedding(vocabulary_size, 256)(posts_input)

# 128 filters with a window of 5 words
x = layers.Conv1D(128, 5, activation = 'relu', padding='same')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation = 'relu', padding='same')(x)
x = layers.Conv1D(256, 5, activation = 'relu', padding='same')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation = 'relu', padding='same')(x)
x = layers.Conv1D(256, 5, activation = 'relu', padding='same')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation = 'relu')(x)

# We need to give names to each output
age_prediction = layers.Dense(1, name = 'age')(x)
income_prediction = layers.Dense(
  num_income_groups,
  activation = 'softmax',
  name = 'income',
)(x)
gender_prediction = layers.Dense(1, activation = 'sigmoid', name = 'gender')(x)
model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

"""
Each output will have its own loss function
and the loss function to minimize could be the sum of those functions.
The loss functions should have the same magnitude to not favour one output over the others.

The two functions below are equivalent, the second one requires the output layers to be named.
"""
model.compile(
  optimizer = 'rmsprop',
  loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy'],
)

model.compile(
  optimizer = 'rmsprop',
  loss = {
    'age': 'mse',
    'income': 'categorical_crossentropy',
    'gender': 'binary_crossentropy',
  },
)

"""
To balance loss functions we can also use weights but we must know how big or small
the error can be for each loss function.
"""
model.compile(
  optimizer = 'rmsprop',
  loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy'],
  loss_weights = [0.25, 1., 10.],
)

model.compile(
  optimizer = 'rmsprop',
  loss = {
    'age': 'mse',
    'income': 'categorical_crossentropy',
    'gender': 'binary_crossentropy',
  },
  loss_weights = {
    'age': 0.25,
    'income': 1.,
    'gender': 10.,
  },
)

# Finally we can train our model with random data
num_samples = 1000
max_length = 100

import numpy as np
from keras.utils.np_utils import to_categorical

posts = np.random.randint(1, vocabulary_size, size = (num_samples, max_length))

age_targets = np.random.randint(0, 100, size = (num_samples, 1))
income_targets = np.random.randint(1, num_income_groups, size = (num_samples, 1))
income_targets = to_categorical(income_targets, num_income_groups)
gender_targets = np.random.randint(0, 2, size = (num_samples, 1))

model.fit(
  posts,
  [age_targets, income_targets, gender_targets],
  epochs = 10,
  batch_size = 64,
)

model.fit(
  posts,
  {
    'age': age_targets,
    'income': income_targets,
    'gender': gender_targets,
  },
  epochs = 10,
  batch_size = 64,
)