import tensorflow as tf
import pandas as pd
import math

data = pd.read_csv(
  'shuffled-auto-mpg.csv',
)
data = data.drop(data.columns[0], axis = 'columns')

training_proportion = 0.95
training_size = int(math.ceil(data.shape[0] * training_proportion))

training_set = data.head(training_size)
training_data = training_set.drop(axis = 'columns', labels = ['mpg'])
training_result = training_set[['mpg']]

test_set = data.tail(data.shape[0] - training_size)
test_data = test_set.drop(axis = 'columns', labels = ['mpg'])
test_result = test_set[['mpg']]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(len(training_data.columns), input_dim = len(training_data.columns), activation = 'elu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(10, kernel_initializer = 'random_uniform'))
model.add(tf.keras.layers.Dense(1, kernel_initializer = 'random_uniform'))
model.compile(loss = 'mean_absolute_error', optimizer = 'adam')

model.fit(training_data, training_result, epochs = 4000, shuffle = True)

predictions = model.predict(test_data)
score = model.evaluate(test_data, test_result)

print('\n')
print(predictions - test_result)
print('\n')

print(score)
