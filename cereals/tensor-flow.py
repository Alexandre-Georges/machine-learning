import tensorflow as tf
import pandas as pd

data = pd.read_csv('cereals.csv', header = 0)
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
data = data.sample(frac = 1).values

maxes = []

for x in range(0, 6):
  data_max = max(data[:, x])
  maxes.insert(x, data_max)
  data[:, x] = data[:, x] / data_max

training_set = data[:60]
test_set = data[60:75]

training_data = training_set[:, 0:5]
training_rating = training_set[:, 5]

test_data = test_set[:, 0:5]
test_rating = test_set[:, 5]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5, input_dim=5, activation='relu', kernel_initializer='normal'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(training_data, training_rating, epochs = 3000, batch_size = 60)

result = model.predict(test_data)
print(result)
print('\n')
print(test_rating)
"""
score = model.evaluate(test_data, test_rating)
model.summary()
print(score)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
 """