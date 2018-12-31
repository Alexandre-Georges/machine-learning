from keras.applications import VGG16
import os
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

conv_base = VGG16(
  # Weights used in the model
  weights = 'imagenet',
  # Removes the densily connected layer (1000 classifier for imagenet), we will replace it with our own (dog or cat)
  include_top = False,
  # Sizes of the images
  input_shape = (150, 150, 3),
)

base_dir = './cat_or_dog_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Creating a new model with the existing one as a base
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()

# In Keras > 2.1, conv_base.trainable works as expected unlike before
# In this case, we will retrain the base to reach the annonced accuracy
# conv_base.trainable = False

# Training the network with data augmentation
train_datagen = ImageDataGenerator(
  rescale = 1. / 255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = True,
  fill_mode = 'nearest',
)
test_datagen = ImageDataGenerator(rescale = 1. / 255)
train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size = (150, 150),
  batch_size = 20,
  class_mode = 'binary',
)
validation_generator = test_datagen.flow_from_directory(
  validation_dir,
  target_size = (150, 150),
  batch_size = 20,
  class_mode = 'binary',
)

model.compile(
  loss = 'binary_crossentropy',
  optimizer = optimizers.RMSprop(lr = 2e-5),
  metrics = ['acc'],
)
history = model.fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50,
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

# Accuracy of 96%