from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt

base_dir = './cat_or_dog_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# To help, we will use data augmentation : it randomly transforms the data in a believable way so we
# in fact create new data that can be used to train the model
datagen = ImageDataGenerator(
  # Rotation of 40 degrees
  rotation_range = 40,
  # Translations up to 20%
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  # Shearing transformations
  shear_range = 0.2,
  # Zoom on the image
  zoom_range = 0.2,
  # Randomly flip half the images
  horizontal_flip = True,
  # If new pixels are created (translation for example) it defines how to colour them (nearest pixel)
  fill_mode = 'nearest'
)

train_cats_dir = os.path.join(train_dir, 'cats')
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# Pick a random image
img_path = fnames[3]

# Read the image and resize it
img = image.load_img(img_path, target_size = (150, 150))

# Converts the image into an numpy array
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1, ) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size = 1):
  plt.figure(i)
  imgplot = plt.imshow(image.array_to_img(batch[0]))
  i += 1
  if i % 4 == 0:
    break

plt.show()

# As a result, an image becomes multiples ones but their inputs are still heavily correlated, we will use a dropout layer to help
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4), metrics = ['acc'])

train_datagen = ImageDataGenerator(
  rescale = 1./255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = True,
)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size = (150, 150),
  batch_size = 32,
  class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
  validation_dir,
  target_size = (150, 150),
  batch_size = 32,
  class_mode = 'binary'
)

history = model.fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

# Saving the model
model.save('cats_and_dogs_small_2.h5')

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

# We now reach 82% accuracy from 72%, with some tweaks we can reach 86-87%