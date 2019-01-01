from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
import matplotlib.pyplot as plt

# We reuse the model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

# We take a random image
img_path = './cat_or_dog_small/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size = (150, 150))
img_tensor = image.img_to_array(img)

# Creates an array of images even if there is just one image here
img_tensor = np.expand_dims(img_tensor, axis = 0)

# This is how the inputs of the model were pre-processed
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()

# Get the outputs of the top 8 layers
layer_outputs = [layer.output for layer in model.layers[:8]]

# New model that will output the existing model's outputs
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

# Returns an array for each layer (8 layers)
activations = activation_model.predict(img_tensor)

# First convolution layer
first_layer_activation = activations[0]

# It has 32 channels
print(first_layer_activation.shape)

# Let's print a few
plt.matshow(first_layer_activation[0, :, :, 3], cmap = 'viridis')
plt.matshow(first_layer_activation[0, :, :, 6], cmap = 'viridis')

# Let's print them all now
layer_names = []

for layer in model.layers[:8]:
  layer_names.append(layer.name)
  images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):

  # Number of features
  n_features = layer_activation.shape[-1]

  # Size of each feature (square shaped : size x size)
  size = layer_activation.shape[1]

  # Create a grid to display each feature
  n_cols = n_features // images_per_row
  display_grid = np.zeros((size * n_cols, images_per_row * size))

  for col in range(n_cols):
    for row in range(images_per_row):
      channel_image = layer_activation[0, :, :, col * images_per_row + row]

      # Some processing to make the images understandable for humans
      channel_image -= channel_image.mean()

      # If the STD is equals to 0, it generates an error but it also means that the feature is not activated
      channel_image /= channel_image.std()
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image, 0, 255).astype('uint8')

      display_grid[
        col * size : (col + 1) * size,
        row * size : (row + 1) * size,
      ] = channel_image

  scale = 1. / size

  plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')

# We can see that the first layer looks for edges and most of the image stays
# As we go up, the network interprets higher-level concepts (ear, eye, etc),
# it becomes more difficult for us to understand as lots of information is discarded
# the network actually prepares those features for the classifier.
# We have fewer and fewer activations as we go up,
# when a feature is not activated, it means it is not found in the image.