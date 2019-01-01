from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(
  weights = 'imagenet',
  include_top = False,
)

def deprocess_image(x):
  x -= x.mean()
  x /= (x.std() + 1e-5)
  x *= 0.1
  x += 0.5
  x = np.clip(x, 0, 1)
  x *= 255
  x = np.clip(x, 0, 255).astype('uint8')
  return x

def generate_pattern(layer_name, filter_index, size = 150):

  # With a gradient descent we will find the image the filter responds to the best
  layer_output = model.get_layer(layer_name).output
  loss = K.mean(layer_output[:, :, :, filter_index])

  # List of tensors of size 1
  grads = K.gradients(loss, model.input)[0]

  # + 1e-5 to avoid dividing by 0
  grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

  iterate = K.function([model.input], [loss, grads])

  # Starts from a grey image with some noise
  input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

  # Size of each step
  step = 1.
  # Run a gradient descent for 40 steps
  for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

  img = input_img_data[0]
  return deprocess_image(img)

# The filter 0 in this layer responds to dot patterns
plt.imshow(generate_pattern('block3_conv1', 0))

# Now let's display the 64 first filters of a layer
def plot_filters(layer_name):
  size = 64
  margin = 5
  results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

  for i in range(8):
    for j in range(8):
      filter_img = generate_pattern(layer_name, i + (j * 8), size = size)
      horizontal_start = i * size + i * margin
      horizontal_end = horizontal_start + size
      vertical_start = j * size + j * margin
      vertical_end = vertical_start + size
      results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

  plt.figure(figsize = (20, 20))
  plt.imshow(results)

# This layer takes care of simple edges and colours
plot_filters('block1_conv1')

# This one works on textures based on the edges and colours of the layer below
plot_filters('block2_conv1')

plot_filters('block3_conv1')
plot_filters('block4_conv1')
