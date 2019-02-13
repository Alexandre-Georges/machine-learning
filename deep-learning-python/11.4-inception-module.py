"""
This is an example of an inception module.
"""

from keras import layers
from keras.layers import Input

x = Input(shape = (28, 28, 1), dtype = 'float32', name = 'images')

"""
The stride needs to be the same for all branches so we have the same output shape.
Strides represent by how much the window will move (2 means a translation of 2 pixels).
By default it is 1 so the window scans every possible option.
"""

# 1x1 layer
branch_a = layers.Conv2D(128, 1, activation = 'relu', strides = 2)(x)

branch_b = layers.Conv2D(128, 1, activation = 'relu')(x)
branch_b = layers.Conv2D(128, 3, activation = 'relu', strides = 2)(branch_b)

branch_c = layers.AveragePooling2D(3, strides = 2)(x)
branch_c = layers.Conv2D(128, 3, activation = 'relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation = 'relu')(x)
branch_d = layers.Conv2D(128, 3, activation = 'relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation = 'relu', strides = 2)(branch_d)

# Concatenates the outputs into the network output
output = layers.concatenate([ branch_a, branch_b, branch_c, branch_d ], axis = -1)