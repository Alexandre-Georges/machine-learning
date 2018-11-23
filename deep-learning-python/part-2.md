# Part 2

## Chapter 5 - Deep learning for computer vision

### Introduction

Convnets : convolutional neural networks

Basic convnet

```python
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.summary()
```

The input shape is the size of the images and number of colours.

Plug the result into a layer that will flatten the input so it can be processed by the dense layers.

```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()
```

With the MNIST dataset :

```python
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 5, batch_size = 64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
```

The accuracy is more than 99%, way better than a regular dense neural network.

Dense networks learn global patterns, they take all the inputs. Convnets find local patterns like patches of pixels that are relevant for the analysis we want to do. Those patterns can be found anywhere in the image, they are not tied to a specific position.

With multiple layers, convnets learn the relevant pieces thanks to the first layer, then the second will use those low-level patterns and make sense of them and so on for the next layers.
