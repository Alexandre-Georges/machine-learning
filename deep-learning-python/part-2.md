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

With multiple layers, convnets learn the relevant pieces thanks to the first layer, then the second will use those low-level patterns and make sense of them and build on top of them. And so on for the next layers, making each layer more global than the previous one.

`layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1))`
In our example there is 32 filters on 3 x 3 patches.

The output is always smaller than the input by 1 (vertically and horizontally). To get an output that has the same size than the input, we can use padding (`padding` argument set to `same` in the Conv2D layer).

By default, each filter is applied on each possible location (stride of 1) so the filter moves by 1 pixel. We could technically use a bigger stride to make the process faster but max-pooling is instead better.

Max-pooling is a layer that cuts down the number of features.

`layers.MaxPooling2D((2, 2)` divides the features by 2 for each dimension. It outputs the maximum value of the input features.

Effectively, using those layers compresses the features and by doing that the system generalizes them so it can learn from them and it also prevents overfitting.

### Training a convnet on a small dataset

Cf 7-cat-or-dog.py

### Pre-trained convnet

The system used a network that has been trained on a huge dataset so it does not require to learn as much.
Even if the system was not trained specifically on what we are trying to do, if it is a subset it can be reused.

The interesting features are extracted and run through a new classifier trained from scratch. As the classifier of the pre-trained network is too generic, we make a new one while reusing the convolutionnal base.

If the dataset differs a lot from the one used to pre-train the model, it is better to discard the last ones because they are higher level layers. Effectiveley the first layers are used to identify local patterns like textures, colours or edges.

Cf 8.1-pre-trained-convnet.py

We have 2 options to add the last layers :

- running the network (without the last layers) over the training set, exporting the output and use that as an input for a new network that contains only the last layers. This is cheap but can not use the data augmentation. Cf 8.2-without-data-augmentation.py
- add the layers on top of the current network and run it on the training set. This can use data augmentation and therefore can be more expensive to run. Cf 8.3-with-data-augmentation.py

To fine-tune the models, another approach is used : unfreeze a few top layers (last ones) and train it along with the added last layers. It adjusts the abstract representations (high level) to better match the current problem.

The last layer needs to be trained first so it will not destroy the potentially unfrozen top layers by propagating large errors and not overfit as we added a lot of parameters. Then the top layers of the original network is unfrozen and train with the added layers.

Cf 8.4-fine-tuning.py

### Conclusion 1

- Convnets are great for vision tasks
- They will overfit on a small dataset but data augmentation will help mitigate this issue
- Existing convnets can be reused for different tasks via feature extraction which is great for small datasets
- Fine-tuning helps a bit to fit the current problem more closely

### Visualization of a convnet

- Visualizing intermediate activations

This shows how the successive layers transform their inputs.

Cf 8.5.1

- Visualizing convnet filters

Shows what patterns filters are receptive to.

Cf 8.5.2

- Visualizing heatmaps of class activation

Explains which parts are used to classify an image.

Cf 8.5.3

### Conclusion 2

- Convnets are great for visual-classification problems
- They use a hierarchy of patterns and concepts to visualize
- Data augmentation is great to avoid overfitting

## Chapter 6 - Deep learning for text and sequences

There are 2 algorithms :

- recurrent neural networks
- 1D convnets (one dimension)

### Working with text data

Models do not take words, they take numeric tensors : words are converted into numbers (vectorizing). The text is broken down into words, characters or n-grams (sequence of characters or words, groups of N words or characters), called tokens.

One can use : one-hot encoding or token/word embedding.

N-gram : for instance a 2-gram with "the cat sat on the table" would be
"the", "cat", "sat", "on", "the", "table", "the cat", "cat sat", "sat on", ...
This is called a bag of 2 grams, they are actually seen as sets which destroys the order of the words. They are used for shallow models and not in deep learning models.

One-hot encoding : associates an integer with each word or character.

Cf 9.1.1, 9.1.2 and 9.1.3

Word embeddings : one-hot encoding gives big vectors that do not contain a lot of data (lots of 0s). Word embedding generates smaller vectors, the information is denser and requires less memory.

It creates a map to know how related words are compared to each other, the more related they are the closer they are on the map. Transformations can be applied to go from one word to another one (for instance king + female -> queen).

The context plays an important role, for better results a word-embedding space can be used only for the context it was trained on.

Cf 9.2.1

We can also use pre-trained word embedding if the problem we want to solve uses a generic enough vocabulary. This approach is very similar to pre-trained convnets.

Word2vec is very popular one that can capture semantic properties like gender.
GloVe is another popular one.

#### Full example
