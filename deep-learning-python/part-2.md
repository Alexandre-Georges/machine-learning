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

Cf 9.1.5

We can also use pre-trained word embedding if the problem we want to solve uses a generic enough vocabulary. This approach is very similar to pre-trained convnets.

Word2vec is very popular one that can capture semantic properties like gender.
GloVe is another popular one.

#### Full example

Cf 9.2

### Recurrent neural networks

The previous neural networks do not have state : each serie of input is presented to the network and the network will process them independently.

The only way to connect different samples would be to have them all as an entire sequence. One example would be processing images vs processing videos (serie of images).

The networks we have seen can process images but those images would be processed indenpendently of each others unless we turn the all images into one serie of inputs (video).

Recurrent networks update their current state as they process the samples (like reading a sentence).

On a technical standpoint, each neuron takes as input a sample and also the previous state of the current neuron.

```python
state_t = 0
for input_t in input_sequence:
  output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
  state_t = output_t
```

Cf 9.3.1 for an implementation

This neuron is called SimpleRNN in Keras.

Recurrent networks have 2 modes : with the first one they will return the whole sequence of outputs (for each sample) or only the last output for each sample.

Keras has two other types of RNN : LSTM and GRU.
SimpleRNN is too simplistic to be used in real models and also it is hard to train such networks (vanishing gradient problem like with a network that has too many layers). LSTM and GRU are designed to solve this problem.

LSTM : Long Short-Term Memory
LSTM can carry (C_t) information accross many timesteps, it saves information for later so the information does not vanish.

```python
output_t = activation(dot(state_t, Uo) + dot(input_t, Wo) + dot(C_t, Vo) + bo)

i_t = activation(dot(state_t, Ui) + dot(input_t, Wi) + bi)
f_t = activation(dot(state_t, Uf) + dot(input_t, Wf) + bf)
k_t = activation(dot(state_t, Uk) + dot(input_t, Wk) + bk)

c_t+1 = i_t * k_t + c_t * f_t
```

`i_t * k_t` provide information about the current inputs (i_t) to update the carry track (k_t)
`c_t * f_t` means the carry (c_t) can be forgotten (f_t)

Basically a carry is computed from the input of the neuron which is then plugged into the next iteration and then becomes an input of the neuron. The neuron has 3 inputs : the regular inputs, the state of the last iteration and the carry of the last iteration.

The theory is not important, the neuron has a way to carry information from before if it thinks it is necessary.

Cf 9.3.4

#### Advanced uses of recurrent neural networks

Cf 9.4.*

To improve the performances we have other leads to explore :

- change the number of units in each layer
- change the learning rate
- switch to LSTM layers instead of GRU
- use bigger densely connected layer or multiple densely connected layers
- run the best performing models against the test set to not overfit to the training and validation sets

#### Sum-up

The procedure that we used is the following :

- start with a baseline so we can know if we improve over it
- try simple models first
- when the order of the samples matter RNN are a good fit
- when using an RNN adding a dropout is different, there is no layer for that but parameters (cf code)
- having multiple recurrent layers improve the representational power but is much more costly
- bi-directional RNN are good for language processing where the order of words matters
- but they are not so good when going through samples where the latest ones are more relevant than the first ones

For language processing, two concepts are particularly relevant but not studied in this book : recurrent attention and sequence masking.

### Sequence processing with convnets

Convnets are good at finding patterns, when we process a sequence we also look for patterns accross time which makes them particularly good for simple problems (text classification and timeseries forecasting) and cheaper to run than RNNs.

The sequence is processed by extracting a few samples with a sliding window and processing the data of those samples. For instance if we have a window of 5 samples when processing text (5 words), the system will be able to match those 5 or less words wherever they appear.

Cf 10.1

Beside on a local scale (window size), convnets do not care about the order of the samples unlike RNNs since they are processed independently. To catch long-term patterns, multiple convnet layers will have to be stacked on top of each others.

Cf 10.2

Now we will combine a convnet with an RNN so the convnet can find higher level features and the RNN can keep track of the order, also it makes the problem easier to solve as the
number of features processed by the RNN is smaller and therefore it is faster. This is especially useful when there are a lot of timesteps.

Cf 10.3

1D convnets perform well when looking for local patterns in temporal data, they are a lot faster than RNNs. Their structure is the same than 2D convnets (Conv1D and Pooling1D layers).

As RNNs are expensive to run, it can be great to have a 1D convnet to pre-process the data and make the learning process faster.

### Sum-up of chapter 6

RNNs are good at timeseries regression (predicting the future), timeseries classification, anomaly detection in timeseries and sequence labelling (like identifying name or dates in a sentence).

1D convnets can be used for sequence to sequence (to extract high-level features), document classification and spelling correction.

If the order of the sequence matters (for example when the more recent data matters more than the old one), it is better to use a RNN.

If the order does not matter (for text for instance when looking for a keyword) a 1D convnet will work as well as an RNN and is a lot cheaper to run.

## Chapter 7 - Advanced deep-learning best practices

### Going beyond the sequential model : the Keras functional API

All the previous networks have been sequential models but in some cases we might want to have indenpendent inputs or multiple outputs and we will end up with a graph-like model instead of a stack.

For instance, we could need to use a convnet for image processing and an RNN for text processing to answer a single problem (a product with a description and an image).

Similarly, we could also predict unrelated outputs using the same inputs.

Connections between layers could also be different, we might want to reuse the output of the 1st layer into the 3rd layer.

To implement those models, we will use the functional API of Keras.

```python
from keras import Input, layers

# Tensor
input_tensor = Input(shape = (32, ))

dense = layers.Dense(32, activation = 'relu')

# The layer called with a tensor returns a tensor
output_tensor = dense(input_tensor)
```

Cf 11.1

When merging multiple layers into one, we could use `keras.layers.add`, `keras.layers.concatenate`, etc.

Cf 11.2

We can also have multiple outputs, for instance if we want to predict the age, gender, etc of a person.

Cf 11.3

Any kind of network can be built as long as it is not cyclical. Inception modules are an example: they can be used to combine multiple convolutional networks with different sizes and windows into one.

There is also the example of a 2D convolutional layer with a width of 1 x 1. This kind of layer is called a pointwise convolution. They are used to extract information from the channels so they can be easier to process (fewer features than channels). For example when we analyze an image each pixel has a RGB channels (one channel per colour), using a 1x1 layer would process those channels into something more meaningful like contrast. They in fact act as a pre-processor that can reduce the number of features.

Cf 11.4

The concept of residual connections can be used in graph-like networks that have a lot of layers (more than 10). When added to the network they help deal with the vanishing gradients problem and representational bottlenecks.

Representational bottleneck : so far a layer can only process the output of the previous layer, if the previous layer destroys too much information, it can never be used in the subsequent layers. Residual connections fix that by having connections between the layer N, N-1 (regular case) and the layer N-2. The layer N-1 can destroy information but the layer N-2 is connected to the layer N so N-2 has access to all the data.

Vanishing gradient : when we have a huge stack of layers, the gradient gets lost and the network can not update itself significantly.

Here is an example where the features are the same for the layers that get merged.

```python
from keras import layers

x = ...

y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(y)
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(y)
y = layers.add([y, x])
```

And another one where they are different.

```python
from keras import layers

x = ...
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)
residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
y = layers.add([y, residual])
```

Another interesting feature is that we can reuse the same layer. If we imagine that we want to process sentences and have 2 inputs with sentences, we could process each input by using the same layer.

```python
from keras import layers
from keras import Input
from keras.models import Model

"""
This layer is reused as it processes text the same way for the 2 inputs and
it trains this layer with the data from both inputs.
"""
lstm = layers.LSTM(32)

left_input = Input(shape = (None, 128))
left_output = lstm(left_input)

right_input = Input(shape = (None, 128))
right_output = lstm(right_input)

merged = layers.concatenate([ left_output, right_output ], axis = -1)
predictions = layers.Dense(1, activation = 'sigmoid')(merged)

model = Model([ left_input, right_input ], predictions)
model.fit([ left_data, right_data ], targets)
```

The last interesting feature is using models as layers, effectively a model will become a layer like so :

```python
y = model(x)
```

With multiple inputs and outputs :

```python
y1, y2 = model([ x1, x2 ])
```

Basically when called, the model will reuse its weights and not update them.
For instance if we want to have 2 cameras to measure the depth, we could reuse the model that processes the images for both cameras.

```python
from keras import layers
from keras import applications
from keras import Input

# This is the base model to process images
xception_base = applications.Xception(weights = None, include_top = False)

# RGB inputs for 250 x 250 images
left_input = Input(shape = (250, 250, 3))
right_input = Input(shape = (250, 250, 3))

left_features = xception_base(left_input)
right_features = xception_base(right_input)

# The merged layer has the 2 cameras' features
merged_features = layers.concatenate([ left_features, right_features ], axis = -1)
```

### Inspecting and monitoring models with Keras callbacks and TensorBoard

When training a model we will get a better view of what is going on inside.

For example when we train a model, we do not know how many epochs will be necessary to reach the optimal point. What we can do instead is have a callback they will get called when the loss function is not improving anymore. This callback can :

- save the current weights at some specific points
- stop the training process
- adjust dynamically some parameters during training like the learning rate of the optimizer
- logging metrics as they change

Example with an early stopping and an export of weights :

```python
import keras

# Here is a list of the callbacks we will use
callbacks_list = [
  # This one stops when the validation accuracy has not improved between epoch X and epoch X + 1
  keras.callbacks.EarlyStopping(
    monitor = 'val_acc',
    patience = 1,
  ),
  # Save the weights
  keras.callbacks.ModelCheckpoint(
    filepath = 'my_model.h5',
    # Weights are saved when the validation loss is better than the best one before
    monitor = 'val_loss',
    save_best_only = True,
  ),
]

model.compile(
  optimizer = 'rmsprop',
  loss = 'binary_crossentropy',
  metrics = ['acc'],
)

# We need to feed validation data to the model to get val_acc and val_loss
model.fit(
  x,
  y,
  epochs = 10,
  batch_size = 32,
  callbacks = callbacks_list,
  validation_data = (x_val, y_val),
)
```

Now we could want to reduce or increase the learning rate when the loss function is not improving anymore. We might be at a local minima for instance.

```python
callbacks_list = [
  keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss'
    # Multiply the learning rate by this factor when the validation loss has not improved for 10 epochs
    factor = 0.1,
    patience = 10,
  ),
]

model.fit(
  x,
  y,
  epochs = 10,
  batch_size = 32,
  callbacks = callbacks_list,
  validation_data = (x_val, y_val),
)
```

It is also possible to write a callback by implementing the `Callback` class. Then the following methods can be used :

- `on_epoch_begin` : called at the beginning of each epoch
- `on_epoch_end` : called at the end of each epoch

- `on_batch_begin` : same for batches
- `on_batch_end` : same for batches

- `on_train_begin` : called at the beginning of the training phase
- `on_train_end` : called at the end of the training phase

Those methods are called with an argument that is a dictionnary of values regarding the situation of the model. The callback also has access to `self.model` and `self.validation_data`.

```python
import keras
import numpy as np

class ActivationLogger(keras.callbacks.Callback):

  # This is called before the training
  def set_model(self, model):
    self.model = model
    layer_outputs = [ layer.output for layer in model.layers ]
    self.activations_model = keras.models.Model(model.input, layer_outputs)

  def on_epoch_end(self, epoch, logs = None):
    if self.validation_data is None:
      raise RuntimeError('Requires validation_data.')

    # First sample for the validation data
    validation_sample = self.validation_data[0][0:1]
    activations = self.activations_model.predict(validation_sample)

    f = open('activations_at_epoch_' + str(epoch) + '.npz', 'wb')
    np.savez(f, activations)
    f.close()
```
