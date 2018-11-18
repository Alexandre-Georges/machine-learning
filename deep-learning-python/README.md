# Deep learning with Python

## Chapter 4

### Definitions

* Sample (input) : one data point
* Prediction (output) : result of the model

* Target : truth, what should have been predicted
* Prediction error (loss value) : distance between the prediction and the target

* Classes : Set of labels to choose from for classification problems ([ 'cat', 'dog' ])
* Label : instance of a class ('cat')
* Ground-truth (annotations) :  Targets for a dataset
* Binary classification : 2 possible categories
* Multiclass classification : multiple categories, a sample can have only one class
* Multilabel classification : multiple categories, a sample can have multiple labels

* Scalar regression : the target is continuous (like a function)
* Vector regression : the target is a set of continuous values

### Evaluation of the model

The dataset is cut into 3 : training, validation and test.

* Training : used to train the model
* Validation : to evaluate the model and tweak it
* Test : when the model is ready, the test set tests it

Tweaking too much the model results in overfitting the validation set and might not be representative of the model performance.

The test set has not been used for any kind of purpose when building or tweaking the model so it should be representative.

When little data is available, we can use a few techniques :
- Simple hold-out validation :  regular case where we have 3 sets; bad when not a lot of data is available, the validation and test sets are too small to be statisctically representative
- K-fold validation :
The data gets splitted in chunks (4-5) of same size. The model is created and trained on the 4/5 of the data and validated for each fold (the validation set changes everytime). The results are averaged over each fold.

With shuffling : the data is shuffled before each fold.

To keep in mind :
- the data should be representative (shuffling in some cases)
- when the data is time-based, no shuffling obviously
- the data should not be redundant in any way or the model will be trained twice on the duplicated sample

### Data preprocessing, feature engineering and feature learning

#### Data preprocessing

Vectorization : data needs to be converted into tensors (words into integer for instance)

Value normalization : large values should not be given to a neural networks or big difference in range between 2 features. All feature values should be between 0 and 1. It is nice to have a mean of 0 and a standard deviation of 1.

```
x -= x.mean(axis=0)
x /= x.std(axis=0)
```

Missing values : can be set to 0, the system will learn to ignore 0s. If we have missing values they should be in the training set so the system can learn them.

#### Feature engineering

The data might need to be processed so the system can learn correctly instead of fiddling with data that seems random to it.

### Overfitting and underfitting

The system needs to learn enough to be able to understand patterns but not too much to prevent learning the training data. The best way to manage this issue is to get more data, beside this solution that is not always available we can use the following ones.

#### Reduce the network's size

A big network will learn the training data and quickly overfit. With a good size network, it will learn the useful patterns and be able to generalize. A small network will not be able to learn any pattern and will underfit.

Multiple architectures have to be tested to see which one is the best:
- start with a small network
- increase its size until its performances get worse

Small network (overfits slowly)
```
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

Original network
```
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

Large model (overfits quickly and the validation error looks almost random)
```
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

#### Weight regularization

The system is forced to use smaller weights by adding a cost.

L1 regularization : add a cost proportional to the absolute value of the weight coefficients

L2 regularization : add a cost proportional to the square of the value of the weight coefficients

```
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001), activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
```

This adds `0.001 * weight coefficient` to each coefficient of the layer to the loss of the network (during the training only).

L1 and L2
```
from keras import regularizers

regularizers.l1(0.001)
regularizers.l1_l2(l1 = 0.001, l2 = 0.001)
```

#### Dropout

Randomly drops features (set to 0) in the layer during the training, it is usually set between 0.2 and 0.5.

It disrupts brittle (therefore insignificant and prone to overfitting) patterns while stronger ones are reinforced.

```
model = models.Sequential()

model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
```
