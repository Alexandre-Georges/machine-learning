# Part 1

## Chapter 4 - Fundamentals of machine learning

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
- Iterated K-fold validation With shuffling : the data is shuffled before each fold and the training is done P times

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

* start with a small network

* increase its size until its performances get worse

Small network (overfits slowly)

```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

Original network

```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

Large model (overfits quickly and the validation error looks almost random)

```python
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

```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001), activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
```

This adds `0.001 * weight coefficient` to each coefficient of the layer to the loss of the network (during the training only).

L1 and L2

```python
from keras import regularizers

regularizers.l1(0.001)
regularizers.l1_l2(l1 = 0.001, l2 = 0.001)
```

#### Dropout

Randomly drops features (set to 0) in the layer during the training, it is usually set between 0.2 and 0.5.

It disrupts brittle (therefore insignificant and prone to overfitting) patterns while stronger ones are reinforced.

```python
model = models.Sequential()

model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
```

### Workflow

#### Define the problem and get the dataset

What should be the input that we want to predict : we need the data for it.

What type of problem are we trying to solve (binary, multiclass or multilabel classification, scalar or vector regression, clustering, etc) will define the model we use.

We have to make sure that the output can be predicted with the inputs. Some problems can not be solved with the available data ! For instance trying to predict which clothes people will buy with one month of data is impossible, it requires a few years of data and to use the time as a feature.

#### Choose a measure of success

We must be able to know how successful the system is by choosing the right loss function.

* Classification (where each class is as likely as the others) : accuracy and ROC AUC
* Imbalanced classification : precision and recall
* Multilabel classification : mean average precision

Kaggle gives access to custom evaluation metrics.

#### Evaluation protocol

We want to measure the progress of the system.

* hold-out validation set when we have a lot of data
* K-fold cross-validation when we have too few samples for hold-out
* iterated K-fold validation with shuffling for a great accuracy when little data available

#### Preparing the data

The data must be formatted as tensors and be in ranges ([-1, 1] or [0, 1]).

#### Developing a model

We want to make a simple model that can beat a random result. If it works it means we can predict at some extent the output with the data.

We will then :

* define the last-layer activation (or not in some cases like the regression example)
* pick a loss function
* select an optimizer

| Type | Activation | Loss |
|---|---|---|
| Binary classification | `sigmoid` | `binary_crossentropy` |
| Multiclass, single-label classification | `softmax` | `categorical_crossentropy` |
| Multiclass, multilabel classification | `sigmoid` | `binary_crossentropy` |
| Regression to arbitrary values | None | `mse` |
| Regression to values between 0 and 1 | `sigmoid` | `mse` or `binary_crossentropy` |

#### Scaling up

We want to find the sweetspot of the model by changing some parameters :

* add more layers
* have bigger layers
* train with more epochs

At the end we will have a model that can generalize without overfitting or underfitting.

#### Regularize and tuning

This step tweaks the model by :

* adding a dropout
* add or remove layers
* add L1 and/or L2 regularization
* try different hyperparameters (like the number of units per layer or the learning rate of the optimizer)
* potentially adding usefull or removing useless features

Too much tuning against the validation set will make the model overfit the validation data. When the process is over, we want to retrain the model and test it from scratch against a brand new test to prevent this issue.