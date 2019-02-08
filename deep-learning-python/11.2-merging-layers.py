"""
Let's imagine that we want to answer a question thanks to a text.
We will need 2 text processing layers (one for the question and one for the text).

We will then merge their outputs into a concatenation layer.
"""

from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# Here is the text input and its network
text_input = Input(shape = (None, ), dtype = 'int32', name = 'text')

# 64 dimension vector
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

# And now we have the question input and its network
question_input = Input(shape = (None, ), dtype = 'int32', name = 'question')

# 32 dimensions this time
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# Here we concatenate the two networks
concatenated = layers.concatenate([encoded_text, encoded_question], axis = -1)

# And we have the dense layer that will give us the answer thanks to the softmax activation
answer = layers.Dense(answer_vocabulary_size, activation = 'softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

model.summary()

import numpy as np
from keras.utils.np_utils import to_categorical

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size = (num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size = (num_samples, max_length))

# Answers are random numbers between 0 and the size - 1
answers = np.random.randint(answer_vocabulary_size, size = (num_samples))

# Answers are converted into a matrix : 2 would be [0, 0, 1, 0, ...]
answers = to_categorical(answers, answer_vocabulary_size)

# The next two functions are identical, the second one requires the inputs to be named : name = 'text')
model.fit([text, question], answers, epochs = 10, batch_size = 128)
model.fit({ 'text': text, 'question': question }, answers, epochs = 10, batch_size = 128)
