from keras.preprocessing.text import Tokenizer
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# Keeps the 1000 most used words
tokenizer = Tokenizer(num_words = 1000)

# Creates the word index
tokenizer.fit_on_texts(samples)

# Strings to numbers
sequences = tokenizer.texts_to_sequences(samples)

# We can also just use that instead (binary means 0 for not this word and 1 when it is this word)
one_hot_results = tokenizer.texts_to_matrix(samples, mode = 'binary')

np.set_printoptions(threshold = np.nan)
print(one_hot_results)

word_index = tokenizer.word_index

# 9 unique words
print('Found %s unique tokens.' % len(word_index))
