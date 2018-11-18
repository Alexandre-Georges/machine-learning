num_validation_samples = 10000
np.random.shuffle(data)

validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

training_data = data[:]

model = get_model()
model.train(training_data)

validation_score = model.evaluate(validation_data)

# Tuning goes here

model = get_model()

# Training a new model of all the data beside the test set
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)
