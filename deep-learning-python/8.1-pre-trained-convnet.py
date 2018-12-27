from keras.applications import VGG16

conv_base = VGG16(
  # Weights used in the model
  weights = 'imagenet',
  # Removes the densily connected layer (1000 classifier for imagenet), we will replace it with our own (dog or cat)
  include_top = False,
  # Sizes of the images
  input_shape = (150, 150, 3),
)

conv_base.summary()

