from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import cv2

# In this case, we keep the top layer as we will use a different photo
model = VGG16(weights = 'imagenet')

img_path = './elephants.jpg'
img = image.load_img(img_path, target_size = (224, 224))

# Array 224 x 224 x 3
x = image.img_to_array(img)

# Array 1 x 224 x 224 x 3
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

preds = model.predict(x)

# Gives the most likely classes with a probability for each
print('Predicted:', decode_predictions(preds, top = 3)[0])

# African elephant class
print(np.argmax(preds[0]))

# Let's find which parts were used for this classification
african_elephant_output = model.output[:, np.argmax(preds[0])]
last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# Vector (512, )
pooled_grads = K.mean(grads, axis = (0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
  conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis = -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./elephant_heatmap.jpg', superimposed_img)

# Now we know why this image was classified as elephant and where is the elephant
