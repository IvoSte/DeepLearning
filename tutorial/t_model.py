import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # set cuda message level   
import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

IMAGE_SHAPE = (224, 224)

# Create a classifier.
    # Give download link for the classifier
classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    # Instantiate it
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

# get example image
grace_hopper = tf.keras.utils.get_file('image.jpg',\
    'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
# reshape
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)

# make to array
grace_hopper = np.array(grace_hopper)/255.0
print(grace_hopper.shape)

# give it to the model
    # Add a batch dimension, and pass the image to the model.
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)
    # result is 1001 element vector of logits,
    # rating the probability of each class for the image

# top class ID from the logits vector
predicted_class = np.argmax(result[0], axis = -1)
print(predicted_class)

# what does that class mean? 
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',\
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
