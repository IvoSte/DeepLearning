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

# Train costum classifier, by retraining the top layer of the classifier.
data_root = tf.keras.utils.get_file(
    'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

    # load these images into the model
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(data_root),
    validation_split = 0.2,
    subset = "training",
    seed=123,
    image_size = (img_height, img_width),
    batch_size=batch_size)

class_names = np.array(train_ds.class_names)
print(class_names)

# tensorflow Hub's conventions for image models is to expect float inputs in de [0,1] range. Use Rescaling later to achieve this.
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Use buffered prefetching. Two important methods to us when loading data.
# More info here: https://www.tensorflow.org/guide/data_performance#prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Run the classifier on a batch of images
result_batch = classifier.predict(train_ds)
predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(predicted_class_names)

# check with the images
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")

# Dowload the headless model
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

# Attack a classification head
num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

model.summary()

predictions = model(image_batch)
predictions.shape

# Train the model
# compile to configure the training process
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])
# fit to train the model
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

batch_stats_callback = CollectBatchStats()

history = model.fit(train_ds, epochs=2,
                    callbacks=[batch_stats_callback])

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)

# Check the predictions
# get ordered list of class names
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis = -1)
predicted_label_batch = class_names[predicted_id]

# plot the result

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")

# Export the model
t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
model.save(export_path)

# Check if the export was correct
reloaded = tf.keras.models.load_model(export_path)

# Retest
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

# Difference
abs(reloaded_result_batch - result_batch).max()