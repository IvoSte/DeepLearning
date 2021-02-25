import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # set cuda message level   
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
from tensorflow.keras import layers

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory="../data/train",
  labels="inferred",
  label_mode="categorical",
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
print(type(train_ds))

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "../data/validation",
  labels="inferred",
  label_mode="categorical",
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "../data/test",
  labels="inferred",
  label_mode="categorical",
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# ### Configure the dataset for performance
# Let's make sure to use buffered prefetching so we can yield data from disk without having I/O become blocking. 
# These are two important methods you should use when loading data.

# `.cache()` keeps the images in memory after they're loaded off disk during the first epoch. 
# This will ensure the dataset does not become a bottleneck while training your model. 
# If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.

# `.prefetch()` overlaps data preprocessing and model execution while training. 
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ### Train a model
# For completeness, we will show how to train a simple model using the datasets we just prepared. This model has not been tuned in any way - the goal is to show you the mechanics using the datasets you just created. To learn more about image classification, visit this [tutorial](https://www.tensorflow.org/tutorials/images/classification).
num_classes = 2

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


# Note: we will only train for a few epochs so this tutorial runs quickly. 
print("fitting...")
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

print("evaluating...")
model.evaluate(test_ds)
