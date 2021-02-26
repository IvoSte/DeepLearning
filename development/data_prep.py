## Official data prep file, consisting of helper functions. 

import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Originated in model.py -> def main
def gen_data_1():
    # data generator object -- used so we can stream data while training to save memory
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # iterator over the dataset, sorted in classes == dir name of files
    train_it = datagen.flow_from_directory('data/train', batch_size= 5, class_mode='binary')
    #test_it = datagen.flow_from_directory('data/test')
    validation_it = datagen.flow_from_directory('data/validation', batch_size= 5, class_mode='binary')

    test_it = datagen.flow_from_directory('data/test', batch_size= 5, class_mode='binary')
    return train_it, validation_it, test_it

# Originated in imagetest.py
def gen_data_2():
    
    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="data/train",
        labels="inferred",
        label_mode="categorical",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/validation",
        labels="inferred",
        label_mode="categorical",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/test",
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
    return train_ds, val_ds, test_ds