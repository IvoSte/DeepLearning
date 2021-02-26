import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # set cuda message level   
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import numpy as np

def create_classifier():
    # create classifier
    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), # shape of images
                                    include_top=False,           # leave out last fc layer
                                    weights = 'imagenet')
    
    for layer in pre_trained_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
            # Flatten the output layer to 1 dimension
            # pre_trained_model.output,
            tf.keras.layers.Flatten(),
            # Add a fully connected layer with 1024 hidden units and a ReLU activation
            tf.keras.layers.Dense(1024, activation='relu'),
            # Add dropout rate of 0.2
            tf.keras.layers.Dropout(0.2),
            # Add a final sigmoid layer for classification
            tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer = RMSprop(lr=0.0001),
                    loss = 'binary_crossentropy',
                    metrics = ['acc'])
    return model

def data_generators(train_dir, validation_dir):
    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255.,
                                        rotation_range = 40,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True)

    # Note that the validation data should not be augmented
    test_datagen = ImageDataGenerator(rescale = 1.0/255.)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                        batch_size = 20,
                                        class_mode = 'binary',
                                        target_size = (150, 150))

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                        batch_size = 20,
                                        class_mode = 'binary',
                                        target_size = (150, 150))

    return train_generator, validation_generator
    
def train(model, train_generator, validation_generator):
    callbacks = tf.keras.callbacks.Callback()
    history = model.fit(
                train_generator,
                validation_data = validation_generator,
                steps_per_epoch = 1,
                epochs = 10,
                validation_steps = 10,
                verbose = 1,
                callbacks = [callbacks])
    


def main():

    model = create_classifier()
    train_generator, validation_generator = data_generators("data/train", "data/validation")
    train(model, train_generator, validation_generator)

if __name__ == main():
    main()