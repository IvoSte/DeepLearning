# Blatantly stolen from https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342

import tensorflow as tf
from tensorflow import keras
import numpy as np

def lenet_model():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    print(len(train_x), len(train_y), len(test_x), len(test_y))

    # Normalize pixel intensities
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    # # 
    train_x = tf.expand_dims(train_x, 3)
    test_x = tf.expand_dims(test_x, 3)
    val_x = train_x[:5000]
    val_y = train_y[:5000]

    print(type(train_x), type(test_x), type(val_x))

    # Actual model
    lenet_5_model = keras.models.Sequential([
        keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same'), #C1
        keras.layers.AveragePooling2D(), #S2
        keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
        keras.layers.AveragePooling2D(), #S4
        keras.layers.Flatten(), #Flatten
        keras.layers.Dense(120, activation='tanh'), #C5
        keras.layers.Dense(84, activation='tanh'), #F6
        keras.layers.Dense(10, activation='softmax') #Output layer
    ])

    # Compile; uses Adam, so not original
    lenet_5_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    # Fit
    print("fitting...")
    lenet_5_model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))

    print("evaluating...")
    # Evaluate: this outputs
    lenet_5_model.evaluate(test_x, test_y)

lenet_model()