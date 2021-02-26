# Blatantly stolen from https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342

import tensorflow as tf
from tensorflow import keras
import numpy as np
import data_prep

def data_mnist():
    train_ds, test_ds = keras.datasets.mnist.load_data()
    print(type(train_ds)) # = np array
    for t in train_ds:
        print("type: ", type(t)) # = np array

        # Normalize pixel intensities
        t = t / 255.0
    
    for t in test_ds:
        print("type: ", type(t)) # = np array

        # Normalize pixel intensities
        t = t / 255.0

    train_ds = tf.expand_dims(train_ds[0], 3)
    test_ds = tf.expand_dims(test_ds[0], 3)
    val_ds = train_ds[:5000]

    print(type(train_ds), type(test_ds), type(val_ds))
    return train_ds, val_ds, test_ds

def our_data():
    # (train_ds, val_ds, test_ds) = data_prep.gen_data_2()
    (train_ds, val_ds, test_ds) = data_prep.gen_data_1()

    # TODO normalize, expand dims

    return train_ds, val_ds, test_ds

def run_lenet(train_ds, val_ds, test_ds):

    # Actual model
    '''Changed input_shape in order to not need train_x. Same values tho'''
    lenet_5_model = keras.models.Sequential([
        keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(28, 28, 1), padding='same'), #C1
        keras.layers.AveragePooling2D(), #S2
        keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
        keras.layers.AveragePooling2D(), #S4
        keras.layers.Flatten(), #Flatten
        keras.layers.Dense(120, activation='tanh'), #C5
        keras.layers.Dense(84, activation='tanh'), #F6
        keras.layers.Dense(10, activation='softmax') #Output layer
    ])

    # Compile; uses Adam, so not original? 
    lenet_5_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    
    # Fit
    print("fitting...")
    lenet_5_model.fit(train_ds, batch_size=5, validation_data=val_ds, epochs=5)

    # Evaluate
    print("evaluating...")
    lenet_5_model.evaluate(test_ds)


# Main part: get data as tuple and pass it to runner
if input("Would you like to use the our data? Default is MNIST. y/[n] ") == 'y':
    print("OK, using our data.")
    run_lenet(*our_data())
else:
    print("OK, using MNIST data.")
    run_lenet(*data_mnist())