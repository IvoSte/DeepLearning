# import the necessary packages
## tensorflow etc
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import *


def create_vgg():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding="same",
			input_shape=(224, 224, 3)))
    model.add(layers.Activation("relu"))
    #model.add(BatchNormalization(axis=chanDim))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
# (CONV => RELU) * 2 => POOL layer set
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.Activation("relu"))
    #model.add(BatchNormalization(axis=chanDim))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.Activation("relu"))
    #model.add(BatchNormalization(axis=chanDim))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))


# (CONV => RELU) * 3 => POOL layer set
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.Activation("relu"))
    #model.add(BatchNormalization(axis=chanDim))
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.Activation("relu"))
    #model.add(BatchNormalization(axis=chanDim))
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.Activation("relu"))
    #model.add(BatchNormalization(axis=chanDim))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

# first (and only) set of FC => RELU layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation("relu"))
    #model.add(BatchNormalization())
    model.add(layers.Dropout(0.5))
    # softmax classifier
    model.add(layers.Dense(units=43))
    model.add(layers.Activation("softmax"))
    # return the constructed network architecture
    return model


def get_vgg():
	model = VGG16()
