# import the necessary packages
## tensorflow etc
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

# Exp 8: AlexNet, a deeper version of LeNet
def alex_net():
	model = Sequential([
		Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
		BatchNormalization(),
		MaxPool2D(pool_size=(3,3), strides=(2,2)),
		Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
		BatchNormalization(),
		MaxPool2D(pool_size=(3,3), strides=(2,2)),
		Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
		BatchNormalization(),
		Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
		BatchNormalization(),
		Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
		BatchNormalization(),
		MaxPool2D(pool_size=(3,3), strides=(2,2)),
		Flatten(),
		Dense(4096, activation='relu'),
		Dropout(0.5),
		Dense(4096, activation='relu'),
		Dropout(0.5),
		Dense(10, activation='softmax')
	])

	return model
