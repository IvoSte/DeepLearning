# import the necessary packages
## tensorflow etc
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

#other
import train_and_test
import data_loading

# Returns lenet model for given number of classes. 
# NB: input shape is (width, height, depth), where depth = 3 for RGB
def lenet_model(width, height, n_classes):
	model = Sequential([
		Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(width, height, 3), padding='same'), #C1
		AveragePooling2D(), #S2
		Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
		AveragePooling2D(), #S4
		Flatten(), #Flatten
		Dense(120, activation='tanh'), #C5
		Dense(84, activation='tanh'), #F6
		Dense(n_classes, activation='softmax') #Output layer
	])
	return model

# Exp 2: LeNet model with addition of a dropout layer
def lenet_model_dropout(width, height, n_classes):
	model = Sequential([
		Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(width, height, 3), padding='same'), #C1
		AveragePooling2D(), #S2
		Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
		AveragePooling2D(), #S4
		Flatten(), #Flatten
		Dense(120, activation='tanh'), #C5
		Dropout(0.5),
		Dense(84, activation='tanh'), #F6
		Dense(n_classes, activation='softmax') #Output layer
	])
	return model

# Exp 3: LeNet model with addition of a batch normalization layer
def lenet_model_batch_norm(width, height, n_classes):
	model = Sequential([
		Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(width, height, 3), padding='same'), #C1
		AveragePooling2D(), #S2
		Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
		AveragePooling2D(), #S4
		Flatten(), #Flatten
		Dense(120, activation='tanh'), #C5
		Dropout(0.5),
		Dense(84, activation='tanh'), #F6
		Dense(n_classes, activation='softmax') #Output layer
	])
	return model

# Exp 5: LeNet model in which all tanh activations have become ReLU
def lenet_model_relu(width, height, n_classes):
	model = Sequential([
		Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(width, height, 3), padding='same'), #C1
		AveragePooling2D(), #S2
		Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
		AveragePooling2D(), #S4
		Flatten(), #Flatten
		Dense(120, activation='tanh'), #C5
		Dense(84, activation='tanh'), #F6
		Dense(n_classes, activation='softmax') #Output layer
	])
	return model
