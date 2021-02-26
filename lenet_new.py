# import the necessary packages
## tensorflow etc
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

#other
import train_and_test

# Returns lenet model for given number of classes. 
# NB: input shape is (width, height, depth), where depth = 3 for RGB
def lenet_model(n_classes):
	model = Sequential([
		Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(28, 28, 3), padding='same'), #C1
		AveragePooling2D(), #S2
		Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
		AveragePooling2D(), #S4
		Flatten(), #Flatten
		Dense(120, activation='tanh'), #C5
		Dense(84, activation='tanh'), #F6
		Dense(n_classes, activation='softmax') #Output layer
	])
	return model

def main():
	model = lenet_model(3)
	train_and_test.learning(model, 28, 28, "SGD", "categorical_crossentropy", 20) #Exp1	
	# ~ train_and_test.learning(model, 28, 28, "Adam", "categorical_crossentropy", 20) #Exp4	

if __name__ == main():
    main()
