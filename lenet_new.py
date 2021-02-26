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

# ~ def main():
	# ~ # Number of classes
	# ~ n_classes = 3
	
	# ~ # Dimensions of the convolutional layers in the model, and thus
	# ~ # in which dimensions to resize the images
	# ~ width = 28
	# ~ height = 28
	
	# ~ model = lenet_model(width, height, n_classes)
	# ~ im_dat = data_loading.load_data_cv2(width, height)
	# ~ print("Welcome! Model is standard LeNet, with 28x28 RGB images.")
	# ~ choice = input("Please specify which optimizer you would like to test. \nFill in optimizer (SGD/Adam/etc by keras): \nOr quit program by pressing enter.")
	# ~ while choice:
		# ~ EPOCHS = int(input("Please specify the number of epochs. "))
		# ~ train_and_test.learning(im_dat, model, width, height, choice, "categorical_crossentropy", EPOCHS)

# ~ if __name__ == main():
    # ~ main()
