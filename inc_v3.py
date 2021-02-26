# import the necessary packages
## tensorflow etc
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import *

#other
import train_and_test

def main():
	# Experiments 1-4: different architectures
	# ~ model = lenet_model(3)
	model = InceptionV3()
	train_and_test.learn(model, 299, 299, "SGD", "categorical_crossentropy", 20) #Exp1	
	# ~ train_and_test(model, "Adam", "categorical_crossentropy", 20) #Exp4


if __name__ == main():
    main()
