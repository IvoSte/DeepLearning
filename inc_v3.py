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
	model = InceptionV3()
	train_and_test.learning(model, 299, 299, "SGD", "categorical_crossentropy", 20) #Exp1


if __name__ == main():
    main()
