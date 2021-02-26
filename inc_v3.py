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
import data_loading

def main():
	im_dat = data_loading.load_data_cv2(299, 299)
	model = InceptionV3(weights=None, classes=3)
	train_and_test.learning(im_dat, model, 299, 299, "SGD", "categorical_crossentropy", 20) #Exp1


if __name__ == main():
    main()
