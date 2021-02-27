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

def inc_v3_model():
	model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299,299,3))
	
	# Since we do not use the top layer, we must make one ourselves
	x = model.output
	top_layer = Sequential([
	Dense(3, activation='softmax')
	])
	new_model = Model(inputs=model.input, outputs=top_layer)
	
	return new_model

def inc_v3_model_2():
	base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299,299,3), classes=3)
	
	# Add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)

	# Add a fully-connected layer and a logistic layer with 3 classes 
	x = Dense(512, activation='relu')(x)
	predictions = Dense(3, activation='softmax')(x)

	# The model we will train
	model = Model(inputs = base_model.input, outputs = predictions)

	# first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
	for layer in base_model.layers:
		layer.trainable = False
	
	return model
