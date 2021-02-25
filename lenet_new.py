# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
## sklearn for data
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

## tensorflow etc
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

## other
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

# Returns lenet model for given number of classes. 
# NB: input shape is (width, height, depth), where depth = 3 for RGB
def lenet_model(n_classes):
	model = Sequential([
		Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(32, 32, 3), padding='same'), #C1
		AveragePooling2D(), #S2
		Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
		AveragePooling2D(), #S4
		Flatten(), #Flatten
		Dense(120, activation='tanh'), #C5
		Dense(84, activation='tanh'), #F6
		Dense(n_classes, activation='softmax') #Output layer
	])
	return model

def load_data():
	# initialize the data and labels
	print("[INFO] loading images...")
	data = []
	labels = []

	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images("img")))
	random.seed(42)
	random.shuffle(imagePaths)

	# loop over the input images
	for imagePath in imagePaths:
				
		# load the image, resize the image to be 32x32 pixels (ignoring
		# aspect ratio), and store the image in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (32, 32))
		data.append(image)
		
		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)

	print("LENGTH OF DATA: ", len(data))

	# partition the data: 20% test, 10% validation, 70% train
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.2, random_state=42)
	(trainX, valX, trainY, valY) = train_test_split(data,
		labels, test_size=0.125, random_state=42)

	print("SHAPE OF DATA: ", trainX[0].shape) #flattened = 3072 otherwise 32,32,3

	''' NB: IK SNAP NOG NIET WAT DIT PRECIES DOET '''
	# convert the labels from integers to vectors (for 2-class, binary
	# classification you should use Keras' to_categorical function
	# instead as the scikit-learn's LabelBinarizer will not return a
	# ~ # vector)
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)
	valY = lb.transform(valY)
	return trainX, trainY, valX, valY, testX, testY
	
''' NB: WERKT NOG NIET '''
def train_adam(trainX, trainY, valX, valY, testX, testY, model):
	print("\n [INFO] training network...")
	# Compile; uses Adam, so not original
	model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

	# Fit
	print("fitting...")
	model.fit(trainX, trainY, epochs=5, validation_data=(valX, valY))

	print("evaluating...")
	# Evaluate: this outputs
	model.evaluate(testX, testY)

''' WERKT WEL, MAAR ERG LAGE ACCURACIES '''
def train_sgd(trainX, trainY, valX, valY, testX, testY, model):
	# initialize our initial learning rate and # of epochs to train for
	INIT_LR = 0.01
	EPOCHS = 3
	# compile the model using SGD as our optimizer and categorical
	# cross-entropy loss (you'll want to use binary_crossentropy
	# for 2-class classification)
	print("\n [INFO] training network...")
	opt = SGD(lr=INIT_LR)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
		
	# train the neural network
	H = model.fit(x=trainX, y=trainY, validation_data=(valX, valY),
		epochs=EPOCHS, batch_size=32)
	
	# evaluate the network
	print("\n [INFO] evaluating network...")
	model.evaluate(testX, testY)
	predictions = model.predict(x=testX, batch_size=32)

	# plot the training loss and accuracy
	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.plot(N, H.history["accuracy"], label="train_acc")
	plt.plot(N, H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy (Simple NN)")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig("output/result.png")

	
(trainX, trainY, valX, valY, testX, testY) = load_data()
# ~ model = lenet_model(3)
train_sgd(trainX, trainY, valX, valY, testX, testY, lenet_model(3))
