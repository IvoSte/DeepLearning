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
from progress.bar import *

# This is slow, but works
def load_data_cv2(width, height):
	# initialize the data and labels
	print("[INFO] loading images...")
	
	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images("img")))
	random.seed(42)
	random.shuffle(imagePaths)
	
	# ~ data = [None]*len(imagePaths)
	# ~ labels = [None]*len(imagePaths)
	data = []
	labels = []
	
	''' quick test to see if cv2 stuff works properly.
	This opens the first (randomized) image. Click or press any button
	to close it and continue.'''
	im = cv2.imread(imagePaths[0])
	cv2.imshow("test", im)
	cv2.waitKey(0)
	print("[INFO] actually starting loading. ")
	
	bar = Bar('Loading', fill='+', max=len(imagePaths))

	# loop over the input images
	for i, imagePath in enumerate(imagePaths):
		bar.next()
				
		# load the image, resize the image to be 32x32 pixels (ignoring
		# aspect ratio), and store the image in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (width, height))
		data.append(image)
		# ~ data[i] = image
		
		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)
		# ~ labels[i] = label
	bar.finish()

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
	# convert the labels from integers to vectors
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)
	valY = lb.transform(valY)
	return trainX, trainY, valX, valY, testX, testY
