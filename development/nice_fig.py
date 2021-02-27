# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
## sklearn for data
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

## other
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from progress.bar import *


# This is slow, but works
def load_data_cv2():
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
	
	# Quick test to see if cv2 stuff works properly.
	# This opens the first (randomized) image. 
	# ~ print("[INFO] testing if the images are correctly loaded. Please close the popup window if OK")
	# ~ im = cv2.imread(imagePaths[0])
	# ~ cv2.imshow("test", im)
	# ~ cv2.waitKey(0)
	# ~ print("[INFO] actually starting loading. ")
	
	''' For development, use imagePaths[:100] or something like that '''
	p = imagePaths[:20]
	
	bar = Bar('Loading', fill='+', max=len(p))

	# loop over the input images
	for i, imagePath in enumerate(p):
	# ~ for imagePath in p:
		bar.next()
				
		# load the image and store the image in the data list
		image = cv2.imread(imagePath)
		
		data.append(image)
		# ~ data[i] = image
		
		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)
		# ~ labels[i] = label
	bar.finish()
	return data, labels
	
def prep_data_cv2(data, labels, width, height):	
	# Resize the image (ignoring aspect ratio)
	bar2 = Bar('Resizing', fill='+', max=len(data))
	for i, _ in enumerate(data):
		bar2.next()
		data[i] = cv2.resize(data[i], (width, height))#.flatten()
	bar2.finish()

	# Scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)

	print("LENGTH OF ALL DATA: ", len(data))

	# Partition the data: 20% test, 10% validation, 70% train
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.2, random_state=42)
	(trainX, valX, trainY, valY) = train_test_split(data,
		labels, test_size=0.125, random_state=42)

	print("SHAPE OF TRAIN DATA: ", trainX[0].shape) # should be (width, height, 3) because RGB

	# convert the labels from integers to vectors
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)
	valY = lb.transform(valY)
	return trainX, trainY, valX, valY, testX, testY

data, labels = load_data_cv2()
trainX = prep_data_cv2(data, labels, 180, 180)[0]

plt.figure(figsize=(10, 10))
for images in trainX.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")