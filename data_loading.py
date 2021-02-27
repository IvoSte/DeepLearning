# import the necessary packages

# sklearn for processing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# other
from imutils import paths
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

    data = []
    labels = []

    # Quick test to see if cv2 stuff works properly.
    # This opens the first (randomized) image.
    print("[INFO] testing if the images will be correctly loaded. Please close the popup window if OK")
    im = cv2.imread(imagePaths[0])
    cv2.imshow("test", im)
    cv2.waitKey(0)
    print("[INFO] actually starting loading. ")

    # For development, we can use imagePaths[:100] or something similar
    p = imagePaths[:10]

    bar = Bar('Loading', fill='+', max=len(p))

    # Loop over the input images
    for i, imagePath in enumerate(p):
        bar.next()

        # Load the image and store the image in the data list
        image = cv2.imread(imagePath)

        data.append(image)

        # extract the class label from the image path,
        # update the list of labels
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    bar.finish()
    return data, labels


# Pre-process the data by resizing, splitting in train/test/validation,
# scaling pixel intensities, and encode the labels
def prep_data_cv2(data, labels, width, height):
    # Resize the image (ignoring aspect ratio)
    bar2 = Bar('Resizing', fill='+', max=len(data))
    for i, _ in enumerate(data):
        bar2.next()
        data[i] = cv2.resize(data[i], (width, height))  # .flatten()
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

    print("SHAPE OF TRAIN DATA: ", trainX[0].shape)  # should be (width, height, 3) because RGB

    # Convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    valY = lb.transform(valY)
    return trainX, trainY, valX, valY, testX, testY
