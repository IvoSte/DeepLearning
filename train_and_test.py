# import the necessary packages

## tensorflow etc
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import *

# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

## other
import numpy as np
import data_loading

def learning(im_dat, model, width, height, opt, chosen_loss, n_epochs):
	(trainX, trainY, valX, valY, testX, testY) = im_dat
	print("SHAPE OF DATA: ", trainX[0].shape) # should be (width,height,3) because RGB
	
	# initialize training epochs
	EPOCHS = n_epochs
	
	# compile the model
	print("\n[INFO] training network...")
	model.compile(loss=chosen_loss, optimizer=opt,
		metrics=["accuracy"])
		
	# train the neural network
	H = model.fit(x=trainX, y=trainY, validation_data=(valX, valY),
		epochs=EPOCHS, batch_size=32)
	
	# evaluate the network
	print("\n[INFO] evaluating network...")
	model.evaluate(testX, testY)
	predictions = model.predict(x=testX, batch_size=32)
	
	print(f"You used optimizer {opt}.")
	exp_nr = input("Which experiment number is this? ")

	# plot the training loss and accuracy
	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.plot(N, H.history["accuracy"], label="train_acc")
	plt.plot(N, H.history["val_accuracy"], label="val_acc")
	plt.title(f"Training Loss and Accuracy with optimizer {opt}")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(f"output/result_{exp_nr}.png")
