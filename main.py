import lenet_new
import data_loading
import train_and_test

def main():
	# Number of classes
	n_classes = 3
	
	# Dimensions of the convolutional layers in the model, and thus
	# in which dimensions to resize the images
	width = 28
	height = 28
	
	im_dat = data_loading.load_data_cv2(width, height)
	print("Welcome!")	
	
	while True:
		model_choice = int(input("Please specify which model you would like to test. \n1 = LeNet (28x28)\n2 = InceptionV3 (299x299, does not work yet, don't choose)\n3 = ResNet-50 (does not work yet, don't choose)\n4: Exit program"))
		# ~ reshaped_im = data_loading.prep_data_cv2(im_dat, width, height)
		if model_choice == 1:
			choice = input("Please specify which optimizer you would like to test. \nFill in optimizer (SGD/Adam/etc by keras): \nOr quit program by pressing enter.  ")
			model = lenet_new.lenet_model(width, height, n_classes)
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(im_dat, model, width, height, choice, "categorical_crossentropy", EPOCHS)
				choice = input("Please specify which optimizer you would like to test. \nFill in optimizer (SGD/Adam/etc by keras): \nOr quit program by pressing enter.  ")
		else:
			print("Sorry, those do not work yet. Working on it")
			break

if __name__ == main():
    main()
