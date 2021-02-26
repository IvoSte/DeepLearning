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
	
	im_dat, im_labels = data_loading.load_data_cv2()
	print("Welcome!")	
	
	while True:
		model_choice = int(input("Please specify which model you would like to test. \n1 = Standard LeNet (28x28)\n2 = InceptionV3 (299x299, does not work yet, don't choose)\n3 = ResNet-50 (does not work yet, don't choose)\n4 = Exit program  "))
		# ~ reshaped_im = data_loading.prep_data_cv2(im_dat, width, height)
		
		if model_choice == 1:
			reshaped_im = data_loading.prep_data_cv2(im_dat, im_labels, 28, 28)
			choice = input("Please specify which optimizer you would like to test. \nFill in optimizer (SGD/Adam/etc by keras): \nOr quit LeNet by pressing enter.  ")
			model = lenet_new.lenet_model(width, height, n_classes)
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(reshaped_im, model, width, height, choice, "categorical_crossentropy", EPOCHS)
				choice = input("Please specify which optimizer you would like to test. \nFill in optimizer (SGD/Adam/etc by keras): \nOr quit LeNet by pressing enter.  ")
		elif model_choice == 4:
			print("Bye bye!")
			break
		else:
			print("Sorry, those do not work yet. Working on it")
			break

if __name__ == main():
    main()
