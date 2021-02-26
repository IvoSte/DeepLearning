import lenet_new
import data_loading
import train_and_test
import inc_v3
import alexnet

def main():
	# Number of classes
	n_classes = 3
	
	im_dat, im_labels = data_loading.load_data_cv2()
	print("Welcome!")	
	
	while True:
		model_choice = int(input("Please specify which model you would like to test. \n1 = Standard LeNet-5 (32x32)\n2 = LeNet-5 with dropout (32x32)\n3 = LeNet-5 with batch normalization (32x32)\n4 = LeNet-5 but tanh=relu (32x32)\n5 = InceptionV3 (299x299)\n6 = LeNet-5 with double dropout (32x32)\n7 = AlexNet (277x277)\n8 = Exit program  "))
		
		if model_choice == 1:
			width = 32
			height = 32
			reshaped_im = data_loading.prep_data_cv2(im_dat, im_labels, width, height)
			print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
			choice = input("Fill in optimizer: ")
			model = lenet_new.lenet_model(width, height, n_classes)
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(reshaped_im, model, width, height, choice, "categorical_crossentropy", EPOCHS, model_choice)
				print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
				choice = input("Fill in optimizer: ")
		elif model_choice == 2:
			width = 32
			height = 32
			reshaped_im = data_loading.prep_data_cv2(im_dat, im_labels, width, height)
			print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
			choice = input("Fill in optimizer: ")
			model = lenet_new.lenet_model_dropout(width, height, n_classes)
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(reshaped_im, model, width, height, choice, "categorical_crossentropy", EPOCHS, model_choice)
				print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
				choice = input("Fill in optimizer: ")
		elif model_choice == 3:
			width = 32
			height = 32
			reshaped_im = data_loading.prep_data_cv2(im_dat, im_labels, width, height)
			print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
			choice = input("Fill in optimizer: ")
			model = lenet_new.lenet_model_batch_norm(width, height, n_classes)
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(reshaped_im, model, width, height, choice, "categorical_crossentropy", EPOCHS, model_choice)
				print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
				choice = input("Fill in optimizer: ")
		elif model_choice == 4:
			width = 32
			height = 32
			reshaped_im = data_loading.prep_data_cv2(im_dat, im_labels, width, height)
			print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
			choice = input("Fill in optimizer: ")
			model = lenet_new.lenet_model_relu(width, height, n_classes)
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(reshaped_im, model, width, height, choice, "categorical_crossentropy", EPOCHS, model_choice)
				print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
				choice = input("Fill in optimizer: ")
		elif model_choice == 5:
			width = 299
			height = 299
			reshaped_im = data_loading.prep_data_cv2(im_dat, im_labels, width, height)
			print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit InceptionV3 by pressing enter.")
			choice = input("Fill in optimizer: ")
			model = inc_v3.inc_v3_model()
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(reshaped_im, model, width, height, choice, "categorical_crossentropy", EPOCHS, model_choice)
				print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit InceptionV3 by pressing enter.")
				choice = input("Fill in optimizer: ")
		elif model_choice == 6:
			width = 32
			height = 32
			reshaped_im = data_loading.prep_data_cv2(im_dat, im_labels, width, height)
			print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
			choice = input("Fill in optimizer: ")
			model = lenet_new.lenet_model_double_dropout(width, height, n_classes)
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(reshaped_im, model, width, height, choice, "categorical_crossentropy", EPOCHS, model_choice)
				print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
				choice = input("Fill in optimizer: ")
		elif model_choice == 7:
			width = 277
			height = 277
			reshaped_im = data_loading.prep_data_cv2(im_dat, im_labels, width, height)
			print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
			choice = input("Fill in optimizer: ")
			model = alexnet.alex_net()
			while choice:
				EPOCHS = int(input("Please specify the number of epochs. "))
				train_and_test.learning(reshaped_im, model, width, height, choice, "categorical_crossentropy", EPOCHS, model_choice)
				print("\t Please specify which optimizer you would like to test (SGD/Adam/RMSprop), \n\t or quit LeNet by pressing enter.")
				choice = input("Fill in optimizer: ")
		elif model_choice == 8:
			print("Bye bye!")
			break
		else:
			print("Sorry, those do not work yet. Working on it")
			break

if __name__ == main():
    main()
