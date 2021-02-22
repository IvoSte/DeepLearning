from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from PIL import Image
import numpy as np


# def load_data():
#     datagen = ImageDataGenerator()
#     # train_it = datagen.flow_from_directory('data/train')
#     # test_it = datagen.flow_from_directory('data/test')
#     # validation_it = datagen.flow_from_directory('data/validation')



# def run_algorithm(data, model):
#     pass

def clear_terminal():
    for _ in range(3):
        print("\n")

def gen_data():
    resized = np.array(Image.open("data/train/class_2/black.png").resize((200,200)))
    for i in range(10):
        Image.fromarray(resized).save("data/train/class_2/black{}.jpg".format(i))
    resized = np.array(Image.open("data/train/class_1/white.png").resize((200,200)))
    for i in range(10):
        Image.fromarray(resized).save("data/train/class_1/white{}.jpg".format(i))

def main():
    # data generator object -- used so we can stream data while training to save memory
    datagen = ImageDataGenerator()
    # iterator over the dataset, sorted in classes == dir name of files
    train_it = datagen.flow_from_directory('data/train', target_size=(200, 200), batch_size= 32, class_mode='binary')
    #test_it = datagen.flow_from_directory('data/test')
    validation_it = datagen.flow_from_directory('data/validation', target_size=(200, 200), batch_size= 32, class_mode='binary')

    # create model
    inception_model = InceptionV3(include_top=False,
                                    weights="imagenet",
                                    input_tensor=None,
                                    input_shape=None,
                                    pooling=None,
                                    classes=2,
                                    classifier_activation="softmax",)
    inception_model.compile()
    inception_model.fit(train_it, steps_per_epoch=10, epochs=50, validation_data=validation_it, validation_steps=100)
    #keras_model.evaluate(validation_it)

if __name__ == main():
    main()