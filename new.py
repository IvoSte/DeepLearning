from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from PIL import Image
import numpy as np


def clear_terminal():
    for _ in range(3):
        print("\n")

def gen_data():
    resized = np.array(Image.open("data/train/black/black.png").resize((200,200)))
    for i in range(10):
        Image.fromarray(resized).save("data/train/black/black{}.jpg".format(i))
    resized = np.array(Image.open("data/train/white/white.png").resize((200,200)))
    for i in range(10):
        Image.fromarray(resized).save("data/train/white/white{}.jpg".format(i))

def main():
    # data generator object -- used so we can stream data while training to save memory
    datagen = ImageDataGenerator()

    # iterator over the dataset, sorted in classes == dir name of files
    train_it = datagen.flow_from_directory('data/train', batch_size= 5, class_mode='binary')
    #test_it = datagen.flow_from_directory('data/test')
    validation_it = datagen.flow_from_directory('data/validation', batch_size= 5, class_mode='binary')

    # create model
    model = keras.models.Sequential()
    inception_model = InceptionV3(include_top=False,
                                    weights="imagenet",
                                    pooling=None)

    model.add(inception_model)
    model.add(keras.layers.Dense(2))

    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(train_it, steps_per_epoch=1, epochs=10, validation_data=validation_it, validation_steps=2)
    #keras_model.evaluate(validation_it)

if __name__ == main():
    main()
