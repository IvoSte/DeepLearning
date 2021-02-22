from prep import *


def main():
    arr = import_image("data/train/class_2/black.png")
    image2, resized = generate_image("data/train/class_2/reshape_test.png", arr)

if __name__ == main():
    main()