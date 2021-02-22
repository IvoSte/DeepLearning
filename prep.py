import os
import shutil
import numpy as np
from PIL import Image

# Import an image from path, print specs, show image
def open_print_specs(im_path, show_im):
    im = Image.open(im_path)
    print("Format:", im.format, 
            "\n Mode:", im.mode, 
            "\n Size:", im.size)
    if show_im:
        im.show()
    
    im_arr = np.array(im)
    print("Pixel intensities: \n", im_arr)
    return im_arr
    
# Get array of pixels from image path
def get_arr_image(image_path):
    # with Image.open(image_path) as image:
    #     im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    #     print(image.size)
    im_arr = np.array(Image.open(image_path))
    print(im_arr)
    return im_arr

# Generate image from array version of test.jpg, save to given image path
# Also, generate image from given array. Return both
def generate_image(image_path, im_arr):
    image2 = Image.fromarray(im_arr)
    resized = np.array(Image.open("data/train/class_2/test.jpg").resize((200,200)))
    Image.fromarray(resized).save(image_path)

    return image2, resized

# Normalizes pixel array with a bunch of checks
def norm_pixels(pixels):
    # confirm pixel range is 0-255
    print('Data Type: %s' % pixels.dtype)
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # confirm the normalization
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    return pixels

# Bit of scratch main code
im_arr = open_print_specs("data/train/class_2/gif_first_frame_11aa (1).jpg", False)
pix = norm_pixels(im_arr)
print(pix)