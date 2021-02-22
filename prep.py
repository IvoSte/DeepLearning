import os
import shutil
import numpy as np
from PIL import Image

# print(len(os.listdir("data")))

# def get_image_arrays(src_path, dest_path, color_mode, dim):
#     cls, lab = get_class_labels(src_path)
#     copy_images_to_folders(src_path, dest_path)
#     data = []
#     labels =[]
#     for class_, label in zip(cls, lab):
#         for files in os.listdir(dest_path):
#             dir_name = os.path.basename(str(dest_path +'/'+files))
#             if dir_name.lower() == class_:
#                 for f in os.listdir(dest_path + '/' + dir_name):
#                     if f.split('.')[-1] == 'jpg':
#                             img = Image.open(dest_path +'/'+class_+'/'+f)
#                             img_cs = img.convert(color_mode)
#                             img_resize = img_cs.resize(new_dim)
#                             img_array = np.asarray(img_resize)
#                             data.append(img_array)
#                             labels.append(label)
    
#     img_X = np.array(data)
#     img_y = np.asarray(labels, dtype='uint8')
   
#     print(
#         'Shape of X:\n', img_X.shape,
#         '\nShape of y:\n', len(img_y),
#         '\nPrint label:\n', img_y
#     )
#     return img_X,img_y

# DIM = 100
# COLOR_MODE = 'L'

# img_X, img_y = get_image_arrays(
#     src_path,
#     dest_path,
#     COLOR_MODE,
#     DIM
# )

def import_image(image_path):
    # with Image.open(image_path) as image:
    #     im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    #     print(image.size)
    im_arr = np.array(Image.open(image_path))
    print(im_arr)
    return im_arr

def generate_image(image_path, im_arr):
    image2 = Image.fromarray(im_arr)
    resized = np.array(Image.open("data/train/class_2/test.jpg").resize((200,200)))
    Image.fromarray(resized).save(image_path)
    return image2, resized
