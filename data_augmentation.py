# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:56:10 2020

@author: Alexandre Coll
"""

import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, image, img_to_array, load_img

# Create a folder to save the modified images
folder = "mod_data"
try:
    os.mkdir(folder)
except:
    print("")

# Define the number of images to create per input
images_increased = input("How many images do you get per input? ")

# Define the params of the modified images
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)

# Data path of the original images
data_path = "images"
data_dir_list = os.listdir(data_path)

# Define the shape of the images
width_shape, height_shape = 224, 244

i = 0
num_images = 0

for image_file in data_dir_list:
    img_list = os.listdir(data_path)
    # Image path
    img_path = data_path + "/" + image_file

    imge = load_img(img_path)
    # Resize the image to desired shape
    imge = cv2.resize(
        image.img_to_array(imge),
        (width_shape, height_shape),
        interpolation=cv2.INTER_AREA,
    )
    x = imge / 255
    x = np.expand_dims(x, axis=0)
    t = 1
    # Modify the images with the desired changes
    for output_batch in train_datagen.flow(x, batch_size=1):
        a = image.img_to_array(output_batch[0])
        imagen = output_batch[0, :, :] * 255
        imgfinal = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        cv2.imwrite(folder + "/%i%i.jpg" % (i, t), imgfinal)
        t += 1

        num_images += 1
        if t > images_increased:
            break
    i += 1

print("images generated", num_images)
