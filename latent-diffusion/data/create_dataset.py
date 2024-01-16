import numpy as np
import cv2
import os

image_dir = "../../DDGAN/data/celebahq256_imgs/"
train = "train/train"
val = "valid/valid"

save_dir = "data/celebahq"

for image_name in os.listdir(os.path.join(image_dir, val)):
    img = cv2.imread(os.path.join(image_dir, train, image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_name = image_name.split('.')[0][4:]
    if len(im_name) < 2:
        im_name = '0000' + im_name
    if len(im_name) < 3:
        im_name = '000' + im_name
    if len(im_name) < 4:
        im_name = '00' + im_name
    if len(im_name) < 5:
        im_name = '0' + im_name

    npy_name = 'imgHQ' + im_name + '.npy'
    np.save((os.path.join(save_dir, npy_name)), img)
