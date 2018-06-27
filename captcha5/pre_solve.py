from skimage import color, filters, io, segmentation
from skimage.morphology import disk
from skimage.morphology import square
from skimage import morphology
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sys

try:
    json_file = open('params.json', 'r')
    params = json.load(json_file)
except IOError:
    print('找不到配置文件params.json！')
    sys.exit(0)
else:
    json_file.close()

TRAIN_DATA_PATH = params['trainDataDir']
TOTAL_COUNT = params['totalCount']
SAVE_DIR = params['presolvedDataSaveDir']

def remove_dot(image, epoch = 3):
    image_temp = image.copy()
    h, w = image_temp.shape

    for x in range(h):
        image_temp[x, w-1] = 1
        image_temp[x, 0] = 1
    for y in range(w):
        image_temp[0, y] = 1
        image_temp[h-1, y] = 1

    step = 0
    while step < epoch:
        for y in range(1, w-1):
            for x in range(1, h-1):
                count = 0
                if image_temp[x, y-1] > 0.94:
                    count = count + 1
                if image_temp[x, y+1] > 0.94:
                    count = count + 1
                if image_temp[x-1, y] > 0.94:
                    count = count + 1
                if image_temp[x+1, y] > 0.94:
                    count = count + 1
                if image_temp[x+1, y+1] > 0.94:
                    count = count + 1
                if image_temp[x+1, y-1] > 0.94:
                    count = count + 1
                if image_temp[x-1, y+1] > 0.94:
                    count = count + 1
                if image_temp[x-1, y-1] > 0.94:
                    count = count + 1
                if count > 6:
                    image_temp[x, y] = 1
        step += 1
    return image_temp

def split_captcha(image):
    split_img = []
    for i in range(4):
        img = image[:,i*37:(i+1)*37]
        split_img.append(img)
    return split_img

def dilation_and_erosion(img):
    dil = morphology.dilation(img, square(2))
    dil = remove_dot(dil)
    ero = morphology.erosion(dil, square(2))
    return ero

def solve_img():
    for i in range(TOTAL_COUNT):
        img_dir = str(i).zfill(4) + '/'
        if os.path.exists(SAVE_DIR + img_dir) is False:
            os.makedirs(SAVE_DIR + img_dir)
        captcha_name = str(i).zfill(4) + '.jpg'
        captcha = color.rgb2gray(io.imread(TRAIN_DATA_PATH + img_dir + captcha_name))
        captcha = dilation_and_erosion(captcha)
        io.imsave(SAVE_DIR + img_dir + captcha_name, captcha)
        for index in range(9):
            img_name = str(index) + '.jpg'
            img = color.rgb2gray(io.imread(TRAIN_DATA_PATH + img_dir + img_name))
            img = dilation_and_erosion(img)
            io.imsave(SAVE_DIR + img_dir + img_name, img)
                  
solve_img()