from skimage import color, io
import numpy as np
import random
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

IMAGE_DATA_DIR = params['presolvedDataSaveDir']

TRAIN_DATA_PATH = '/home/rivercrown/try/train_5_pre/'

def get_one_batch(is_random=True, index=0):
    
    if is_random:
        index = random.randint(0, 7500)

    match = []
    with open('/home/rivercrown/try/train-5/mappings.txt', 'r') as map_file:
        map_content = map_file.readlines()
        label = map_content[index]
        label = label[label.find(',')+1:label.find('\n')]
        for char in label:
            match.append(int(char))

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    labels = match + labels
    labels = np.array(labels)
    
    img_dir = str(index).zfill(4) + '/'
    captcha_name = str(index).zfill(4) + '.jpg'
    captcha_img = color.rgb2gray(io.imread(TRAIN_DATA_PATH + img_dir + captcha_name))

    split_captcha = []
    img_list = []

    for i in range(4):
        img = captcha_img[:,i*37:(i+1)*37]
        split_captcha.append(img)
    for i in range(9):
        imgName = str(i) + '.jpg'
        img = color.rgb2gray(io.imread(TRAIN_DATA_PATH + img_dir + imgName))
        img = img[:,4:41]
        img_list.append(img)
    
    final_img_list = split_captcha + img_list
    final_img_list = np.array(final_img_list)
    final_img_list = final_img_list.reshape((-1, 1665))

    return final_img_list, labels

def get_one_image_batch(is_random=True, index=0):
    
    if is_random:
        index = random.randint(0, 7500)
    
    img_dir = str(index).zfill(4) + '/'
    captcha_name = str(index).zfill(4) + '.jpg'
    captcha_img = color.rgb2gray(io.imread(IMAGE_DATA_DIR + img_dir + captcha_name))

    split_captcha = []
    img_list = []

    for i in range(4):
        img = captcha_img[:,i*37:(i+1)*37]
        split_captcha.append(img)
    for i in range(9):
        imgName = str(i) + '.jpg'
        img = color.rgb2gray(io.imread(IMAGE_DATA_DIR + img_dir + imgName))
        img = img[:,4:41]
        img_list.append(img)
    
    final_img_list = split_captcha + img_list
    final_img_list = np.array(final_img_list)
    final_img_list = final_img_list.reshape((-1, 1665))

    return final_img_list