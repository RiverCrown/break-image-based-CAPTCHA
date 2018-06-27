from PIL import Image
import random
import numpy as np
import sys
import json

try:
    json_file = open('params.json', 'r')
    params = json.load(json_file)
except IOError:
    print('找不到配置文件params.json！')
    sys.exit(0)
else:
    json_file.close()

IMAGE_DATA_DIR = params['imageDataDir']


def get_text_and_image(is_random=True, index=0):
    if (is_random):
        index = random.randint(0, 7999)
    text = ''
    imgName = str(index).zfill(4) + '.jpg'
    image = Image.open('/home/rivercrown/try/train-4/' + imgName).convert('L')
    with open('/home/rivercrown/try/train-4/mappings.txt', 'r') as map_file:
        map_content = map_file.readlines()
        text = map_content[index]
        text = text[5:6]
    image = image.resize((160, 60), Image.ANTIALIAS)
    image = np.array(image)
    return text, image


def get_image(is_random=True, index=0):
    if (is_random):
        index = random.randint(0, 7999)
    img_name = str(index).zfill(4) + '.jpg'
    image = Image.open(IMAGE_DATA_DIR + img_name).convert('L')
    image = image.resize((160, 60), Image.ANTIALIAS)
    image = np.array(image)
    return image
