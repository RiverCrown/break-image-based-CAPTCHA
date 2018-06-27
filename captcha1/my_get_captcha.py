from PIL import Image
import random
import numpy as np
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

MAPPING_DIR = '/home/rivercrown/try/train/'
IMAGE_DATA_DIR = params['presolvedDataSaveDir']
MAPPING_FILE_NAME = 'mappings.txt'


def get_text_and_image(is_random=True, index=0):
    if is_random:
        index = random.randint(0, 7499)
    text = ''
    with open(MAPPING_DIR + MAPPING_FILE_NAME, 'r') as map_file:
        map_content = map_file.readlines()
        text = map_content[index]
        start = text.find(',') + 1
        end = text.find('=')
        text = text[start:end]
        pos = random.randint(0, len(text) - 1)
        text = text[pos]
    imgName = str(index).zfill(4) + '_' + str(pos) + '.jpg'
    image = Image.open(IMAGE_DATA_DIR + imgName).convert('L')
    image = np.array(image)
    return text, image


def get_ans_and_image(is_random=True, index=0):
    if is_random:
        index = random.randint(0, 7499)
    ans = ''
    with open(MAPPING_DIR + MAPPING_FILE_NAME, 'r') as map_file:
        map_content = map_file.readlines()
        ans = map_content[index]
        start = ans.find(',') + 1
        end = ans.find('\n')
        fake_end = ans.find('=')
        ans = ans[start:end]

    img_list = []

    char_len = fake_end - start
    for i in range(char_len):
        img_name = str(index).zfill(4) + '_' + str(i) + '.jpg'
        img = Image.open(IMAGE_DATA_DIR + img_name).convert('L')
        img = np.array(img)
        img_list.append(img)

    return ans, img_list, char_len


def get_image(is_random=True, index=0):
    if is_random:
        index = random.randint(0, 7499)

    with open('split_record.txt', 'r') as split_record:
        content = split_record.readlines()
        split_record_str = content[index]
        start = split_record_str.find(',') + 1
        end = split_record_str.find('\n')
        char_len = int(split_record_str[start:end])

    img_list = []

    for i in range(char_len):
        img_name = str(index).zfill(4) + '_' + str(i) + '.jpg'
        img = Image.open(IMAGE_DATA_DIR + img_name).convert('L')
        img = np.array(img)
        img_list.append(img)

    return img_list, char_len
