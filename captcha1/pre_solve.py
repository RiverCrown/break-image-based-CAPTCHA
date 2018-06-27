from skimage import color, filters, io
from skimage.morphology import disk
from skimage.morphology import square
from skimage import restoration
from skimage import morphology
from skimage import transform
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

TRAIN_DATA_PATH = params['trainDataDir']
TOTAL_COUNT = params['totalCount']
SAVE_DIR = params['presolvedDataSaveDir']

print('原图片地址', TRAIN_DATA_PATH)
print('图片总数', TOTAL_COUNT)
print('预处理图片存放地址', SAVE_DIR)


def remove_dot(image, epoch=3):
    image_temp = image.copy()
    h, w = image_temp.shape

    for x in range(h):
        image_temp[x, w - 1] = 1
        image_temp[x, 0] = 1
    for y in range(w):
        image_temp[0, y] = 1
        image_temp[h - 1, y] = 1

    step = 0
    while step < epoch:
        for y in range(1, w - 1):
            for x in range(1, h - 1):
                count = 0
                if image_temp[x, y - 1] == 1:
                    count = count + 1
                if image_temp[x, y + 1] == 1:
                    count = count + 1
                if image_temp[x - 1, y] == 1:
                    count = count + 1
                if image_temp[x + 1, y] == 1:
                    count = count + 1
                if image_temp[x + 1, y + 1] == 1:
                    count = count + 1
                if image_temp[x + 1, y - 1] == 1:
                    count = count + 1
                if image_temp[x - 1, y + 1] == 1:
                    count = count + 1
                if image_temp[x - 1, y - 1] == 1:
                    count = count + 1
                if count > 4:
                    image_temp[x, y] = 1
        step += 1
    return image_temp


def split_image(img):
    h, w = img.shape
    ver_list = []
    for x in range(w):
        black = 0
        for y in range(h):
            if img[y, x] == 0:
                black += 1
        ver_list.append(black)
    l, r = 0, 0
    flag = False
    cuts = []
    for i, count in enumerate(ver_list):
        if flag is False and count > 0:
            l = i
            flag = True
        if flag and count < h * 0.05:
            r = i - 1
            flag = False
            cuts.append((l, r))

    return cuts


index = 0
split_record = open('split_record.txt', 'w')
h, w = color.rgb2gray(io.imread(TRAIN_DATA_PATH + '0000.jpg')).shape
while index < TOTAL_COUNT:

    img_name = str(index).zfill(4) + '.jpg'
    img = color.rgb2gray(io.imread(TRAIN_DATA_PATH + img_name))
    img = transform.resize(img, (80, 350))

    threshold = filters.threshold_otsu(img)
    new_image = (img > threshold) * 1.0

    new_image = remove_dot(new_image)
    # test = filters.median(new_image, disk(3))
    new_image = morphology.dilation(new_image, square(4))
    new_image = morphology.erosion(new_image, square(4))

    cuts = split_image(new_image)
    img_dir = str(index).zfill(4) + '_'
    split_order = 0
    for l, r in cuts:
        split_width = r - l + 1
        if split_width <= w * 0.06:
            continue
        if (h - split_width) % 2 == 1:
            left_pixel = int((h - split_width) / 2)
            right_pixel = int((h - split_width) / 2) + 1
        else:
            left_pixel = right_pixel = int((h - split_width) / 2)
        left_blank = np.full((h, left_pixel), 1, np.uint8)
        right_blank = np.full((h, right_pixel), 1, np.uint8)
        final_img = np.concatenate((left_blank, new_image[:, l:r], right_blank), axis=1)
        split_img_name = str(split_order) + '.jpg'
        io.imsave(SAVE_DIR + img_dir + split_img_name, final_img)
        split_order += 1

    split_record_str = str(index).zfill(4) + ',' + str(split_order) + '\n'
    split_record.write(split_record_str)

    index += 1

split_record.close()
