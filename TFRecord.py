import tensorflow as tf
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import random

namespace = {
    'space': 0, 'j': 1, 'c': 2, 'u': 3, 'k': 4, 'e': 5, 'n': 6, 'g': 7, 'sh': 8, 'sht': 9,
    'z': 10, 'h': 11, 'hard_sign': 12, 'f': 13, 'y': 14, 'v': 15, 'a': 16, 'p': 17, 'r': 18,
    'o': 19, 'l': 20, 'd': 21, 'zh': 22, 'je': 23, 'ja': 24, 'ch': 25, 's': 26, 'm': 27,
    'i': 28, 't': 29, 'easy_sign': 30, 'b': 31, 'ju': 32, 'jo': 33,
    'J': 34, 'C': 35, 'U': 36, 'K': 37, 'E': 38, 'N': 39, 'G': 40, 'SH': 41, 'SHT': 42,
    'Z': 43, 'H': 44, 'HARD_SIGN': 45, 'F': 46, 'Y': 47, 'V': 48, 'A': 49, 'P': 50, 'R': 51,
    'O': 52, 'L': 53, 'D': 54, 'ZH': 55, 'JE': 56, 'JA': 57, 'CH': 58, 'S': 59, 'M': 60,
    'I': 61, 'T': 62, 'EASY_SIGN': 63, 'B': 64, 'JU': 65, 'JO': 66,
    '.': 67, ',': 68, '!': 69, '@': 70, '#': 71, '$': 72, '%': 73, '^': 74,
    '&': 75, '*': 76, '(': 77, ')': 78, '-': 79, '=': 80, '+': 81,
    'back_slesh': 82, 'mid_slesh': 83, 'slesh': 84, ':': 85,
    ';': 86, "'": 87, '"': 88, '№': 89, '?': 90, '1': 91, '2': 92, '3': 93,
    '4': 94, '5': 95, '6': 96, '7': 97, '8': 98, '9': 99, '0': 100,
}

fn = "C:/Users/Admin/images/"

p = [fn + '/' + f for f in listdir(fn) if isfile(join(fn, f)) and f[-1] == 'l']


def load_img(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 256
    img = tf.image.resize(img, (1024, 1024))
    return img


writer = tf.io.TFRecordWriter('dataset.tfrecord')


def saveinrecord(img, name):
    serialized_img = tf.io.serialize_tensor(img).numpy()
    serialized_name = tf.io.serialize_tensor(name).numpy()
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img])),
        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_name]))
    }))
    writer.write(example.SerializeToString())


for xml in p:
    tree = ET.parse(xml)  # адрес файла
    root = tree.getroot()  # парсим
    num_objects = len(root) - 6
    w = int(root[4][0].text)  # ширина x
    h = int(root[4][1].text)  # высота y
    img = load_img(root[2].text)
    for num in range(num_objects):
        xmin = tf.clip_by_value(int(int(root[num + 6][4][0].text) / w * 1024), 0, 1024)
        ymin = tf.clip_by_value(int(int(root[num + 6][4][1].text) / h * 1024), 0, 1024)
        xmax = tf.clip_by_value(int(int(root[num + 6][4][2].text) / w * 1024), 0, 1024)
        ymax = tf.clip_by_value(int(int(root[num + 6][4][3].text) / h * 1024), 0, 1024)
        offset_height = ymin
        offset_width = xmin
        target_height = ymax - ymin
        target_width = xmax - xmin
        cropped = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
        cropped = tf.image.resize(cropped, (128, 128))
        name = namespace[root[num + 6][0].text]
        saveinrecord(cropped, name)

counter = 0
goal = 5

while counter < goal:
    gxmin = random.randint(1, 900)
    gymin = random.randint(1, 900)
    gxsize = random.randint(10, 100)
    gysize = random.randint(10, 100)
    gxmax = gxmin + gxsize
    gymax = gymin + gysize
    notintersect = True
    for num in range(num_objects):
        xmin = tf.clip_by_value(int(int(root[num + 6][4][0].text) / w * 1024), 0, 1024)
        ymin = tf.clip_by_value(int(int(root[num + 6][4][1].text) / h * 1024), 0, 1024)
        xmax = tf.clip_by_value(int(int(root[num + 6][4][2].text) / w * 1024), 0, 1024)
        ymax = tf.clip_by_value(int(int(root[num + 6][4][3].text) / h * 1024), 0, 1024)
        x_overlap = tf.maximum(0, tf.minimum(gxmax, xmax) - tf.maximum(gxmin, xmin))
        y_overlap = tf.maximum(0, tf.minimum(gymax, ymax) - tf.maximum(gymin, ymin))
        if x_overlap > 0 and y_overlap > 0:
            notintersect = False
            break

    if notintersect:
        cropped = tf.image.crop_to_bounding_box(img, gymin, gxmin, gysize, gxsize)
        cropped = tf.image.resize(cropped, (128, 128))
        name = 0
        saveinrecord(cropped, name)
        counter += 1

writer.close()
print('Done')
