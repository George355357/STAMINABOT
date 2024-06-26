import mss
import keyboard as kd
from tkinter import messagebox
import numpy as np
import os
import time as tm
import xml.etree.ElementTree as ET
from PIL import Image

kd.add_hotkey('insert', lambda: start())
path = "c:/users/admin/images/"
reg = 1720, 678, 1766, 750

namespace = {
    0: 'space', 1: 'j', 2: 'c', 3: 'u', 4: 'k', 5: 'e', 6: 'n', 7: 'g', 8: 'sh', 9: 'sht',
    10: 'z', 11: 'h', 12: 'hard_sign', 13: 'f', 14: 'y', 15: 'v', 16: 'a', 17: 'p', 18: 'r',
    19: 'o', 20: 'l', 21: 'd', 22: 'zh', 23: 'je', 24: 'ja', 25: 'ch', 26: 's', 27: 'm',
    28: 'i', 29: 't', 30: 'easy_sign', 31: 'b', 32: 'ju', 33: 'jo',
    34: 'J', 35: 'C', 36: 'U', 37: 'K', 38: 'E', 39: 'N', 40: 'G', 41: 'SH', 42: 'SHT',
    43: 'Z', 44: 'H', 45: 'HARD_SIGN', 46: 'F', 47: 'Y', 48: 'V', 49: 'A', 50: 'P', 51: 'R',
    52: 'O', 53: 'L', 54: 'D', 55: 'ZH', 56: 'JE', 57: 'JA', 58: 'CH', 59: 'S', 60: 'M',
    61: 'I', 62: 'T', 63: 'EASY_SIGN', 64: 'B', 65: 'JU', 66: 'JO',
    67: '.', 68: ',', 69: '!', 70: '@', 71: '#', 72: '$', 73: '%', 74: '^',
    75: '&', 76: '*', 77: '(', 78: ')', 79: '-', 80: '=', 81: '+',
    82: 'back_slesh', 83: 'mid_slesh', 84: 'slesh', 85: ':',
    86: ';', 87: "'", 88: '"', 89: '№', 90: '?', 91: '1', 92: '2', 93: '3',
    94: '4', 95: '5', 96: '6', 97: '7', 98: '8', 99: '9', 100: '0',
}

namespace_write = {
    1: 'й ', 2: 'ц ', 3: 'у ', 4: 'к ', 5: 'е ', 6: 'н ', 7: 'г ', 8: 'ш ', 9: 'щ ',
    10: 'з ', 11: 'х ', 12: 'ъ ', 13: 'ф ', 14: 'ы ', 15: 'в ', 16: 'а ', 17: 'п ', 18: 'р ',
    19: 'о ', 20: 'л ', 21: 'д ', 22: 'ж ', 23: 'э ', 24: 'я ', 25: 'ч ', 26: 'с ', 27: 'м ',
    28: 'и ', 29: 'т ', 30: 'ь ', 31: 'б ', 32: 'ю ', 33: 'ё ',
    34: 'Й ', 35: 'Ц ', 36: 'У ', 37: 'К ', 38: 'Е ', 39: 'Н ', 40: 'Г ', 41: 'Ш ', 42: 'Щ ',
    43: 'З ', 44: 'Х ', 45: 'Ъ ', 46: 'Ф ', 47: 'Ы ', 48: 'В ', 49: 'А ', 50: 'П ', 51: 'Р ',
    52: 'О ', 53: 'Л ', 54: 'Д ', 55: 'Ж ', 56: 'Э ', 57: 'Я ', 58: 'Ч ', 59: 'С ', 60: 'М ',
    61: 'И ', 62: 'Т ', 63: 'Ь ', 64: 'Б ', 65: 'Ю ', 66: 'Ё ',
    67: '. ', 68: ', ', 69: '! ', 70: '@ ', 71: '# ', 72: '$ ', 73: '% ', 74: '^ ',
    75: '& ', 76: '* ', 77: '( ', 78: ') ', 79: '- ', 80: '= ', 81: '+ ',
    82: '\ ', 83: '| ', 84: '/ ', 85: ': ',
    86: '; ', 87: "' ", 88: '" ', 89: '№ ', 90: '? ', 91: '1 ', 92: '2 ', 93: '3 ',
    94: '4 ', 95: '5 ', 96: '6 ', 97: '7 ', 98: '8 ', 99: '9 ', 100: '0 ',
}


def start():
    st = tm.time()
    sct = mss.mss()
    # png_files = [file for file in os.listdir(path) if file.endswith('.png')]
    # cnt = len(png_files)
    cnt = 0
    for class_name in range(1, 101):
        for _ in range(5):
            cnt += 1
            scr = np.array(sct.grab(reg))
            image = Image.fromarray(scr)
            image.save(f'{path}img{cnt}.png')
        kd.write(namespace_write[class_name])
    png_cnt = 0
    name_class = 0
    for _ in range(1, 101):
        name_class += 1
        for _ in range(5):
            png_cnt += 1
            name = f'img{png_cnt}'
            p = ET.Element('annotation')
            c = ET.SubElement(p, 'folder')
            c.text = 'images'
            c = ET.SubElement(p, 'filename')
            c.text = name + '.png'
            c = ET.SubElement(p, 'path')
            c.text = path + name + '.png'
            c = ET.SubElement(p, 'sourse')
            c = ET.SubElement(c, 'database')
            c = ET.SubElement(p, 'size')
            s = ET.SubElement(c, 'width')
            s.text = '46'
            s = ET.SubElement(c, 'height')
            s.text = '72'
            s = ET.SubElement(c, 'depth')
            s.text = '3'
            c = ET.SubElement(p, 'segmented')
            c.text = '0'
            o = ET.SubElement(p, 'object')
            n = ET.SubElement(o, 'name')
            n.text = f'{namespace[name_class]}'
            c = ET.SubElement(o, 'pose')
            c.text = 'un'
            c = ET.SubElement(o, 'truncated')
            c.text = '1'
            c = ET.SubElement(o, 'difficult')
            c.text = '0'
            bb = ET.SubElement(o, 'bndbox')
            xmin = 1
            xmax = 46
            ymin = 1
            ymax = 72
            xmin_ = ET.SubElement(bb, 'xmin')
            xmin_.text = str(xmin)
            ymin_ = ET.SubElement(bb, 'ymin')
            ymin_.text = str(ymin)
            xmax_ = ET.SubElement(bb, 'xmax')
            xmax_.text = str(xmax)
            ymax_ = ET.SubElement(bb, 'ymax')
            ymax_.text = str(ymax)
            tree = ET.ElementTree(p)
            ET.indent(tree, '  ')
            tree.write(path + name + '.xml', encoding="utf-8")
    fin = tm.time()
    messagebox.showinfo('done', message='Выполненно за\n'
                                        f'{fin - st} секунд')


while True:
    pass
