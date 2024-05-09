import mss
import keyboard as kd
import tkinter as tk
import numpy as np
import tensorflow as tf

# region SETTINGS

kd.add_hotkey('insert', lambda: start_ins())
reg = 1720, 678, 1766, 750
flag = False
gpu = False

namespace = {
    0: ' ', 1: 'й', 2: 'ц', 3: 'у', 4: 'к', 5: 'е', 6: 'н', 7: 'г', 8: 'ш', 9: 'щ',
    10: 'з', 11: 'х', 12: 'ъ', 13: 'ф', 14: 'ы', 15: 'в', 16: 'а', 17: 'п', 18: 'р',
    19: 'о', 20: 'л', 21: 'д', 22: 'ж', 23: 'э', 24: 'я', 25: 'ч', 26: 'с', 27: 'м',
    28: 'и', 29: 'т', 30: 'ь', 31: 'б', 32: 'ю', 33: 'ё',
    34: 'Й', 35: 'Ц', 36: 'У', 37: 'К', 38: 'Е', 39: 'Н', 40: 'Г', 41: 'Ш', 42: 'Щ',
    43: 'З', 44: 'Х', 45: 'Ъ', 46: 'Ф', 47: 'Ы', 48: 'В', 49: 'А', 50: 'П', 51: 'Р',
    52: 'О', 53: 'Л', 54: 'Д', 55: 'Ж', 56: 'Э', 57: 'Я', 58: 'Ч', 59: 'С', 60: 'М',
    61: 'И', 62: 'Т', 63: 'Ь', 64: 'Б', 65: 'Ю', 66: 'Ё',
    67: '.', 68: ',', 69: '!', 70: '@', 71: '#', 72: '$', 73: '%', 74: '^',
    75: '&', 76: '*', 77: '(', 78: ')', 79: '-', 80: '=', 81: '+', 82: ':',
    83: ';', 84: "'", 85: '"', 86: '№', 87: '?', 88: '1', 89: '2', 90: '3',
    91: '4', 92: '5', 93: '6', 94: '8', 95: '9', 96: '0', 97: '7',
}

classifier_all = tf.keras.models.load_model('classifier.h5')
if gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# endregion

# region FUNCTIONS


def start():
    sct = mss.mss()
    while flag:
        scr = np.array(sct.grab(reg))
        if scr.shape[2] == 4:
            scr = scr[:, :, :3]
        image = tf.convert_to_tensor(scr, dtype=tf.uint8)
        image = tf.cast(image, dtype=tf.float32) / 256
        small_image = tf.image.resize(image, (128, 128))
        image_exp = tf.expand_dims(small_image, axis=0)

        # классифицируем
        predictions = classifier_all(image_exp)
        max_probability = np.max(predictions, axis=-1)
        predicted_class = np.argmax(predictions, axis=-1)
        predicted_class_scalar = predicted_class[0] if isinstance(predicted_class, np.ndarray) else predicted_class
        print(f'Класс с самой высокой вероятностью: {predicted_class_scalar} -- {namespace[int(predicted_class_scalar)]}')
        print(f'Максимальная вероятность: {max_probability}\n')
        key = namespace[int(predicted_class_scalar)]
        kd.write(key)

    print('Done')


def start_ins():
    global flag
    if lab.winfo_viewable():
        lab.pack_forget()
    flag = True
    button_stop.pack()
    lab_stop.pack_forget()
    start()


def stop():
    global flag
    flag = False
    button_stop.pack_forget()
    lab_stop.pack()


# endregion

# region TKINTER

main = tk.Tk()

lab = tk.Label(text='Здесь будет кнопка Stop\n'
                    'Начать - Insert', width=35, height=4)
lab_stop = tk.Label(text='Остановленно', width=35, height=4)
button_stop = tk.Button(text='Stop', command=stop, width=35, bg='red', height=4, fg='white')

lab.pack()

# button_start = tk.Button(text='Start', command=start, width=20, bg='teal', height=2)
# button_start.pack()

main.attributes('-topmost', True)
main.geometry('+200+200')
if __name__ == '__main__':
    main.mainloop()

# endregion
