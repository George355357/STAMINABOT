import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
import matplotlib.pyplot as plt
import numpy as np

gpu = False
train = True
read_data = True
save = True
load = True
try_to = True

epochs = 100

# region MODEL

base_model = tf.keras.applications.MobileNetV2(weights='imagenet',
                                                   include_top=False,
                                                   input_shape=(128,128, 3))
base_model.trainable = False
inputs = Input((128, 128, 3))
x = base_model(inputs)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(101, activation = 'softmax')(x)
outputs = x
classifier = keras.Model(inputs, outputs)


class Model(tf.keras.Model):
    def __init__(self, nn):
        super(Model, self).__init__()
        self.nn = nn
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    def get_loss(self, y, preds):
        loss = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y, 101), preds)
        return loss

    @tf.function
    def training_step(self, x, y):
        with tf.GradientTape() as tape:
            preds = self.nn(x)
            loss = self.get_loss(y, preds)

        gradients = tape.gradient(loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))
        return tf.reduce_mean(loss)


model = Model(classifier)

# endregion

if load:
    model.nn.load_weights('classifier.h5')
    print('Loaded')

if gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(gpus)

if read_data:
    def parse_record(record):
        feature_description = {
            'img': tf.io.FixedLenFeature([], tf.string),
            'name': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_record = tf.io.parse_single_example(record, feature_description)
        img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
        name = tf.io.parse_tensor(parsed_record['name'], out_type=tf.int32)
        return img, name


    dataset = tf.data.TFRecordDataset('dataset.tfrecord')
    dataset = dataset.map(parse_record)
    dataset = dataset.shuffle(50).cache().prefetch(buffer_size=tf.data.AUTOTUNE).batch(64).shuffle(50)

    for i, n in dataset.take(1):
        plt.figure(figsize=(10, 6))
        i = i.numpy()
        n = n.numpy()
        for nn in range(50):
            ax = plt.subplot(5, 10, 1 + nn)
            plt.title(n[nn])
            plt.imshow(i[nn])
            plt.axis('off')
        plt.show()

if train:
    for i, c in dataset.take(1):
        print("Ошибка сейчас -", tf.reduce_mean(model.training_step(i, c)))

    for epoch in range(1, epochs + 1):
        loss = 0
        for step, (i, n) in enumerate(dataset):
            loss+=tf.reduce_mean(model.training_step(i,n))
        print(f'{epoch} эпоха, ошибка = {loss}')
    print('Trained')


if try_to:
    def try_pred():
        n = 50
        images_per_row = 10
        num_rows = n // images_per_row + (n % images_per_row > 0)

        for images, _ in dataset.take(1):
            preds = model.nn(images)
            fig, axes = plt.subplots(num_rows, images_per_row, figsize=(10, 6))
            axes = axes.flatten() if num_rows > 1 else [axes]

            for i in range(n):
                ax = axes[i]
                ax.imshow(images[i], cmap='gist_gray')
                pred = preds[i]
                max_pred_index = np.argmax(pred)
                max_pred_value = pred[max_pred_index]
                ax.set_title(f"{max_pred_index} {max_pred_value:.3f}")
                ax.axis('off')

            plt.tight_layout()
            plt.show()
    try_pred()

if save:
    model.nn.save('classifier.h5')
    print('Saved')
