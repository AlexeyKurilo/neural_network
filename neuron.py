import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

matplotlib.use('TkAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# отображение первых 25 изображений из обучающей выборки
# plt.figure(figsize=(10,5))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#
# plt.show()

###Create model
def make_model():
    image_size = 28
    num_channels = 1
    num_classes = 10
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     padding='same',
                     input_shape=(image_size, image_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# model = make_model()
# print(model.summary())

def train_model(model):
    (train_digits, train_labels), (test_digits, test_labels) = mnist.load_data()

    # Get image size
    image_size = 28
    num_channels = 1
    num_classes = 10

    train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
    train_data = train_data.astype('float32') / 255.0
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)
    val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

    # Start training the network
    model.fit(train_data, train_labels_cat, epochs=5, batch_size=32,  ###########################################
              validation_data=(val_data, val_labels_cat))

    print('-------------------------------------------------------------------------')
    model.evaluate(val_data, val_labels_cat)

    return model


# ###Convert image
def digits_predict(model, image_file):
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size),
                                             color_mode='grayscale', interpolation='box')
    img_arr = keras.preprocessing.image.img_to_array(img)
    img_arr = 1 - img_arr / 255.0
    img_arr = np.array([img_arr])

    result = model.predict_classes([img_arr])
    return result[0]


def show(image_file):
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size),
                                             color_mode='grayscale', interpolation='box')
    # img_arr = np.expand_dims(img, axis=0)
    # img_arr = 1 - img_arr / 255.0
    # img_arr = img_arr.reshape((-1, 28, 28, 1))
    img_arr = keras.preprocessing.image.img_to_array(img)
    img_arr = 1 - img_arr / 255.0
    # img_arr = np.array([img_arr])

    fig, ax = plt.subplots()

    ax.imshow(img_arr)

    fig.set_figwidth(5)  # ширина и
    fig.set_figheight(5)  # высота "Figure"
    plt.show()
    # print(img_arr)
    # print('--------------------------')
    # print(img_arr2)


# Train model
# train_model(model)
# model.save('3digits_28x28.h5')
##

# train_data
# val_data
# train_labels_cat
# val_labels_cat

# def test(n):
#     model = tf.keras.models.load_model('1digits_28x28.h5')
#     x = np.expand_dims(x_test[n], axis=0)
#     res = model.predict(x)
#     # print( res )
#     print(np.argmax(res))
#
#     plt.imshow(x_test[n], cmap=plt.cm.binary)
#     plt.show()

# test(15)
def start(filename):
    model = tf.keras.models.load_model('model/1digits_28x28.h5')
    result = digits_predict(model, filename)
    print(result)
    return result


def start1(filename):
    return filename


# start('image/2_28x28.png')
# model = tf.keras.models.load_model('model/1digits_28x28.h5')
# print(digits_predict(model, '0.png'))
# print(digits_predict(model, 'image/1.png'))
# print(digits_predict(model, '2.png'))
# print(digits_predict(model, '3.png')) +
# print(digits_predict(model, '4.png')) +
# print(digits_predict(model, '5.png'))
# print(digits_predict(model, '6.png'))
# print(digits_predict(model, '7.png'))
# print(digits_predict(model, '8.png'))
# print(digits_predict(model, '9.png'))
# print(digits_predict(model, '2_28x28.png')) +
# print(digits_predict(model, '2_depth3.png'))
# print(digits_predict(model, '5_774_459.png')) --1
# print(digits_predict(model, '7_226_199.png')) +
# print(digits_predict(model, '5_300_300.png')) +
# print(digits_predict(model, '2_700_735png.png'))
# show('2_700_735png.png')

# show('2.png')
