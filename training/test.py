import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os

from echotrain.utils.handling_yaml import load_config_file
from echotrain.training.trainer_base import TrainerBase

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[:5000] / 255.0
y_train = to_categorical(y_train[:5000])

x_test = x_test[:1000] / 255.0
y_test = to_categorical(y_test[:1000])

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test :", x_test.shape)
print("y_test :", y_test.shape)


def train_generator(batch_size):
    datagen_train = ImageDataGenerator(
        width_shift_range=3 / 32,
        height_shift_range=3 / 32,
        brightness_range=(0.3, 1.8),
        horizontal_flip=True,
    )
    train_gen = datagen_train.flow(x_train, y_train, batch_size=batch_size)
    return train_gen


def val_generatior(batch_size):
    datagen_validation = ImageDataGenerator(
        width_shift_range=3 / 32,
        height_shift_range=3 / 32,
        brightness_range=(0.3, 1.8),
        horizontal_flip=True,
    )

    val_gen = datagen_validation.flow(x_test, y_test, batch_size=batch_size)
    return val_gen


def create_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', ))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', ))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', ))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', ))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = create_model()

    batch_size = 100
    train_gen = train_generator(batch_size)
    val_gen = val_generatior(batch_size)
    train_steps = int((x_train.shape)[0] / batch_size)
    val_steps = int((x_test.shape)[0] / batch_size)
    model_name = 'unet'
    base_addr = f'Experience/{model_name}'
    config = load_config_file('../config/config_example.yaml')
    trainer = TrainerBase(base_addr, config.trainer)
    # trainer._train(model, train_data_gen=train_gen, val_data_gen=val_gen,
    #                n_iter_train=train_steps, n_iter_val=val_steps)
    trainer._export()
