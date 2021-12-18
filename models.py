import tensorflow as tf
from tensorflow.keras import layers

import util
from tensorflow.python.keras.engine import training


def simple_six_layers(num_classes):
    return tf.keras.Sequential([
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])


def leNet_inspired(num_classes):
    return tf.keras.Sequential([
        layers.Conv2D(8, 3, activation='relu'),
        layers.AveragePooling2D(),

        layers.Conv2D(16, 3, activation='relu'),
        layers.AveragePooling2D(),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])


def simple_eight_layers(num_classes):
    return tf.keras.Sequential([
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])


def vgg_inspired(num_classes):
    return tf.keras.Sequential([
        layers.Conv2D(8, 3, activation='relu', padding='same'),
        layers.Conv2D(8, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])


def squeezenet(num_classes):
    bnmomemtum = 0.9
    x = tf.keras.Input(shape=(128, 128, 3))
    x1 = tf.keras.layers.Rescaling(1. / 255)(x)
    x2 = util.data_augmentation_layer()(x1)

    def fire(xn, squeeze, expand):
        yn = layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(xn)
        yn = layers.BatchNormalization(momentum=bnmomemtum)(y)
        y1 = layers.Conv2D(filters=expand // 2, kernel_size=1, activation='relu', padding='same')(yn)
        y1 = layers.BatchNormalization(momentum=bnmomemtum)(y1)
        y3 = layers.Conv2D(filters=expand // 2, kernel_size=3, activation='relu', padding='same')(yn)
        y3 = layers.BatchNormalization(momentum=bnmomemtum)(y3)
        return tf.keras.layers.concatenate([y1, y3])

    def fire_module(squeeze, expand):
        return lambda xn: fire(x, squeeze, expand)

    y = layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(x2)
    y = layers.BatchNormalization(momentum=bnmomemtum)(y)
    y = fire_module(24, 48)(y)
    y = layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(48, 96)(y)
    y = layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(64, 128)(y)
    y = layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(48, 96)(y)
    y = layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(24, 48)(y)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dense(num_classes, activation='softmax')(y)

    return tf.keras.Model(x, y)


def VGG16(num_classes):
    img_input = tf.keras.Input(shape=(128, 128, 3))
    # Block 1
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)

    x = layers.Dense(num_classes, activation='relu',
                     name='predictions')(x)
    # Create model.
    model = training.Model(img_input, x, name='vgg16')
