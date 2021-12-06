import tensorflow as tf
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt

import numpy as np

batch_size = 32
img_height = 128
img_width = 128
epochs = 50
data_dir = '/home/maary/文档/project2/'
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(train_ds.class_names)
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(buffer_size=1024).prefetch(buffer_size=AUTOTUNE)
# train_ds = train_ds.shuffle(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(
            0.3,
            fill_mode='nearest',
            interpolation='nearest'),
    ]
)
#
# augmented_ds = train_ds.map(
#     lambda x, y: (data_augmentation(x, training=True), y))

# plt.figure(figsize=(10, 10))
# for images, labels in augmented_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         print(images[i].shape)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    data_augmentation,
    # tf.keras.layers.Rescaling(1. / 255),

    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

parent_dir = '/home/maary/文档/'
path = os.path.join(parent_dir, 'savedModel')
try:
    os.mkdir(path)
except OSError as error:
    print(error)

model.save(path, include_optimizer=False)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
