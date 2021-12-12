import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


def get_dataset(data_dir, batch_size, img_height, img_width):
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
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, num_classes


def data_augmentation_layer():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(
                0.3,
                fill_mode='nearest',
                interpolation='nearest'),
        ]
    )


def get_sample_image(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            print(images[i].shape)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(dataset.class_names[labels[i]])
            plt.axis("off")
    plt.show()


def save_model(model, parent_dir):
    path = os.path.join(parent_dir, 'savedModel')
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    model.save(path, include_optimizer=False)


def visualize_history(history, epochs):
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
