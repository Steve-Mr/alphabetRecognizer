import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import load_model

import util

pretrained_model_path = '/home/maary/文档/savedModel'
batch_size = 8
img_height = 128
img_width = 128
epochs = 15

data_dir = '/home/maary/文档/Bonus/'
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

model = load_model(pretrained_model_path)
model.summary()

# model.get_layer('sequential_1').summary()
model = model.get_layer('sequential_1')
model.summary()
model = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_4').output)
model.summary()

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
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

feature_extractor = hub.KerasLayer(model, input_shape=(img_height, img_width, 3))
feature_extractor.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    feature_extractor,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds
)

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

