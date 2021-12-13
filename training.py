import tensorflow as tf
import models
import util
from tensorflow.keras.applications import VGG16
batch_size = 32
img_height = 128
img_width = 128
epochs = 50
data_dir = '/home/maary/文档/project2/'

train_ds, val_ds, num_classes = util.get_dataset(data_dir, batch_size, img_height, img_width)

# augmented_ds = train_ds.map(
#     lambda x, y: (data_augmentation(x, training=True), y))

model = tf.keras.Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Rescaling(1. / 255),
    util.data_augmentation_layer(),
    models.simple_thirteen_layers(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

parent_dir = '/home/maary/文档/'
util.save_model(model, parent_dir)

util.visualize_history(history, epochs)
