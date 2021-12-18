import tensorflow as tf
import models
import util

batch_size = 32
img_height = 128
img_width = 128
epochs = 50

data_dir = '/home/maary/文档/project2/'

train_ds, val_ds, num_classes = util.get_dataset(data_dir, batch_size, img_height, img_width)

util.get_sample(train_ds, data_augmentation=util.data_augmentation_layer())

model = tf.keras.Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    # util.data_augmentation_layer(),
    tf.keras.layers.Rescaling(1. / 255),
    models.simple_eight_layers(num_classes)
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
model_name = 'trainingSavedModel'
util.save_model(model, parent_dir, model_name)

util.visualize_history(history, epochs)

