import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Model
from keras.models import load_model

import util

pretrained_model_path = '/home/maary/文档/savedModel'
batch_size = 8
img_height = 128
img_width = 128
epochs = 15

data_dir = '/home/maary/文档/Bonus/'

model = load_model(pretrained_model_path)
model.summary()

model = model.get_layer('sequential_1')
model.summary()
model = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_4').output)
model.summary()

train_ds, val_ds, num_classes = util.get_dataset(data_dir, batch_size, img_height, img_width)

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
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds
)

parent_dir = '/home/maary/文档/'
model_name = 'transferSavedModel'
util.save_model(model, parent_dir, model_name)

util.visualize_history(history, epochs)

