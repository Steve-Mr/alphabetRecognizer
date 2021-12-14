import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimage

import util

batch_size = 32
img_height = 128
img_width = 128
epochs = 15

model_path = '/home/maary/文档/transferSavedModel'
image_path = '/home/maary/文档/project2'
bonus_path = '/home/maary/文档/Bonus'

model = load_model(model_path)
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

train_ds, val_ds, num_classes = util.get_dataset(image_path, batch_size, img_height, img_width)

results = model.evaluate(val_ds, batch_size=batch_size)
print("project2")
print("test loss, test acc:", results)

bonus_val_ds = keras.utils.image_dataset_from_directory(
        bonus_path,
        image_size=(img_height, img_width),
        batch_size=batch_size)
bonus_results = model.evaluate(bonus_val_ds, batch_size=batch_size)
print("bonus")
print("test loss, test acc:", bonus_results)
