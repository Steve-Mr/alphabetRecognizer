import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimage

model_path = '/home/maary/文档/savedModel'
image_path = '/home/maary/文档/project2/Sample014/img014-00011.png'

model = load_model(model_path)



test_image = image.load_img(image_path, target_size=(128, 128))
plt.imshow(test_image)
plt.show()
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
print(test_image.shape)

result = model.predict(test_image)
print(result)
predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)
