import tensorflow as tf
# import tensorflow.compat.v1 as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
# from keras.utils import layer_utils
from keras.applications.imagenet_utils import preprocess_input
from keras .utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.preprocessing import image
import pydot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import resnets_utils

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()

X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T
model = load_model("ResNet50.h5")
# print(model.predict(X_test))
# print(model.predict(X_test[0:5]))
# preds = model.evaluate(X_test, Y_test)
# print("误差值 = " + str(preds[0]))
# print("准确率 = " + str(preds[1]))
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# img_path = 'datasets/4.jpg'
my_images = []
for i in range(0, 5):
    img_path = "datasets/2." + str(i + 1) + ".jpg"
    my_image = image.load_img(img_path, target_size=(64, 64))  # 这里还可以设置导入的像素
    my_image = image.img_to_array(my_image)
    # my_image = np.expend_dims(my_image, axis=0)
    my_image = np.expand_dims(my_image, axis=0)
    # my_images = [my_images, my_image]
    # my_images.append(my_image)
    my_image = preprocess_input(my_image) # 把x = preprocess_input(x) 注释掉，会显示概率，比较准确
    my_images.append(my_image)
    # print("my_image.shape = " + str(my_image.shape))
    print("class prediction vector  = ")
    print(model.predict(my_image))

# my_image = image.load_img(img_path)
# plt.imshow(my_image)
# plt.show()

# print(model.predict(my_images[0:3]))
my_images = np.array(my_images)
my_images = np.squeeze(my_images)
print(my_images.shape)
print(model.predict(my_images))