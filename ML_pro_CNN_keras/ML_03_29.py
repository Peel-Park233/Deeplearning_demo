import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import kt_utils
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
# import os
# os.environ['KERAS_BACKEND']='TensorFlow'

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# print("number of training examples = " + str(X_train.shape[0]))
# print("number of test example = " + str(X_test.shape[0]))
# print("X_train shape: ", X_train.shape)
# print("Y_train shape: ", Y_train.shape)
# print("X_test shape：", X_test.shape)
# print("Y_test shape:", Y_test.shape)
# help(BatchNormalization)
# 而假设经过BN后，均值是0，方差是1，那么意味着95%的x值落在了[-2,2]区间内，很明显这一段是sigmoid(x)函数接近于线性变换的区域，
# 意味着x的小变化会导致非线性函数值较大的变化，也即是梯度变化较大，对应导数函数图中明显大于0的区域，就是梯度非饱和区。
# 它位于Z=WU+B激活值获得之后，非线性函数变换之前


def HappyModel(input_shape):
    # 使用0填充：X_input的周围填充0
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    # 最大值池化层
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # 降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


# 创建一个模型实体
happy_model = HappyModel(X_train.shape[1:])
# 编译模型
happy_model.compile("adam", "binary_crossentropy", metrics=['accuracy'])
# 训练模型
happy_model.fit(X_train, Y_train, epochs=10, batch_size=50)
# 评估模型
preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1)
print("误差值 = " + str(preds[0]))
print("准确度 = " + str(preds[1]))
img_path = 'datasets/smile.jpeg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happy_model.predict(x))