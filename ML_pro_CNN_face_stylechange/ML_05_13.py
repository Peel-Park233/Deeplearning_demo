from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
# python是一种边编译边运行的语言和R不一样，R是运行完了有数据和函数保存的，python如果要像R一样运行其中一段代码可以用火星
#    用于绘制模型细节
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

K.set_image_data_format('channels_first')

import time
import cv2  # 应该是待会导入的包
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import fr_utils
from inception_blocks_v2 import *  # 就是import所有
#   获取模型
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
#   打印模型的总参数数量
print("参数数量：" + str(FRmodel.count_params()))
#   生成图片用的
# plot_model(FRmodel, to_file='FRmodel.png')
# SVG(model_to_dot(FRmodel).create(prog='dot', format='svg'))


def triplrt_loss(y_true, y_pred, alpha = 0.2):

    #   tf.reduce_sum() 用于计算张量tensor沿着某一维度的和，可以在求和后降维。
    #   tf.square()是对a里的每一个元素求平方
    #   tf.subtract()是对两个矩阵相减
    anchor, positive,  negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)

    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


# with tf.Session() as test:
#     tf.set_random_seed(1)
#     y_true = (None, None, None)  # 表示一个不知长度的三维数组？
#     y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
#               tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
#               tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
#     loss = triplrt_loss(y_true, y_pred)
#
#     print("loss = " + str(loss.eval()))

#  计算时间
start_time = time.process_time()

#  编译模型
FRmodel.compile(optimizer='adam', loss=triplrt_loss, metrics=['accuracy'])

#   加载权值
fr_utils.load_weights_from_FaceNet(FRmodel)

#   结束时间
end_time = time.process_time()

#   计算时差
minium = end_time - start_time

print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")

database = {}  # 定义一个字典
database["danielle"] = fr_utils.img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = fr_utils.img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = fr_utils.img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = fr_utils.img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = fr_utils.img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = fr_utils.img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = fr_utils.img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = fr_utils.img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = fr_utils.img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = fr_utils.img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = fr_utils.img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = fr_utils.img_to_encoding("images/arnaud.jpg", FRmodel)


def verify(image_path, identity, database, model):

    encoding = fr_utils.img_to_encoding(image_path, model)

    dist = np.linalg.norm(encoding - database[identity])

    if dist < 0.7:
        print("欢迎" + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证， 您与" + str(identity) + "不符！")
        is_door_open = False

    return dist, is_door_open

#
# verify("images/camera_0.jpg", "younes", database, FRmodel)
# verify("images/camera_2.jpg", "kian", database, FRmodel)


def who_is_it(image_path, database, model):


    encoding = fr_utils.img_to_encoding(image_path, model)

    min_dist = 100

    for (name, db_enc) in database.items():     # 字典的名字和对应的内容

        dist = np.linalg.norm(encoding - db_enc)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("抱歉， 您的信息不在数据库中。")
    else:
        print("姓名" + str(identity) + "  差距:" + str(min_dist))

    return min_dist, identity

def who_are_you(image_path, database, model):


    encoding = fr_utils.img_to_encoding_plus(image_path, model)

    min_dist = 100

    for (name, db_enc) in database.items():     # 字典的名字和对应的内容

        dist = np.linalg.norm(encoding - db_enc)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("抱歉， 您的信息不在数据库中。")
    else:
        print("姓名" + str(identity) + "  差距:" + str(min_dist))

    return min_dist, identity

who_is_it("images/camera_1.jpg", database, FRmodel)
who_is_it("images/camera_2.jpg", database, FRmodel)
who_are_you("images/peel_park.jpg", database, FRmodel)  # good，成功了
