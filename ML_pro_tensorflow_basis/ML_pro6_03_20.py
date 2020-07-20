import numpy as np

import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import tf_utils
import time
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# tf.disable_v2_behavior()
np.random.seed(1)
y_hat = tf.constant(36, name="y_hat")
y = tf.constant(39, name="y")

loss = tf.Variable((y - y_hat)**2, name="loss")  # 为损失函数创建一个变量

init = tf.global_variables_initializer()   # 运行之后的初始化(session.run(init))

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)

# print(c)
# 正如预料中一样，我们并没有看到结果20，不过我们得到了一个Tensor类型的变量，没有维度，
# 数字类型为int32。我们之前所做的一切都只是把这些东西放到了一个“计算图(computation graph)”中，
# 而我们还没有开始运行这个计算图，为了实际计算这两个数字，我们需要创建一个会话并运行它：

sess = tf.Session()

print(sess.run(c))

x = tf.placeholder(tf.int64, name="x")
print(sess.run(2 * x, feed_dict={x:3}))
sess.close()
#  当我们第一次定义x时，我们不必为它指定一个值。 占位符只是一个变量，我们会在运行会话时将数据分配给它。相当于sys x
