# import numpy
# import tensorflow as tf
# # print("tf")
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.python.framework import ops
# import tf_utils
# import time
# #%matplotlib inline #如果你使用的是jupyter notebook取消注释

# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a, b)
#
# # print(c)
# #
# # sess = tf.Session()
# #
# # hello = tf.constant('Hello,World!')
# #
# # print(sess.run(hello))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
version = tf.__version__
gpu_ok = tf.test.is_gpu_available()
print("tf version:", version, "\nuse GPU",gpu_ok)
