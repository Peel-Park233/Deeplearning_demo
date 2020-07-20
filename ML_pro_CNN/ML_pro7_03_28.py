import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
import tf_utils
# import keras
import cnn_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()
# index = 6
# plt.imshow(X_train_orig[index])
# print("y = " +str(np.squeeze(Y_train_orig[:, index])))
# plt.show()
X_train = X_train_orig / 255
X_test = X_test_orig / 255
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
# print("number of training example = " + str(X_train.shape[0]))
# print("number of test example = " + str(X_test.shape[0]))
# print("X_train shape: " + str(X_train.shape))   # 样品个数，行，列，通道（RGB)
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))
# conv_layers = {}


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y


# X, Y = creat_placeholders(64, 64, 3, 6)
# print("X = " + str(X))
# print("Y = " + str(X))

#   来初始化权值/过滤器W1 W1W1、W2 W2W2。在这里，我们不需要考虑偏置，因为TensorFlow会考虑到的。
#   需要注意的是我们只需要初始化为2D卷积函数，全连接层TensorFlow会自动初始化的。

def initialize_parameters():

    tf.set_random_seed(1)

    # W1 = tf.get_variable("W1", [4, 4, 3, 8])
    # W2 = tf.get_variable("W2", [2, 2, 8, 16])
    W1 = tf.Variable(tf.random.normal([4, 4, 3, 8]) / 10)  # 行，列，上层滤波器的数量，这层滤波器的数量
    W2 = tf.Variable(tf.random.normal([2, 2, 8, 16]) / 10)

    parameters = {
        "W1": W1,
        "W2": W2
    }

    return parameters
# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()  # 参数初始化
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1, 1, 1]))
#     print("W2 = " + str(parameters["W2"].eval()[1, 1, 1]))
#     sess_test.close()


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters["W2"]
    # Conv2d : 步伐：1，填充方式：“SAME”
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    # Max pool : 窗口大小：8x8，步伐：8x8，填充方式：“SAME”
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # 一维化上一层的输出
    P = tf.layers.flatten(P2)

    Z3 = tf.layers.dense(P, 6, activation=None)

    return Z3


# tf.reset_default_graph()
# np.random.seed(1)
#
# with tf.Session() as sess_test:
#     X, Y =create_placeholders(64, 64, 3, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#
#     a = sess_test.run(Z3, {X: np.random.randn(2, 64, 64, 3), Y: np.random.randn(2, 6)})
#     print("Z3 = " +str(a))
#     sess_test.close()


def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))

    return cost

# tf.reset_default_graph()

# with tf.Session() as sess_test:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 64, 3, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#
#     init = tf.global_variables_initializer()
#     # init = tf.global_variables_initializer()
#     sess_test.run(init)
#     a = sess_test.run(cost, {X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 6)})
#     print("cost = " + str(a))
#
#     sess_test.close()


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True, isPlot=True):

        ops.reset_default_graph()
        tf.set_random_seed(1)
        seed = 3
        (m, n_H0, n_W0, n_C0)  =X_train.shape
        n_y = Y_train.shape[1]
        costs = []
        # 为当前的维度创建占位符
        X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
        # 初始化参数
        parameters = initialize_parameters()
        # 前向传播
        Z3 = forward_propagation(X, parameters)
        # 计算陈本
        cost = compute_cost(Z3, Y)

        # 反向传播，由于框架已经实现了反向传播，我们只需要选择一个优化器就可了
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # 全局初始化所有变量
        init = tf.global_variables_initializer()

        # 开始运行
        with tf.Session() as sess:

            # 初始化参数
            sess.run(init)
            # 开始便利数据集
            for epoch in range(num_epochs):
                minibatch_cost = 0
                num_minibatches = int(m / minibatch_size)
                seed = seed + 1
                minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

                #   对每个数据块进行处理
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    #   最小化这个数据化的成本（这里应该是进行一次梯度下降吧）
                    _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    minibatch_cost += temp_cost/num_minibatches

                if print_cost:
                    if epoch % 5 == 0:
                        print("当前的成本为： ", epoch, "代，成本为：" + str(minibatch_cost))

                if epoch % 5 == 0:
                    costs.append(minibatch_cost)

            if isPlot:
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('daishu')
                plt.title("learning rate = " + str(learning_rate))
                plt.show()

            # 开始预测数据
            # 计算当前的预测情况
            predict_op = tf.arg_max(Z3, 1)
            corrent_prediction = tf.equal(predict_op, tf.arg_max(Y, 1))

            # 计算准确度
            accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
            print("corrent_prediction accuracy= " + str(accuracy))

            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            print("训练集准确度： " + str(train_accuracy))
            print("测试集准确度： " + str(test_accuracy))

            return(train_accuracy, test_accuracy, parameters)


        # _, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=150)

train_accuracy, test_accuracy, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=150)
