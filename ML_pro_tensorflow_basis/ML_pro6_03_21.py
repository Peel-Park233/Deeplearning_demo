import numpy as np
import h5py
import matplotlib.pyplot as plt
# import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time
import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


y_hat = tf.constant(36, name="y_hat")            # 定义y_hat为固定值36
y = tf.constant(39, name="y")                    # 定义y为固定值39

loss = tf.Variable((y-y_hat)**2,name="loss" )   # 为损失函数创建一个变量

init = tf.global_variables_initializer()        # 运行之后的初始化(ession.run(init))
# 方法二：创建session                                             # 损失变量将被初始化并准备计算
with tf.Session() as session:                   # 创建一个session并打印输出
    session.run(init)                           # 初始化变量
    print(session.run(loss))                    # 打印损失值


def linear_funciton():

    np.random.seed(2)

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    Y = tf.add(tf.matmul(W, X), b)
# 方法一：创建session
    sess = tf.Session()
    result = sess.run(Y)
    sess.close()

    return result


# print("result = " + str(linear_funciton()))

def sigmoid(z):

    x = tf.placeholder(tf.float32, name="x")  # 就相当于matlab里面sys x
    sigmoid = tf.sigmoid(x)

    #  创建一个绘画
    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x:z})

    return result

#
# print("sigmoid(0) = " + str(sigmoid(0)))
# print("sigmoid(12) = " + str(sigmoid(12)))

#  然后计算交叉熵成本J,为啥都要用交叉熵成本。

def one_hot_matrix(lables, C):

    C = tf.constant(C, name="C")    # 就和之前定义常数一样的

    matrix = tf.one_hot(indices=lables, depth=C, axis=0)

    sess = tf.Session()

    one_hot = sess.run(matrix)

    sess.close()

    return one_hot


# labels = np.array([1, 2, 3, 0, 2, 1])
# one_hot = one_hot_matrix(labels, C=4)
# print(str(one_hot))

def ones(shape):

    ones = tf.ones(shape)

    sess = tf.Session()

    ones = sess.run(ones)

    sess.close()

    return ones


# print("ones = " + str(ones([3])))
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()
index = 11
plt.imshow(X_train_orig[index])
print("Y = " + str(np.squeeze(Y_train_orig[:, index])))
# plt.show()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T  # 行数不变，列数自己算，每一列就是一个样本
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# 归一化输入数据
X_train= X_train_flatten / 255
X_test = X_test_flatten / 255

# 转化为独立热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)

# print("训练集样本数 = " + str(X_train.shape[1]))
# print("测试集样本数 = " + str(X_test.shape[1]))
# print("X_train.shape: " + str(X_train.shape))
# print("Y_train.shape: " + str(Y_train.shape))
# print("X_test.shape: " + str(X_test.shape))
# print("Y_test.shape: " + str(Y_test.shape))


def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y


# X, Y = create_placeholders(12288, 6)
# print("X = " + str(X))
# print("Y = " + str(Y))

# 仅仅改变了他的初始化方式，为何效果这么好？
def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.variance_scaling_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.variance_scaling_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.variance_scaling_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))
#     print("W3 = " + str(parameters["W3"]))
#     print("b3 = " + str(parameters["b3"]))

def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


# tf.reset_default_graph()    # 用于清除默认图形堆栈并重置全局默认图形。
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     print("Z3 = " + str(Z3))

def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)   # 转置
    labels = tf.transpose(Y)    # 转置
    # tf.reduce_mean应该是求和再求平均吧
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return cost


tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters =initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))


def model(X_train, Y_train, X_test, Y_test,
          learning_rate=0.0001, num_epochs=1500, minibatch_size=32,
          print_cost=True, is_plot=False):

    ops.reset_default_graph()   # 能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    # 给X, Y占位符
    X, Y  = create_placeholders(n_x, n_y)

    # 初始化参数
    parameters = initialize_parameters()

    # 前向传播
    Z3 = forward_propagation(X, parameters)

    #   计算误差
    cost = compute_cost(Z3, Y)

    #   反向传播
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #   初始化所有变量
    # init = tf.global_variables_initializer()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:  # 除了这个范围session就自动over了
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            #  下面就是帮你分割好了的
            minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                # 这里应该是进行一次正向传播，计算误差，然后反向传播
                #  运行tf.session时，必须将此对象与成本函数一起调用，当被调用时，它将使用所选择的方法和学习速率对给定成本进行优化。
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                # _ 作为一次性变量来存储我们稍后不需要使用的值

                epoch_cost = epoch_cost + minibatch_cost / num_minibatches

            if epoch % 10 == 0:
                costs.append(epoch_cost)

                if print_cost and epoch % 100 == 0:
                        print("epoch = " + str(epoch) + " epoch_cost = " + str(epoch_cost))

        if is_plot:
            plt.plot(np.squeeze(costs))

            plt.ylabel('cost')
            plt.xlabel('daishu（10）')
            plt.title("learning rate =" + str(learning_rate))
            plt.show()

        # 保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率： ", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率： ", accuracy.eval({X: X_train, Y: Y_train}))

        return parameters


# #   开始时间
# start_time = time.process_time()
#   开始训练
parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=800)
#   结束时间
# end_time = time.process_time()
# #   计算时差
# print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )

# %%  分段落运行好像不可以哎
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
#
# my_image1 = "2.png"
# # fileName1 = "images/fingers/" + my_image1
# fileName1 = "images/fingers/" + my_image1
# image1 = mpimg.imread(fileName1)
# plt.imshow(image1)
# my_image1 = image1.reshape(1, 64*64*3).T
# my_image_prediction = tf_utils.predict(my_image1, parameters)
# print("预测结果： y = " + str(np.squeeze(my_image_prediction)))