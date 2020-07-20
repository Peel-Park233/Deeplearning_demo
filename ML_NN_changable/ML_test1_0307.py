import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCase
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils
np.random.seed(1)

# 为了初始化两层网络参数设置


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape ==(n_h, n_x))
    assert(b1.shape ==(n_h, 1))
    assert(W2.shape ==(n_y, n_h))
    assert(b2.shape ==(n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])/np.sqrt(layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        # print(layers_dims[l - 1])
        # print(np.sqrt(layers_dims[l - 1]))
        # print(parameters["W" + str(l)].shape)
# A = np.random.randn(3,4)
# print(str(len(A)) + str(A.shape))
        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters


# #测试initialize_parameters_deep
# print("==============测试initialize_parameters_deep==============")
# layers_dims = [5, 4, 3]
# parameters = initialize_parameters_deep(layers_dims)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def linear_forward(A, W, b):

    Z = np.dot(W, A) +b
    assert(Z.shape == (W.shape[0], A.shape[1]))  # 取决于W的行数和A的列数
    cache = (A, W, b)

    return Z, cache


# #测试linear_forward
# print("==============测试linear_forward==============")
# A, W, b = testCase.linear_forward_test_case()
# Z,linear_cache = linear_forward(A,W,b)
# print("Z = " + str(Z))

def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    else:
        print("不存在这样的激活函数")

    assert(A.shape ==(W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

# matrix = [
#     [-1, -2, -3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]
# # k = np.random.randn(3, 2)
# matrix = np.array(matrix)
# A, actibation_cache = relu(matrix)
# print(A)
# 测试linear_activation_forward


# print("==============测试linear_activation_forward==============")
# A_prev, W, b = testCase.linear_activation_forward_test_case()
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
# print("sigmoid，A = " + str(A))
# print(linear_activation_cache)
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
# print("ReLU，A = " + str(A))


def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1, L):  # 左闭右开
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
        # print(l)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))

    return AL, caches


# # print(11//3)
# #测试L_model_forward
# print("==============测试L_model_forward==============")
# X, parameters = testCase.L_model_forward_test_case()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("caches 的长度为 = " + str(len(caches)))

# squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m  # np.multiply 数组对应元素位置相乘

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


# #测试compute_cost
# print("==============测试compute_cost==============")
# Y, AL = testCase.compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m  # np.dot矩阵乘法已经自动求和了，dW, db, dA和他前向传播时W, b ,A的维度相同, W*A_prev=Z的反向操作，这里A_prev的列数是样本数量，所以除以m.本身的矩阵点乘求和并没有除掉
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    # print(db.shape)
    # print(b.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db


#测试linear_backward
print("==============测试linear_backward==============")
dZ, linear_cache = testCase.linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

# def linear_activation_forward(A_prev, W, b, activation):
#
#     if activation == "sigmoid":
#         Z, linear_cache = linear_forward(A_prev, W, b)
#         A, activation_cache = sigmoid(Z)
#     elif activation == "relu":
#         Z, linear_cache = linear_forward(A_prev, W, b)
#         A, avtivation_cache = relu(Z)
#
#     assert(A.shape == (W.shape[0], A_prev.shape[1]))
#     cache = (linear_cache, activation_cache )
#     return A, cache

# def linear_activation_backward(dA, cache, activation="relu"):
#     linear_cache, activation_cache = cache
#     if activation == "relu":
#         dZ = relu_backward(dA, activation_cache)
#         dA_prev, dW, db = linear_backward(dZ, linear_cache)
#     elif activation  == "sigmoid":
#         dZ = sigmoid_backward(dA, activation_cache)
#         dA_prev, dW, db = linear_backward(dZ, linear_cache)
#
#
#     return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation="relu"):
    """
    实现LINEAR-> ACTIVATION层的后向传播。

    参数：
         dA - 当前层l的激活后的梯度值
         cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
         activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    返回：
         dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
         db - 相对于b（当前层l）的成本梯度值，与b的维度相同
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


#  不确定第一步输出层的误差AL究竟怎么算的
# 测试linear_activation_backward
# print("==============测试linear_activation_backward==============")
# AL, linear_activation_cache = testCase.linear_activation_backward_test_case()
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
# print("sigmoid:")
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db) + "\n")
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="relu")
# print("relu:")
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db))
def L_model_backward(AL, Y, caches):
    grads = {}
    L =len(caches)
    m = AL.shape[1]   # AL概率向量，正向传播的输出,这里求AL的列数
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1-AL))

    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]   # cache层数不对了，sigmoid对应relu了，所以出问题了
        # dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)], current_cache, "relu")
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")
        grads["dA" + str(l+1)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads



# # 测试L_model_backward
# print("==============测试L_model_backward==============")
# AL, Y_assess, caches = testCase.L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print ("dW1 = " + str(grads["dW1"]))
# print ("db1 = " + str(grads["db1"]))
# print ("dA1 = " + str(grads["dA1"]))
def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate*grads["db" + str(l+1)]

    return parameters


# # 测试update_parameters
# print("==============测试update_parameters==============")
# parameters, grads = testCase.update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


#  试一下两层的神经网络
def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):

    np.random.seed(1)
    grads = {}
    costs = []
    # (n_x, n_h, n_y) = layers_dims
    #
    # #  初始化参数
    # parameters = initialize_parameters(n_x, n_h, n_y)
    parameters = initialize_parameters_deep(layers_dims)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        # 前向传播
        # A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        # A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        caches = 1
        AL, caches = L_model_forward(X, parameters)
        # 计算误差
        cost = compute_cost(AL, Y)
        # 反向传播
        grads = L_model_backward(AL, Y, caches)
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第", i, "次迭代，成本为：", np.squeeze(cost))  # print里面用,也可以连接，之前都是用+

    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('daishu')
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.title("learning rate = " + str(learning_rate))
        plt.show()
    return parameters


# train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
# # 把图片的255个点转移成一列
# print(test_set_x_orig.shape)
# print(test_set_x_orig.shape[3])
# train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   # 改变维度为m行、1列, -1的作用就在此: 自动计算d：d=数组或者矩阵里面所有的元素个数
# test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T   # 注意这里转置了
# print(test_x_flatten.shape)
# train_x = train_x_flatten / 255
# train_y = train_set_y
# test_x = test_x_flatten/255
# test_y = test_set_y
# n_x = train_x_flatten.shape[0]   # 就是之前算出来
# n_2 = 20
# n_3 = 7
# n_y = 1
# layers_dims = (n_x, n_2, n_3, n_y)  # 第一层到第二层是relu,第二层到第三层是sigmoid
# parameters = two_layer_model(train_x, train_set_y, layers_dims, num_iterations=2500, print_cost=True, isPlot=False)


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("准确度为：", float(np.sum((p == y))/m))   # 这个p == y就直接可以每个数组与数组相对

    return p


# pre_train = predict(train_x, train_y, parameters)
# pre_test = predict(test_x, test_y, parameters)


def print_mislabeled_images(classes, X, y, p):
    """
    绘制预测和实际不同的图像。
        X - 数据集
        y - 实际的标签
        p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))  # 只有（0，1）或者（1，0）错误判断了才会为1
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # 设置图片的大小
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        print(index)
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0, index])].decode("utf-8") + "\n Class:" + classes[y[0, index]].decode("utf-8"))
    plt.show()


# print_mislabeled_images(classes, test_x, test_y, pre_test)