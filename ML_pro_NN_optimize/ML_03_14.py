"""
1. 初始化参数：
	1.1：使用0来初始化参数。
	1.2：使用随机数来初始化参数。
	1.3：使用抑梯度异常初始化参数（参见视频中的梯度消失和梯度爆炸）。
2. 正则化模型：
	2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
	2.2：使用随机删除节点的方法精简模型，同样是为了尝试避免过拟合。
3. 梯度校验  ：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况。

"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils    # 第一部分初始化
import reg_utils    # 第二部分，正则化
import gc_utils     # 第三部分，梯度检测

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'  # 最近邻差值: 像素为正方形
plt.rcParams['image.cmap'] = 'gray'   # 使用灰度输出而不是彩色输出
train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)
# plt.show()


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True,initialization="he", is_plot=True):
    grads = {}  # 这表示一个字典
    costs = []  # 这表示一个矩阵
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else :
        print("错误的初始化参数！退出程序")
        exit

    for i in range(0, num_iterations):
        a3, cache = init_utils.forward_propagation(X, parameters)

        #计算成本
        cost = init_utils.compute_loss(a3, Y)

        #反向传播
        grads = init_utils.backward_propagation(X, Y, cache)

        #更新参数
        parameters = init_utils.update_parameters(parameters, grads, learning_rate)

        #记录成本
        if i%1000 == 0:
            costs.append(cost)
            # 打印成本
            if print_cost:
                print("第" +str(i) +"次迭代，成本为：" +str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration (per hundreds)')
        plt.title("learning rate =" +str(learning_rate))
        plt.show()

    return parameters


def initialize_parameters_zeros(layers_dims):

    parameters ={}

    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters   # 缩进问题- -


def initialize_parameters_random(layers_dims):
    parameters ={}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])
        # parameters["b" + str(l)] = np.random.randn(layers_dims[l], 1)
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))  # zeros要加双括号

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))
    return parameters
# parameters = initialize_parameters_zeros([3, 2, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


parameters = model(train_X, train_Y, initialization="he", is_plot=True)

print("训练集")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集")
predicitons_test = init_utils.predict(test_X, test_Y, parameters)
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predicitons_test))

plt.title("model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

# 正则化对算法影响
train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=False)
# plt.show()
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.nansum(logprobs) + 1. / m * lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

    return cost


def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m) # 对W求偏导
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)   # 对b求偏导

    dA2 = np.dot(W3.T, dZ3)  # 对x求偏导
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))   # 若A2>0就是原来的导数，若A2<0就是0
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = reg_utils.relu(Z1)

    # 使用这样的方法来Drop_out
    D1 = np.random.rand(A1.shape[0], A1.shape[1])   #步骤1：初始化矩阵D2 = np.random.rand(..., ...)
    D1 = D1 < keep_prob      # 步骤2：将D1的值转换为0或1（使​​用keep_prob作为阈值）
    A1 = A1 * D1            # 步骤3：舍弃A1的一些节点（将它的值变为0或False）
    A1 = A1 / keep_prob     # 步骤4：缩放未舍弃的节点(不为0)的值

    Z2 = np.dot(W2, A1) + b2
    A2 = reg_utils.relu(Z2)

    D2 = np.random.randn(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def backward_propagation_with_dropout(X, Y, cache, keep_prob):

    m = X.shape[1]   # X  - 输入数据集，维度为（2，示例数）
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = 1. /m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = dA2 * D2  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA2 = dA2 / keep_prob   # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./ m * np.dot(dZ1, X.T)
    db1 = 1./ m * np.sum(dZ1, axis=1, keepdims=True)
    # 发现错误了可以多找找字典
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dZ1": dZ1, "dW1": dW1, "db1": db3}   # 这个真的难找"db1"字典里面对应错了导致后面查询字典的时候不对了

    return gradients


def model2(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):

    grads = {}
    costs = []
    m =X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    parameters = reg_utils.initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        # 前向传播
        # 询问是否删除节点
        if keep_prob == 1:
            # 不随机删除节点
            a3, cache = reg_utils.forward_propagation(X, parameters)
        elif keep_prob < 1:
            # 随即删除节点
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        else:
            print("keep_prob参数错误！程序退出")
            exit

        # 计算成本
        # 是否正则化
        if lambd == 0:
            cost = reg_utils.compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
        # 本次实验不同时使用正则化和随即删除节点
        assert(lambd == 0 or keep_prob == 1)

        if(lambd == 0 and keep_prob == 1):
            grads = reg_utils.backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads =backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

         # 更新参数
        parameters = reg_utils.update_parameters(parameters, grads, learning_rate)

        if i % 1000 == 0:
            # 要记录的
            costs.append(cost)
            if(print_cost and i % 10000 == 0):
                # 要输出的
                print("第" + str(i) + "次迭代， 成本为：" +str(cost))


    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(x1,000)')
        plt.title("learning rate =" + str(learning_rate))
        plt.show()

    return parameters


#  哪件比较难的事不是从copy开始学习的呢，多抄几遍就熟了啊，当然要保持思考
# 不用正则化，也不用drop_out
# parameters = model2(train_X, train_Y, is_plot=True)
# 使用正则化
# parameters = model2(train_X, train_Y, lambd=0.7, is_plot=True)   # 正则化之后确实好多了，不过lambd这个参数需要自己跳
# 使用drop_out
parameters = model2(train_X, train_Y, keep_prob=0.86, learning_rate=0.3, is_plot=True)
print("训练集")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("测试集")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.4])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)  #  很明显的过拟合特性

