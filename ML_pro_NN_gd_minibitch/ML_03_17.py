import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
# 右边的提示键很好用啊。。。deBug不用翻来翻去
import opt_utils
import testCase

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# plt.show()


def update_parameters_with_gd(parameters, grads, learning_rate):
    # 使用梯度下降算法

    L = len(parameters) // 2    # 神经网络的层数

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

# print(" 测试 ")
# parameters, grads, learning_rate = testCase.update_parameters_with_gd_test_case()
# parameters = update_parameters_with_gd(parameters, grads, learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

#   但是还有其他不同的梯度下降算法，
#   1.随机梯度下降算法：当训练集很大时，使用随机梯度下降算法的运行速度会很快，但是会存在一定的波动。
#   2.小批量梯度下降: 小批量梯度下降法是一种综合了梯度下降法和随机梯度下降法的方法，在它的每次迭代中，既不是选择全部的数据来学习，也不是选择一个样本来学习，而是把所有的数据集分割为一小块一小块的来学习，它会随机选择一小块
# k = 10
# for i in range(10):
#     print(i)
#
# for i in range(0, 10):
#     print(i)


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 第一步：打乱顺序
    permutation = list(np.random.permutation(m))  # 它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))   # 将每一列的数据按permutation的顺序来重新排列

    # 第二部： 分割
    num_complete_minibatches = math.floor(m / mini_batch_size)   # 把你的训练集分割成多少份,请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
    for k in range(0, num_complete_minibatches):
        # print(num_complete_minibatches)
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k+1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# print("测试")
# X_assess, Y_assess, mini_batch_size = testCase.random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
# print("第1个mini_batch_X 的维度为：", mini_batches[0][0].shape)
# print("第1个mini_batch_Y 的维度为：", mini_batches[0][1].shape)
# print("第2个mini_batch_X 的维度为：", mini_batches[1][0].shape)
# print("第2个mini_batch_Y 的维度为：", mini_batches[1][1].shape)
# print("第3个mini_batch_X 的维度为：", mini_batches[2][0].shape)
# print("第3个mini_batch_Y 的维度为：", mini_batches[2][1].shape)
# print(mini_batches.shape)  # 这个无法运行，因为这里mini_batches经过append操作后是列表的形式


# 包含动量的梯度下降由于小批量梯度下降只看到了一个子集的参数更新，更新的方向有一定的差异，
# 所以小批量梯度下降的路径将“振荡地”走向收敛，使用动量可以减少这些振荡，动量考虑了过去的梯度以平滑更新，
# 我们将把以前梯度的方向存储在变量v中，从形式上讲，这将是前面的梯度的指数加权平均值。我们也可以把V看作是滚下坡的速度

def initialize_velocity(parameters):

    L = len(parameters) // 2
    v = {}  # 新建一个字典

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v


# #测试initialize_velocity
# print("-------------测试initialize_velocity-------------")
# parameters = testCase.initialize_velocity_test_case()
# v = initialize_velocity(parameters)
#
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
def update_parameters_with_momentun(parameters, grads, v, beta, learning_rate):

    L = len(parameters) // 2
    for l in range(L):
        v["dw" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
    # print(str(type(parameters)), "3")
    return parameters, v

# print("测试")
# parameters, grads, v = testCase.update_parameters_with_momentum_test_case()
# update_parameters_with_momentun(parameters, grads, v, beta=0.9, learning_rate=0.01)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))   # 这个是字典输出前后加''
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))


# Adam算法 Adam本质上实际是 随机梯度下降+动量
# Adam算法是训练神经网络中最有效的算法之一，它是RMSProp算法与Momentum算法的结合体。我们来看看它都干了些什么吧~
# 物理观点建议梯度只是影响速度，然后速度再影响位置，
# 比如当 dW 或者 db 中有一个值比较大的时候，那么我们在更新权重或者偏置的时候除以它之前累积的梯度的平方根，这样就可以使得更新幅度变小

def initialize_adam(parameters):


    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return(v, s)


# # 测试initialize_adam
# print("-------------测试initialize_adam-------------")
# parameters = testCase.initialize_adam_test_case()
# v, s = initialize_adam(parameters)
#
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
# print('s["dW1"] = ' + str(s["dW1"]))
# print('s["db1"] = ' + str(s["db1"]))
# print('s["dW2"] = ' + str(s["dW2"]))
# print('s["db2"] = ' + str(s["db2"]))


# def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2 = 0.999, epsilon=1e-8):
#     '''
#         v - Adam的变量，第一个梯度的移动平均值，是一个字典类型的变量
#         s - Adam的变量，平方梯度的移动平均值，是一个字典类型的变量
#         t - 当前迭代的次数
#         learning_rate - 学习率
#         beta1 - 动量，超参数,用于第一阶段，使得曲线的Y值不从0开始（参见天气数据的那个图）
#         beta2 - RMSprop的一个参数，超参数
#         epsilon - 防止除零操作（分母为0）
#     '''
#     L = len(parameters) // 2
#     v_corrected = {}    # 偏差修正后的值
#     s_corrected = {}    # 偏差修正后的值
#     print(parameters["b1"].shape)
#     for l in range(L):
#         v["dw" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
#         v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
#
#         v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
#         v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
#
#         s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
#         s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])
#
#         s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
#         s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
#
#         parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate*(v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
#         parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate*(v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))
#     print(parameters["b1"].shape)
#     return parameters, v, s
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用Adam更新参数

    参数：
        parameters - 包含了以下字段的字典：
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
        grads - 包含了梯度值的字典，有以下key值：
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        v - Adam的变量，第一个梯度的移动平均值，是一个字典类型的变量
        s - Adam的变量，平方梯度的移动平均值，是一个字典类型的变量
        t - 当前迭代的次数
        learning_rate - 学习率
        beta1 - 动量，超参数,用于第一阶段，使得曲线的Y值不从0开始（参见天气数据的那个图）
        beta2 - RMSprop的一个参数，超参数
        epsilon - 防止除零操作（分母为0）

    返回：
        parameters - 更新后的参数
        v - 第一个梯度的移动平均值，是一个字典类型的变量
        s - 平方梯度的移动平均值，是一个字典类型的变量
    """
    L = len(parameters) // 2
    v_corrected = {}  # 偏差修正后的值
    s_corrected = {}  # 偏差修正后的值
    # print(parameters["b1"].shape)
    for l in range(L):
        # 梯度的移动平均值,输入："v , grads , beta1",输出：" v "
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # 计算第一阶段的偏差修正后的估计值，输入"v , beta1 , t" , 输出："v_corrected"
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        # 计算平方梯度的移动平均值，输入："s, grads , beta2"，输出："s"
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])

        # 计算第二阶段的偏差修正后的估计值，输入："s , beta2 , t"，输出："s_corrected"
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        # print(parameters["b2"].shape)
        # 更新参数，输入: "parameters, learning_rate, v_corrected, s_corrected, epsilon". 输出: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
                    v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (
                    v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))
        # print(parameters["b2"].shape)
    return (parameters, v, s)


#测试update_with_parameters_with_adam
# print("-------------测试update_with_parameters_with_adam-------------")
# parameters , grads , v , s = testCase.update_parameters_with_adam_test_case()
# update_parameters_with_adam(parameters,grads,v,s,t=2)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
# print('s["dW1"] = ' + str(s["dW1"]))
# print('s["db1"] = ' + str(s["db1"]))
# print('s["dW2"] = ' + str(s["dW2"]))
# print('s["db2"] = ' + str(s["db2"]))


train_X, train_Y = opt_utils.load_dataset(is_plot=False)
# plt.show()


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007,
          mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=20000, print_cost=True, is_plot=True):
    # 那就来看看哪个梯度下降算法号
    L = len(layers_dims)
    costs = []
    t = 0 # 每学完一个就增加1
    seed = 10

    # 初始化参数
    parameters = opt_utils.initialize_parameters(layers_dims)
    # print(str(type(parameters)), "1")

    # 选择优化器
    if optimizer == "gd":
        pass # 不用任何优化器，批量梯度下降算法
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)   # 使用动量优化算法
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)  # 使用adam优化算法
    else:
        print("optimizer参数错误，程序退出")
        exit(1)

    # 开始学习
    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            # print(len(minibatches))
            # 选择一个minibatch
            # (minibatch_X, minibatch_Y) = minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # 前向传播
            A3, cache = opt_utils.forward_propagation(minibatch_X, parameters)

            # 计算误差
            cost = opt_utils.compute_cost(A3, minibatch_Y)

            # 反向传播
            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)

            # 更新参数
            if optimizer == "gd":
                parameters  = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                # 这里少了一个数，导致parameters格式输出错误。。，变成元组了
                parameters, v = update_parameters_with_momentun(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，误差值为：" + str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('dai_shu(100)')
        plt.title("learning_rate = " + str(learning_rate))
        plt.show()

    return parameters


# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer="gd", is_plot=True)
layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer="momentum", beta=0.9, is_plot=True)
parameters = model(train_X, train_Y, layers_dims, optimizer="adam", is_plot=True)
# predictions = opt_utils.predict(train_X, train_Y, parameters)
#
# plt.title(" model 1")
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
#预测
preditions = opt_utils.predict(train_X, train_Y, parameters)

#绘制分类图
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)

# 总结亚当算法(adam为什么这么强？？？？