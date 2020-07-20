import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#   卷积模块，包含了以下函数：
#   使用0扩充边界
#   卷积窗口
#   前向卷积
#   反向卷积

#   池化模块，包括了以下函数：
#   前向池化
#   创建掩码
#   值分配
#   反向池化
#   我们将在这里从底层搭建一个完整的模块，之后我们会用TensorFlow实现。

# 使用0填充边界有以下好处：
# 1.卷积了上一层之后的CONV层，没有缩小高度和宽度。 这对于建立更深的网络非常重要，否则在更深层时，高度/宽度会缩小。
# 一个重要的例子是“same”卷积，其中高度/宽度在卷积完一层之后会被完全保留。
# 2.它可以帮助我们在图像边界保留更多信息。在没有填充的情况下，卷积过程中图像边缘的极少数值会受到过滤器的影响从而导致信息丢失。

# a = np.pad(a, ((0, 0), (1, 1), (0, 0), (3, 3), (0, 0)), 'constant', constant_values = (..,..))
# 表示在第一个维度上水平方向上padding=1,垂直方向上padding=2,在第二个维度上水平方向上padding=2,垂直方向上padding=2。
#  如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
#   mode为填补类型，即怎样去填补，有“constant”，“edge”等模式
# arr3D = np.array([[[1, 1, 2, 2, 3, 4],
#                  [1, 1, 2, 2, 4, 4],
#                  [3, 5, 2, 2, 3, 5]],
#
#                  [[0, 1, 2, 3, 4, 2],
#                   [2, 2, 2, 2, 2, 2],
#                   [7, 7, 7, 7, 7, 7]],
#
#                  [[1, 1, 1, 1, 1, 1],
#                   [2, 2, 2, 2, 2, 2],
#                   [3, 3, 3, 3, 3, 3]]])
# print(str(np.pad(arr3D, ((0, 0), (1, 1), (2, 2)), 'constant'))) # 第一个维度就是最外面的，其次是行，其次是列
# # print('constant:  \n' + str(np.pad(arr3D, ((0, 0), (1, 1), (2, 2)), 'constant')))
import keras

def zeros_pad(X, pad):

    X_paded = np.pad(X, (
        (0, 0),      # 样本数，不填充
        (pad, pad),  # 图像高度,你可以视为上面填充x个，下面填充y个(x,y)
        (pad, pad),  # 图像宽度,你可以视为左边填充x个，右边填充y个(x,y)
        (0, 0)),     # 通道数，不填充
        'constant', constant_values=0)

    return X_paded

# np.random.seed(1)
# x = np.random.randn(4, 3, 3, 2)  # 品一下
# x_paded = zeros_pad(x, 2)
# print("x.shape=", x.shape)
# print("x_paded.shape", x_paded.shape)
# print("x[1, 1] = ", x[1])
# print("x_paded[1, 1] = ", x_paded[1])
#
# fig, axarr = plt.subplots(1, 2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0, :, :, 0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0, :, :, 0])


def conv_single_step(a_slice_prev, W, b):

    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)

    return Z
# np.random.seed(2)
#
# a_slice_prev = np.random.randn(4, 4, 3)
# W = np.random.randn(4, 4, 3)
# b = np.random.randn(1, 1, 1)
# Z = conv_single_step(a_slice_prev, W, b)
#
# print("result = ", Z)

# 前向传播：在前向传播的过程中，我们将使用多种过滤器对输入的数据进行卷积操作，
# 每个过滤器会产生一个2D的矩阵，我们可以把它们堆叠起来，于是这些2D的卷积矩阵就变成了高维的矩阵。


def conv_forward(A_prev, W, b, hparameters):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # 维度为(m, n_H_prev, n_W_prev, n_C_prev)，（样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量）
    (f, f, n_C_prev, n_C) = W.shape
    # 维度为(f, f, n_C_prev, n_C)，（过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量）

    stride = hparameters["stride"]
    pad = hparameters['pad']

    # 计算卷积后的图像的宽度高度，参考上面的公式，使用int()来进行整数化
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

# print(int(1.1))
    # 用0来初始化卷积输出Z
    Z = np.zeros((m, n_H, n_W, n_C))

    # 通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zeros_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]  # 选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride   # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride    # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]  # 这里的最后的：代表（RGB）三种颜色
                    #   执行单步卷积, 这就是一个点
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])    # 因为b这里是randn(1, 1, 1, 8)
    assert(Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev, W, b, hparameters)

    return(Z, cache)


# np.random.seed(1)
#
# A_prev = np.random.randn(10, 4, 4, 3)
# W = np.random.randn(2, 2, 3, 8)   # 有八个滤波器，每个是2*2*3(3是RGB)
# b = np.random.randn(1, 1, 1, 8)
#
# hparameters = {"pad": 2, "stride": 1}
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
#
# print("np.mena(Z) = ", np.mean(Z))
# print("cache_conv[0][1][2][3] = ", cache_conv[0][0][0][0])  # cache第一个数字表示parameters,第二个数字代表W,第三个是b，
# # 第四个是个字典

# 搞定了，下一个池化层
# 最大值池化层，均值池化层

def pool_forward(A_prev, hparameters, mode="max"):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # 获取字典中的参数
    f = hparameters["f"]    # 滤波器的宽度
    stride = hparameters["stride"]   # 间隔
    # 计算输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # 这里直接调用了，前面是先取出来再用
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end,c]

                    # 对切片进行池化
                    if mode =="max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode=="average":
                        A[i, h, w, c] = np.mean(a_slice_prev)
    assert(A.shape == (m, n_H, n_W, n_C))

    cache =(A_prev, hparameters)

    return A, cache

# np.random.seed(1)
# A_prev = np.random.randn(2, 4, 4, 3)
# hparameters = {"f": 4, "stride": 1 }
#
# # A, cache = pool_forward(A_prev, hparameters, mode="max")
# A, cache = pool_forward(A_prev, hparameters)
# print("mode = max")
# print("A = ", A)
# print("------------------------")
# A, cache = pool_forward(A_prev, hparameters, mode="average")
# print("mode = average")
# print("A = ", A)

# 来吧，cnn的反向传播，一直以来的疑惑， 用numpy编写深度学习一定是最帅的
#   池化层就是upsample


def conv_backward(dZ, cache):
    #   获取cache的信息
    (A_prev, W, b, hparameters) = cache
    #   获取A_prev的信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    #   获取dz的基本信息
    (m, n_H, n_W, n_C) = dZ.shape
    #   获取W的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    pad = hparameters["pad"]
    stride = hparameters["stride"]

    #   初始化各个梯度的结构
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # 前向传播中我们使用了pad，反向传播也需要使用，这是为了保证数据结构一致
    A_prev_pad = zeros_pad(A_prev, pad)
    dA_prev_pad = zeros_pad(dA_prev, pad)

    for i in range(m):
        # 选择第i个扩充了的数据的样本,降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # 定位完毕，开始切片

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end]

                    # 切片完毕，使用上面的公式计算梯度
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad: -pad, pad:-pad, :]

    assert(dA_prev.shape ==(m, n_H_prev, n_W_prev, n_C_prev))

    return(dA_prev, dW, db)

# np.random.seed(1)
# A_prev = np.random.randn(10, 4, 4, 3)
# W = np.random.randn(2, 2, 3, 8)
# b = np.random.randn(1, 1, 1, 8)
# hparameters = {"pad" : 2, "stride": 1}
#
# #   前向传播
# Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# #   反向传播
# dA, dW, db = conv_backward(Z, cache_conv)
# print("dA_mean = ", np.mean(dA))
# print("dW_mean = ", np.mean(dW))
# print("db_mean = ", np.mean(db))
#  即使池化层没有反向传播过程中要更新的参数，我们仍然需要通过池化层反向传播梯度

def create_mask_from_window(x):
    mask = x ==np.max(x)
    return mask


# np.random.seed(1)
#
# x = np.random.randn(2, 3)
# mask = create_mask_from_window(x)
# print("x = " + str(x))
# print(np.max(x))
# print("mask = " + str(mask))

def distribute_value(dz, shape):

    (n_H, n_W) = shape

    average = dz / (n_H * n_W)

    a = np.ones(shape) * average

    return a

# dz = 2
# shape = (2, 2)
# a = distribute_value(dz, shape)
# print("a = " + str(a))


#   池化层的反向传播,没有dW,db,只需要算dA就可以了
def pool_backward(dA, cache, mode="max"):

    (A_prev, hparameters) = cache

    f = hparameters["f"]
    stride = hparameters["stride"]

    # 获取A_prev和dA的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    # 初始化输出结构
    dA_prev = np.zeros_like(A_prev)  # np.zeros_like好用啊

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride": 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode="max")
print("mode = max")
print("mean of dA = ", np.mean(dA))
print('dA_prev[1, 1] = ', dA_prev[1, 1])
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1, 1] = ', dA_prev[1, 1])