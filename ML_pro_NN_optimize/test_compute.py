import numpy as np


def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J


print("测试前向传播")
x, theta = 2, 4
J = forward_propagation(x, theta)
print("j=" + str(J))


def backward_propagation(x, theta):

    dtheta = x

    return dtheta
print("反向传播")
x, theta = 2, 4
dtheta = backward_propagation(x, theta)
print("dtheta = " + str(dtheta))


def gradient_check(x, theta, epsilon=1e-7):

    thetaplus = theta +epsilon
    thetaminus = theta -epsilon
    J_plus = forward_propagation(x, thetaplus)
    J_minus =forward_propagation(x, thetaminus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    grad = backward_propagation(x, theta)
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print("梯度检测：梯度正常")
    else:
        print("梯度超过阈值")

    return difference

#测试gradient_check
print("-----------------测试gradient_check-----------------")
x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))

