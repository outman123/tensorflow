'''
@version: python3.6
@author: Administrator
@file: 非线性回归.py
@time: 2019/08/21
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点，转化成维度（200行一列）
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder占位符（若干行一列,和样例相同）
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))  # 中间层10个神经原
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 偏值
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1  # 线性变换
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 激活函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
predict = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - predict))  # 样例值和预测值的差的平方作为代价
# 梯度下降算法训练器
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 训练过程在减少代价损失

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练1000次
    for _ in range(1000):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    predict_value = sess.run(predict, feed_dict={x: x_data})
    plt.figure()
    # 离散点
    plt.scatter(x_data, y_data)
    # 连续预测值
    plt.plot(x_data, predict_value, 'r-', lw=5)
    plt.show()
