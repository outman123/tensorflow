'''
@version: python3.6
@author: Administrator
@file: 卷积神经网络应用-mnist数据集分类.py
@time: 2019/09/05
'''

# 卷积层就是对图像像素进行特征提取，多个不同卷积核增加深度
# 池化层就是降低分辨率，缩小尺寸，减少参数，加快计算

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100  # 一个批次的大小，可优化
n_batch = int(mnist.train.num_examples / batch_size)  # 一共有多少个批次


# 权值初始化
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))  # 截断的正态分布


# 偏置值初始化
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 卷基层
def conv2d(x, W):
    # x表示输入,是一个张量，W表示过滤器，即卷积核，strides表示步长，padding表示是否补零
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义两个占位符（训练数据集和标签）
x = tf.placeholder(tf.float32, [None, 784])  # 28*28
y = tf.placeholder(tf.float32, [None, 10])

# 将x转化为4d向量[batch,height,width,channel]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权值(卷积核)和偏置
w_con1 = weight_variable([5, 5, 1, 32])  # 5*5窗口大小，从一个平面中进行32次卷积，抽取32次特征
b_con1 = bias_variable([32])  # 每一次卷积对应一个偏置，所以32

# 将输入和权值进行卷积，加上偏置后激活，再进行池化
h_con1 = tf.nn.relu(conv2d(x_image, w_con1) + b_con1)
h_pool1 = max_pool_2x2(h_con1)

# 初始化第二个卷积层的权值(卷积核)和偏置
w_con2 = weight_variable([5, 5, 32, 64])  # 5*5窗口大小，64次卷积从32个平面中抽取64次特征
b_con2 = bias_variable([64])  # 每一次卷积对应一个偏置，所以32

# 将输入和权值进行卷积，加上偏置后激活，再进行池化
h_con2 = tf.nn.relu(conv2d(h_pool1, w_con2) + b_con2)
h_pool2 = max_pool_2x2(h_con2)

#第一个全连接层的权值和偏置
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
#池化后的输出扁平化为1维
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
#第一个全连接层的输出
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
#神经元输出概率
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#第二个全连接层
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
predict = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)  # softmax将输出信号转化为概率值（10个概率值）

# 二次代价函数（可使用交叉熵代价函数或对数似然代价函数来优化）
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=predict))
# 使用梯度下降法训练，使得loss最小（#可优化）
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

# 比较概率最大的标签是否相同，结果存放在一个布尔型列表中
correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))  # argmax返回一维张量中最大值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))  # reduce_mean求平均值

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):  # 可优化
        for batch in range(n_batch):  # 把所有图片训练一次
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys,keep_prob:0.7})
        # 用测试数据来检验训练好的模型
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0})
        print("Iter " + str(epoch) + "Test accuracy" + str(acc))
