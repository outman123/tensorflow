'''
@version: python3.6
@author: Administrator
@time: 2019/08/19
'''

import tensorflow as tf
import numpy as np
# #1、创建图和运行图
# #创建两个常量
# m1 = tf.constant([[3, 3]])
# m2 = tf.constant([[2], [3]])
# #创建一个矩阵乘法op
# product = tf.matmul(m1, m2)
# print(product)
# #启动图(会话中run触发op)
# sess=tf.Session()
# res=sess.run(product)
# print(res)
# sess.close()


# # 2、变量使用
# #变量
# x = tf.Variable([1, 2])
# # 常量
# a = tf.constant([3, 3])
# # 减法op
# sub = tf.subtract(x, a)
# add = tf.add(x, a)
# # 变量需要初始化
# init = tf.global_variables_initializer()
# # 会话中执行op
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))


# # 创建一个变量开始为0
# state = tf.Variable(0, name="counter")
# # 加法op
# new_value = tf.add(state, 1)
# # 赋值op
# update = tf.assign(state, new_value)
# # 初始化
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(state))
#     for _ in range(5):
#         sess.run(update)
#         print(sess.run(state))


# # 3、fetch与feed
# #fetch
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# add = tf.add(input2, input3)
# mul = tf.multiply(input1, add)
# with tf.Session() as sess:
#     print(sess.run([mul, add]))
# feed
# # 创建占位符
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1, input2)
# with tf.Session() as sess:
#     # feed的数据以字典形式传入
#     print(sess.run(output,feed_dict={input1:[8],input2:[2]}))


# 简单示例
# 生成100个随机点
x = np.random.rand(100)
y1 = x * 0.1 + 0.2
# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y2 = k * x + b
# 二次代价函数
loss = tf.reduce_mean(tf.square(y1 - y2))
# 定义一个梯度下降来训练优化器（改变k和b）
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(200):
        sess.run(train)
        # 每训练20次打印出变量k和b
        if step % 20 == 0:
            print(step, sess.run([k, b]))
