'''
@version: python3.6
@author: Administrator
@file: RNN循环神经网络和LSTM.py
@time: 2019/09/08
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# LSTM实现了三个门计算，即遗忘门、输入门和输出门
# 遗忘门负责决定保留多少上一时刻的单元状态到当前时刻的单元状态；
# 输入门负责决定保留多少当前时刻的输入到当前时刻的单元状态；
# 输出门负责决定当前时刻的单元状态有多少输出。
# 算法核心在于记忆单元(过去记忆*遗忘门+现在输入*输入门)。会产生两种输出，一个是记忆单元c，一个是隐藏层的输出h


#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

batch_size=50#每个批次50个样本
n_batch=mnist.train.num_examples//batch_size#数据集共有多少批次
max_time=28#每个样本的长度(timestep)
n_input=28#每个样本的维度
lstm_size=100#隐藏单元数量为100
n_class=10#10个分类

#定义两个占位符，用于输入样本的值和标签
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#初始化权值和偏置
weight=tf.Variable(tf.truncated_normal([lstm_size,n_class],stddev=0.1))
bias=tf.Variable(tf.constant(0.1,shape=[n_class]))

def RNN(x,weight,bias):
    inputs=tf.reshape(x,[-1,max_time,n_input])#dynamic_run函数的输入格式固定[样本数量，时间步(理解为每个样本长度，可改)，每个时间步维度]
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)#定义LSTM的cell
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)#循环运行该神经网络，使用LSTM提高长期记忆；outputs表示返回所以值，final_state表示最后一个timestep返回的值
    #final_state[0]表示记忆单元c,final_state[1]表示隐藏层输出h
    result=tf.nn.softmax(tf.matmul(final_state[1],weight)+bias)
    return  result

#带入数据后，RNN返回结果
prediction=RNN(x,weight,bias)
# 交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
# 使用AdamOptimizer优化，使得loss最小
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

# 比较概率最大的标签是否相同，结果存放在一个布尔型列表中
correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))  # reduce_mean求平均值

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(8):  # 可优化
        for batch in range(n_batch):  # 把所有图片训练一次
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        # 用测试数据来检验训练好的模型
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + "Test accuracy" + str(acc))

