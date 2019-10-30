#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tf CNN训练识别身份证数字(18个字符)图片


把OCR的问题当做一个多标签学习的问题。
4个数字组成的验证码就相当于有4个标签的图片识别问题（这里的标签还是有序的）,用CNN来解决。
"""
from genIDCard  import *

import numpy as np
import tensorflow as tf

obj = gen_id_card()#实例化一个类
image,text,vec = obj.gen_image()#调用方法返回图像，标签以及标签的one-hot编码

#图像大小
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 256
MAX_CAPTCHA = obj.max_size#验证最长18位
CHAR_SET_LEN = obj.len#有10中字符可选

# 生成一个训练batch，每个批次默认128个验证码样本
def get_next_batch(batch_size=128):
    obj = gen_id_card()
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    for i in range(batch_size):
        image, text, vec = obj.gen_image()#返回的image是一个二维矩阵，text是字符串，vec是一维01向量
        batch_x[i,:] = image.reshape((IMAGE_HEIGHT*IMAGE_WIDTH))
        batch_y[i,:] = vec
    return batch_x, batch_y
 
####################################################################
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout
# 是否在训练阶段
train_phase = tf.placeholder(tf.bool)


#
'''
使用过程：归一化就是将数据的输入值减去其均值然后除以数据的标准差，几乎所有数据预处理都会使用这一步骤,
在网络的每一层都进行数据归一化处理，但每一层对所有数据都进行归一化处理的计算开销太大，因此就和使用最小批量梯度下降一样，
批量归一化中的“批量”其实是采样一小批数据，然后对该批数据在网络各层的输出进行归一化处理.

使用原因：神经网络本质是学习数据分布，如果训练数据与测试数据分布不同，网络的泛化能力将降低，batchnorm就是通过对每一层的计算做scale和shift的方法，
通过规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到正太分布，减小其影响，让模型更加健壮；
使得不同层不同scal的权重变化整体步调更一致，可以使用更高的学习率，加快训练速度；除此之外，防止过拟合。
'''

def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
    with tf.variable_scope(scope):
        #beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
        #gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed
 
 
# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 4 conv layer
    w_c1 = tf.Variable(w_alpha*tf.random_normal([5, 5, 1, 32]))#卷积核
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))#偏置
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)#卷基层
    conv1 = batch_norm(conv1, tf.constant(0.0, shape=[32]), tf.random_normal(shape=[32], mean=1.0, stddev=0.02), train_phase, scope='bn_1')#归一化
    conv1 = tf.nn.relu(conv1)#激活函数
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#池化
    conv1 = tf.nn.dropout(conv1, keep_prob)#dropout使得部分神经节点失效，降低过拟合

    w_c2 = tf.Variable(w_alpha*tf.random_normal([5, 5, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
    conv2 = batch_norm(conv2, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_2')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
    conv3 = batch_norm(conv3, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_3')
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c4 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
    conv4 = batch_norm(conv4, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_4')
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4 = tf.nn.dropout(conv4, keep_prob)

    #经过4次卷积层提取出样本中的特征，再输出到全连接层中去分类
    # Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([2*16*64, 1024]))#1024个隐藏节点
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    #out = tf.nn.softmax(out)
    return out
 
# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    # loss
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)#预测下标值
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)#真实下标值
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#平均误差

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75, train_phase:True})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0 and step != 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1., train_phase:False})
                print("第%s步，训练准确率为：%s" % (step, acc))
                # 如果准确率大80%,保存模型,完成训练
                if acc > 0.8:
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    break
            step += 1

train_crack_captcha_cnn()


