'''
@version: python3.6
@author: Administrator
@file: Dropout.py
@time: 2019/08/25
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100  # 可优化
n_batch = int(mnist.train.num_examples/batch_size)

# 定义两个placeholder（训练数据集和标签）
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#dropout的所要训练神经元的比例
keep_prob=tf.placeholder(tf.float32)

# 创建简单神经网络
W1 = tf.Variable(tf.truncated_normal([784, 2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop=tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop=tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop=tf.nn.dropout(L3,keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
predict = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# 交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=predict))
# 使用梯度下降法训练，使得loss最小（#可优化）
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
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
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0})

        print("Iter " + str(epoch) + " Test accuracy " + str(test_acc)+" train_accuracy "+str(train_acc))
