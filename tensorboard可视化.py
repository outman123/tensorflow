'''
@version: python3.6
@author: Administrator
@file: tensorboard可视化.py
@time: 2019/08/30
'''

# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib.tensorboard.plugins import projector
#
# mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#
# max_steps = 1001#运行次数
# image_num = 3000#图片数量
# DIR = "G:/Tensorflow"#文件路径
# #定义会话
# sess=tf.Session()
# #载入图片（将3000个图片打包成一个矩阵）
# embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False)
# #参数概要
# def variable_summaries(var):  # 在tensorboard中显示var的相关属性值
#     with tf.name_scope("summaries"):
#         mean = tf.reduce_mean(var)  # 平均值
#         tf.summary.scalar("mean", mean)
#         with tf.name_scope("stddev"):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar("stddev", stddev)  # 标准差
#         tf.summary.scalar("max", tf.reduce_max(var))  # 最大值
#         tf.summary.scalar("min", tf.reduce_min(var))  # 最小值
#         tf.summary.histogram("histogram", var)  # 直方图
#
# #命名空间
# with tf.name_scope("input"):
#     x = tf.placeholder(tf.float32, [None, 784], name="x-input")
#     y = tf.placeholder(tf.float32, [None, 10], name="y-input")
# #显示图片
# with tf.name_scope("input_reshape"):
#     image_shaped_input=tf.reshape(x,[-1,28,28,1])
#     tf.summary.image("input",image_shaped_input,10)
#
# # 创建简单神经网络（可优化）
# with tf.name_scope("layout"):
#     with tf.name_scope("wights"):
#         W = tf.Variable(tf.random_normal([784, 10]), name="W")
#         variable_summaries(W)
#     with tf.name_scope("biases"):
#         b = tf.Variable(tf.zeros([10]), name="b")
#         variable_summaries(b)
#     with tf.name_scope("wx_plus_b"):
#         wx_plus_b = tf.matmul(x, W) + b
#     with tf.name_scope("predict"):
#         predict = tf.nn.softmax(wx_plus_b)  # softmax将输出信号转化为概率值（10个概率值）
#
# # 可使用交叉熵代价函数来优化
# with tf.name_scope("loss"):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=predict))
#     tf.summary.scalar("loss",loss)
# # 使用梯度下降法训练，使得loss最小（#可优化）
# with tf.name_scope("train"):
#     train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# with tf.name_scope("accuracy"):
#     # 比较概率最大的标签是否相同，结果存放在一个布尔型列表中
#     with tf.name_scope("correct_predict"):
#         correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))  # argmax返回一维张量中最大值所在的位置
#     # 求准确率
#     with tf.name_scope("accu"):
#         accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))  # reduce_mean求平均值
#         tf.summary.scalar("accuracy",accuracy)
#
# #产生matedata文件
# if(tf.gfile.Exists(DIR+"/projector/projector/metadata.tsv")):
#     tf.gfile.DeleteRecursively(DIR+"/projector/projector/metadata.tsv")
# with open(DIR + '/projector/projector/metadata.tsv','w') as f:
#     labels=sess.run(tf.argmax(mnist.test.labels[:],1))
#     for i in range(image_num):
#         f.write(str(labels[i]) + '\n')
#
# # 合并所有summar
# merged=tf.summary.merge_all()
#
#
# # 定义一个writer，写入图
# projector_writer = tf.summary.FileWriter(DIR + '/projector/projector',sess.graph)
# #用来保存网络模型
# saver = tf.train.Saver()
# # 定义配置文件
# config = projector.ProjectorConfig()
# embed = config.embeddings.add()
# embed.tensor_name = embedding.name
# embed.metadata_path = DIR + '/projector/projector/metadata.tsv'
# embed.sprite.image_path = DIR + '/projector/data/mnist_10k_sprite.png'
# # 切分图片
# embed.sprite.single_image_dim.extend([28,28])
# projector.visualize_embeddings(projector_writer, config)
#
# for i in range(max_steps):
#     #每个批次100个样本
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
#                           run_metadata=run_metadata)
#     projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
#     projector_writer.add_summary(summary, i)
#
#     if i % 100 == 0:
#         acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#         print("Iter " + str(i) + ", Testing Accuracy= " + str(acc))
#
# saver.save(sess, DIR + '/projector/projector/a_model.ckpt', global_step=max_steps)
# projector_writer.close()
# sess.close()


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

#读取数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
max_steps=1001#运行次数
image_num=3000#图片数量
DIR='G:/Tensorflow'#文件路径

sess=tf.Session()#定义会话

#载入图片（将3000个图片打包成一个矩阵）
embedding=tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name='embedding')

#参数概要
def variable_summaries(var):  # 在tensorboard中显示var的相关属性值
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)  # 平均值
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)  # 标准差
        tf.summary.scalar("max", tf.reduce_max(var))  # 最大值
        tf.summary.scalar("min", tf.reduce_min(var))  # 最小值
        tf.summary.histogram("histogram", var)  # 直方图
#命名空间
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x_input')
    y=tf.placeholder(tf.float32,[None,10],name='y_input')
#显示10个数字的图片
with tf.name_scope('input_reshape'):
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)
#创建简单神经网络
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        w=tf.Variable(tf.random_normal([784,10]),name='W')
        variable_summaries(w)
    with tf.name_scope('biases'):
        b=tf.Variable(tf.zeros([10]),'b')
        variable_summaries(b)
    with tf.name_scope('Wx_plus_b'):#线性
        Wx_plus_b=tf.matmul(x,w)+b
    with tf.name_scope('softmax'):#通过softmax激活函数变成非线性
        y_pred=tf.nn.softmax(Wx_plus_b)
#交叉熵代价函数
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_pred))
    tf.summary.scalar('loss',loss)
#精确度
with tf.name_scope('accuracy'):
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1)),tf.float32))# argmax返回一维张量中最大值所在的位置
    tf.summary.scalar('accuracy',accuracy)
#使用梯度下降算法来训练，使得loss最小（可优化）
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
sess.run(tf.global_variables_initializer())
#产生metadata文件
if tf.gfile.Exists(DIR + '/projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + '/projector/projector/metadata.tsv')
with open(DIR + '/projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + '/projector/projector', sess.graph)#定义一个writer，写入图
# writer=tf.summary.FileWriter(DIR+'/projector/projector/logs',sess.graph)
saver = tf.train.Saver()#保存图
config = projector.ProjectorConfig()#配置
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + '/projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + '/projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

#开始训练
for i in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                          run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)

    if i % 100 == 0:#每100个图片打印一次结果
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Iter ' + str(i) + ', accuracy ' + str(acc))

saver.save(sess, DIR + '/projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()
