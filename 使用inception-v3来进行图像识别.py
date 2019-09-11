'''
@version: python3.6
@author: Administrator
@file: 使用inception-v3来进行图像识别.py
@time: 2019/09/09
'''
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt


class NodeLookup1(object):
    def __init__(self):
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符转n*******对应各分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        # 一行一行读取数据
        for line in proto_as_ascii_lines:
            line = line.strip('\n')#除去换行
            parsed_items = line.split('\t')#以tab来分割
            uid = parsed_items[0]#取出分类编码
            human_string = parsed_items[1]#取出分类分类名称
            uid_to_human[uid] = human_string#构造分类编码和分类名称之间的映射

        # 加载分类字符串n*******（对应分类编号1-1000的文件）
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.strip().startswith('target_class:'):
                target_class = int(line.strip().split(':')[1])# 取出分类编号(1-1000某个数)
            elif line.strip().startswith('target_class_'):
                target_class_string = line.strip().split(':')[1].strip()# 取出编号字符串
                node_id_to_uid[target_class] = target_class_string[1:-1]# 构造分类编号和编号字符串的映射关系

        # 建立分类编号 1-1000 与对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name


    # 传入分类编号1-1000 返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


# 创建一个图来存放模型
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # 给图命名
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    for root, dirs, files in os.walk('image/'):
        for file in files:
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()#读取图片
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)  # 结果转化为一维

            # 打印图片路径和图片本身
            image_path = os.path.join(root, file)
            print(image_path)
            imag = Image.open(image_path)
            plt.imshow(imag)
            plt.axis('off')
            plt.show()

            top_k = predictions.argsort()[-5:][::-1]#从小到大排序后取出后5位，在倒置
            node_look_up = NodeLookup1()
            for node_id in top_k:
                human_string = node_look_up.id_to_string(node_id)#得到node_id对应的分类名称
                score = predictions[node_id]#得到该分类的可能性
                print("%s (%.5f)" % (human_string, score))
