'''
@version: python3.6
@author: Administrator
@file: 利用tfrecord类型进行训练图像分类.py
@time: 2019/09/17
'''

import tensorflow as tf
import os
import random
import math
import sys
import types


_NUM_TEST=300
_RANDOM_SEED=0
_NUM_SHARDS=5
_DATASET_DIR="F:/git_code/slim/images/"
LABELS_FILENAME='F:/git_code/slim/images/labels.txt'


# 定义tfrecord文件的路径和名字
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'image_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


# 判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        for shard_id in range(_NUM_SHARDS):
            # 定义tfrecord文件的路径+名字
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
        if not tf.gfile.Exists(output_filename):
            return False
    return True


# 获取所有文件以及分类  传入图片的路径
def _get_filenames_and_classes(dataset_dir):
    # 数据目录
    directories = []
    # 分类名称
    class_names = []
    for filename in os.listdir(dataset_dir):
        # 合并文件路径
        path = os.path.join(dataset_dir, filename)
        # 判断该路径是否为目录
        if os.path.isdir(path):
            # 加入数据目录
            directories.append(path)
            # 加入类别名称
            class_names.append(filename)
    photo_filenames = []
    # 循环每个分类的文件夹
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            # 把图片加入图片列表
            photo_filenames.append(path)
    return photo_filenames, class_names

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values=[values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data,image_format,class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format' : bytes_feature(image_format),
        'image/class/label' : int64_feature(class_id)
    }))

def write_label_file(labels_to_class_names, dataset_dir, filename='label.txt'):
    # 拼接目录
    labels_file_name = os.path.join(dataset_dir, filename)
    print(dataset_dir)
    # with open(labels_file_name,'w') as f:
    with tf.gfile.Open(labels_file_name, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d;%s\n' % (label, class_name))

 # 把数据转为TFRecord格式
def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    # assert 断言   assert expression 相当于 if not expression raise AssertionError
    assert split_name in ['train', 'test']
    # 计算每个数据块有多少个数据
    num_per_shard = int(len(filenames) / _NUM_SHARDS)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                # 定义tfrecord文件的路径+名字
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecore_writer:
                    # 每一个数据块开始的位置
                    start_ndx = shard_id * num_per_shard
                    # 每一个数据块最后的位置
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))

                    for i in range(start_ndx, end_ndx):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id))
                            sys.stdout.flush()
                            # 读取图片
                            # image_data = tf.gfile.FastGFile(filenames[i],'rb').read()
                            img_data = tf.gfile.FastGFile(filenames[i],'rb').read()
                            # img = img.resize((224, 224))
                            #img_raw = img.tobytes()
                            # 获取图片的类别名称
                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            # 找到类别名称对应的id
                            class_id = class_names_to_ids[class_name]
                            # 生成tfrecord文件
                            example = image_to_tfexample(img_data, b'jpg', class_id)
                            # print(filenames[i])
                            tfrecore_writer.write(example.SerializeToString())
                        except IOError as e:
                            print("Could not read: ", filenames[i])
                            print("Error: ", e)
                            print("Skip it \n")

    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__=='__main__':
    if _dataset_exists(_DATASET_DIR):
        print("tfrecord文件已经存在")
    else:
        photo_filenames,class_names=_get_filenames_and_classes(_DATASET_DIR)
        class_names_to_ids=dict(zip(class_names,range(len(class_names))))

        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        train_filenames=photo_filenames[_NUM_TEST:]
        test_filenames=photo_filenames[:_NUM_TEST]
        # 数据转换
        _convert_dataset('train', train_filenames, class_names_to_ids, _DATASET_DIR)
        _convert_dataset('test', test_filenames, class_names_to_ids, _DATASET_DIR)

        # 输出labels文件
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(labels_to_class_names, _DATASET_DIR)