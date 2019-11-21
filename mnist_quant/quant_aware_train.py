import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU  
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import TensorBoard 
from tensorflow.contrib.quantize import experimental_create_training_graph 

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# 将数据归一化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 将标签进行one—hot编码
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 构建模型
input_tensor = Input(shape=(28, 28, 1), name='input_tensor')
x = Conv2D(32, (3, 3), name='conv1')(input_tensor)
x = ReLU(name='relu1')(x)
x = Conv2D(64, (3, 3), name='conv2')(x)
x = ReLU(name='relu2')(x)
x = MaxPooling2D(pool_size=(2, 2), name='maxpool')(x)
x = Flatten(name='flatten')(x)
x = Dense(128)(x)
x = ReLU(name='relu3')(x)
output_tensor = Dense(10, name='output_tensor')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

sess = tf.compat.v1.keras.backend.get_session()

# 记录和准备预先训练的模型的变量和权重
per_trained_model_path = './models/float_point/model.ckpt'
restore_dict = {}
reader = tf.train.NewCheckpointReader(per_trained_model_path)
for v in tf.compat.v1.global_variables():
    tensor_name = v.name.split(':')[0]
    if reader.has_tensor(tensor_name):
        restore_dict[tensor_name] = v

# 重写图，向训练图中添加伪量化节点
experimental_create_training_graph(input_graph=sess.graph, 
                                   weight_bits=8, 
                                   activation_bits=8)

# 初始化非量化操作的变量
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# 取出之前训练的模型的变量
saver = tf.train.Saver(restore_dict)
saver.restore(sess, per_trained_model_path)

# 检查增加伪量化节点是否成功
for node in sess.graph.as_graph_def().node:
    if 'AssignMaxLast' in node.name or 'AssignMinLast' in node.name:
        print('node name: {}'.format(node.name))

# 以较小的学习率编译模型
model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])

# evaluate the fake quantization ops added model's performence
# before quantize-aware training begin
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# create a tensorboard 来观察重写的图的细节，比如在act_quant subgraph的最大值，最小值
tensorboard = TensorBoard('logs')

# 开始quantize-aware 训练
model.fit(x_train, y_train,
          batch_size=128,
          epochs=3,
          verbose=1,
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard])

# 评估 quantization-aware训练后模型的表现 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 保存 quantize-aware trained model 
saver = tf.train.Saver()
saver.save(sess, './models/quant_aware_trained/model1.ckpt')
