import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU  
from tensorflow.keras import backend as K 
from tensorflow.contrib.quantize import experimental_create_eval_graph 

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

# 添加伪量化节点到eval garph
experimental_create_eval_graph(input_graph=sess.graph, 
                               weight_bits=8, 
                               activation_bits=8)


for node in sess.graph.as_graph_def().node:
    if 'AssignMaxLast' in node.name or 'AssignMinLast' in node.name:
        print('node name: {}'.format(node.name))

# 加载quantize-aware 训练的模型的权重
saver = tf.train.Saver()
saver.restore(sess, './models/quant_aware_trained/model.ckpt')

# 固化图
const_graph = tf.graph_util.convert_variables_to_constants(
    sess=sess,
    input_graph_def=sess.graph.as_graph_def(),
    output_node_names=['output_tensor/BiasAdd']) 

# 将固化图序列化保存成pb文件
with tf.gfile.GFile('./models/frozen1.pb', "wb") as f:
    f.write(const_graph.SerializeToString())
