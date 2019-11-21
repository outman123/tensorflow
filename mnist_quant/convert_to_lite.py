import tensorflow as tf 
from tensorflow.lite.python import lite 
FLAGS = tf.app.flags.FLAGS

# 构建一个转换器
converter = lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='./models/frozen.pb', 
    input_arrays=['input_tensor'], 
    output_arrays=['output_tensor/BiasAdd'], 
    input_shapes={'input_tensor': [1, 28, 28, 1]})

# 设置属性
converter.inference_type = tf.uint8 
converter.inference_input_type = tf.uint8
converter.post_training_quantize = False
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
converter.default_ranges_stats = (0, 6)

#将pb文件转化成tflite文件
tflite_quantized_model = converter.convert()
with open('models/model1.tflite', 'wb') as f:
    f.write(tflite_quantized_model)
