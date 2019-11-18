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

# load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# normalize the input data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# build the model
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

# record all the variables in the per-train model. 
# prepare for restore per-trained model's weights. 
# cause the variables added by the next step's rewrite get 
# added to the global variables collection, 
# we need do this record before rewrite the graph
per_trained_model_path = './models/float_point/model.ckpt'
restore_dict = {}
reader = tf.train.NewCheckpointReader(per_trained_model_path)
for v in tf.compat.v1.global_variables():
    tensor_name = v.name.split(':')[0]
    if reader.has_tensor(tensor_name):
        restore_dict[tensor_name] = v

# rewrite the graph, add fake quantize ops to the training graph
experimental_create_training_graph(input_graph=sess.graph, 
                                   weight_bits=8, 
                                   activation_bits=8)

# we added lots of variables when add fake quantize ops
# so we we have to initizlize those variables
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# restore variables which included in the per-trained model
saver = tf.train.Saver(restore_dict)
saver.restore(sess, per_trained_model_path)

# check if add the fake quantize ops successfully
for node in sess.graph.as_graph_def().node:
    if 'AssignMaxLast' in node.name or 'AssignMinLast' in node.name:
        print('node name: {}'.format(node.name))

# compile the model. usually, we use a smaller learning rate when we load 
# already trained floating point model 
model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])

# evaluate the fake quantization ops added model's performence
# before quantize-aware training begin
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# create a tensorboard callback to see the the details of the re-writed graph 
# of the model. for example, inside the conv1 scope we can find act_quant
# subgraph which contain information about quantization like min and max.
tensorboard = TensorBoard('logs')

# begin quantize-aware training
model.fit(x_train, y_train,
          batch_size=128,
          epochs=3,
          verbose=1,
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard])

# evaluate quantization-awar trained model's performence 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save quantize-aware trained model to checkpoint file
saver = tf.train.Saver()
saver.save(sess, './models/quant_aware_trained/model1.ckpt')
