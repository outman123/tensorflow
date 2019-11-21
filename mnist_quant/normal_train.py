import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU  
from tensorflow.keras import backend as K 

#加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

#归一化数据
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 对标进行one_hot编码
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

# 编译模型
model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])

#训练
model.fit(x_train, y_train,
          batch_size=128,
          epochs=3,
          verbose=1,
          validation_data=(x_test, y_test))

# 评估
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 保存成检查点文件
sess = K.get_session()
saver = tf.train.Saver()
saver.save(sess, './models/float_point/model.ckpt')
