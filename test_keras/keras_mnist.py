import os

# os.environ["KERAS_BACKEND"] = "cntk"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# 读入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 查看数据什么样子
print(X_train)
print(X_train.shape)  # (60000, 28, 28)
print(X_train[0].shape)  # (28, 28)
print(y_train)  # 标签类分别是0-9的数字[5 0 4 ..., 5 6 8]
print(y_train.shape)  # (60000,)
print(y_train[0].shape)  # ()

# 将手写黑白字体变成标准四维张量，即（样本数，长，宽，1），并把像素值变成浮点格式
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# 由于每个像素值是介于0-255的，所以统一除以255，把像素值控制在0-1的范围
X_train /= 255
X_test /= 255


# 由于输入层需要10个节点，所以最好把目标数字0-9做成One-Hot编码的形式
def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe


# 把标签用OneHot编码重新表示一下
y_train_one = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
y_test_one = np.array([tran_y(y_test[i]) for i in range(len(y_test))])

# 接着搭建卷积神经网络
model = Sequential()
# 添加一层卷积层，构造64个过滤器，每个过滤器覆盖范围是3x3x1。
# 过滤器挪动步长为1，图像四周补一圈0，并用relu进行非线性变换
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 input_shape=(28, 28, 1),
                 activation='relu'))
# 添加一层MaxPooling,在2x2的格子中取最大值。
model.add(MaxPooling2D(pool_size=(2, 2)))
# 设立dropout层。将Dropout的概率设为0.5，todo 也可以设为0.2或0.3等常用值
model.add(Dropout(0.5))
# 重复构造，搭建深度网络
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))  # 以0.5的概率dropout
model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# 把当前节点展平
model.add(Flatten())

# 构造全连接神经网络层
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 网络配置
model.compile(loss='categorical_crossentropy',  # 损失函数，一般来说分类问题的损失函数都选择用交叉熵
              optimizer='adagrad',  # 优化器
              metrics=['accuracy'])  # metrics性能评估参数

# 打印模型
model.summary()
# 模型可视化
from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True)
# ==========================================================

'''
# 放入批量样本，进行训练
model.fit(X_train,
          y_train_one,
          validation_data=(X_test, y_test_one),
          epochs=20,  # 迭代轮数
          batch_size=128)  # 60000个数据里每次取128个数据 Number of samples per gradient update
# 在测试集上评价模型的准确度：
scores = model.evaluate(X_test,
                        y_test_one,
                        verbose=0)
print(scores)
'''

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 64)        640       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 14, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 7, 7, 128)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 7, 256)         295168    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 3, 3, 256)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 3, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               295040    
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_3 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                330       
=================================================================
Total params: 675,370
Trainable params: 675,370
Non-trainable params: 0
_________________________________________________________________

'''
