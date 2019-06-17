import os
import numpy as np
np.random.seed(123)
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,10
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#%matplotlib inline



import tensorflow as tf
tf.set_random_seed(123)

import keras
from keras.layers import Dense, Input, LeakyReLU, Activation
from keras.layers import UpSampling2D, Conv2D, Reshape, Flatten, MaxPooling2D
from keras.models import Sequential, Model




tf.reset_default_graph()
keras.backend.clear_session()

#获取数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot=True)
x_train = mnist.train.images
x_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels


#归一化
def norm(x):
    return (x-0.5)/0.5
#展示图像
def display_images(images):
    for i in range(images.shape[0]):
        plt.subplot(1, 8, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#超参数
pixel_size = 28
#随机噪声维数,测试生成模型
n_z = 256
z_test = np.random.uniform(-1.0,1.0,size=[8,n_z])
#学习率
g_learning_rate = 0.00001
d_learning_rate = 0.01
#图像的像素
n_x = 784
#生成器和判别器隐藏层数
g_n_layers = 3
d_n_layers = 1
#每个隐藏层的神经元
g_n_filters = [64,32,16]
d_n_filters = [64]

n_width=28
n_height=28
n_depth=1

n_epochs = 400
batch_size = 100
n_batches = int(mnist.train.num_examples / batch_size)
n_epochs_print = 50


#生成网络
g_model = Sequential(name='g')
g_model.add(Dense(units=5*5*128,  input_shape=(n_z,),name='g_in'))
#g_model.add(BatchNormalization())
g_model.add(Activation('tanh',name='g_in_act'))
g_model.add(Reshape(target_shape=(5,5,128), input_shape=(5*5*128,),name='g_in_reshape'))
for i in range(0,g_n_layers):
    g_model.add(UpSampling2D(size=[2,2],name='g_{}_up2d'.format(i)))
    g_model.add(Conv2D(filters=g_n_filters[i],kernel_size=(5,5),padding='same',name='g_{}_conv2d'.format(i)))
    g_model.add(Activation('tanh',name='g_{}_act'.format(i)))
g_model.add(Flatten(name='g_out_flatten'))
g_model.add(Dense(units=n_x, activation='tanh',name='g_out'))
print('生成网络:')
g_model.summary()
g_model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr=g_learning_rate))

#判别网路
d_model = Sequential(name='d')
d_model.add(Reshape(target_shape=(n_width,n_height,n_depth), input_shape=(n_x,),name='d_0_reshape'))


#######作业#######补充下面的代码#######作业########
### https://keras.io/zh/ 参考网站
for i in range(0,d_n_layers):
	#1.添加的卷层 Conv2D (参数 filters=d_n_filters[i], kernel_size=(5,5), padding='same'）
	
	#2.添加激活函数 Activation  (参数 'tanh')
	
 	#3.添加池化层  MaxPooling (参数pool_size=(2,2), strides=(2,2))
 	
d_model.add(Flatten(name='d_out_flatten'))
d_model.add(Dense(units=1, activation='sigmoid',name='d_out'))


print('判别网路:')
d_model.summary()
d_model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.SGD(lr=d_learning_rate))

# 定义 DCGAN network
d_model.trainable=False
z_in = Input(shape=(n_z,),name='z_in')
x_in = g_model(z_in)
gan_out = d_model(x_in)
gan_model = Model(inputs=z_in,outputs=gan_out,name='gan')
print('DCGAN:')
gan_model.summary()
gan_model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr=g_learning_rate))

for epoch in range(n_epochs+1):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    for batch in range(n_batches):
        x_batch, _ = mnist.train.next_batch(batch_size)
        x_batch = norm(x_batch)
        z_batch = np.random.uniform(-1.0,1.0,size=[batch_size,n_z])
        g_batch = g_model.predict(z_batch)
        
        x_in = np.concatenate([x_batch,g_batch])
        
        y_out = np.ones(batch_size*2)
        y_out[:batch_size]=0.9
        y_out[batch_size:]=0.1
        
        d_model.trainable=True
        batch_d_loss = d_model.train_on_batch(x_in,y_out)

        z_batch = np.random.uniform(-1.0,1.0,size=[batch_size,n_z])
        x_in=z_batch
        
        y_out = np.ones(batch_size)
            
        d_model.trainable=False
        batch_g_loss = gan_model.train_on_batch(x_in,y_out)
        
        epoch_d_loss += batch_d_loss 
        epoch_g_loss += batch_g_loss 
    if epoch%n_epochs_print == 0:
        average_d_loss = epoch_d_loss / n_batches
        average_g_loss = epoch_g_loss / n_batches
        print('epoch: {0:04d}   d_loss = {1:0.6f}  g_loss = {2:0.6f}'.format(epoch,average_d_loss,average_g_loss))
        #使用已经训练的生产器预测图像          
        x_pred = g_model.predict(z_test)
        display_images(x_pred.reshape(-1,pixel_size,pixel_size))





