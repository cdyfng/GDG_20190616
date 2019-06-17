import os
import numpy as np
np.random.seed(123)
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,10
import tensorflow as tf
tf.set_random_seed(123)
import keras
from keras.layers import Dense, Input, LeakyReLU, Dropout
from keras.models import Sequential, Model
from keras.utils import plot_model
#import seaborn as sns
#sns.set_style("whitegrid")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#%matplotlib inline


tf.reset_default_graph()
keras.backend.clear_session()

#获取数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot=True)
x_train = mnist.train.images
x_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

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
g_n_neurons = [256, 512, 1024]
d_n_neurons = [256]
n_epochs = 400
batch_size = 100
n_batches = int(mnist.train.num_examples / batch_size)
n_epochs_print = 50

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



#定义生成器
g_model = Sequential()
g_model.add(Dense(units=g_n_neurons[0], input_shape=(n_z,),name='g_0'))
g_model.add(LeakyReLU())
for i in range(1,g_n_layers):
    g_model.add(Dense(units=g_n_neurons[i],name='g_{}'.format(i)))
    g_model.add(LeakyReLU())
g_model.add(Dense(units=n_x, activation='tanh',name='g_out'))
print('生成器:')
g_model.summary()
g_model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr=g_learning_rate))

#定义判别器
d_model = Sequential()
d_model.add(Dense(units=d_n_neurons[0],  input_shape=(n_x,),name='d_0'))
d_model.add(LeakyReLU())
d_model.add(Dropout(0.3))
for i in range(1,d_n_layers):
    d_model.add(Dense(units=d_n_neurons[i], name='d_{}'.format(i)))
    d_model.add(LeakyReLU())
    d_model.add(Dropout(0.3))
d_model.add(Dense(units=1, activation='sigmoid',name='d_out'))
print('判别器:')
d_model.summary()

d_model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.SGD(lr=d_learning_rate) )



#定义 GAN 网络
#将判别器可训练属性设置为false,GAN 仅用于训练生成器
d_model.trainable=False
z_in = Input(shape=(n_z,),name='z_in')
x_in = g_model(z_in)
gan_out = d_model(x_in)
gan_model = Model(inputs=z_in,outputs=gan_out,name='gan')
print('GAN:')
gan_model.summary()
####plot_model(gan_model,'02_1_simplegan.png')

gan_model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr=g_learning_rate) )



#训练模型
for epoch in range(n_epochs+1):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    for batch in range(n_batches):
        x_batch, _ = mnist.train.next_batch(batch_size)
        #真实数据
        x_batch = norm(x_batch)
        z_batch = np.random.uniform(-1.0,1.0,size=[batch_size,n_z])
        #生成网络产生的数据
        g_batch = g_model.predict(z_batch)
        
        x_in = np.concatenate([x_batch,g_batch])
        
        y_out = np.ones(batch_size*2)
        #真假图像的标签设置为0.9 0.1
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
        
    

        