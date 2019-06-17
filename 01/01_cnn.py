import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Reshape
from keras.optimizers import SGD
from keras.utils import plot_model
#获取数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot=True)

#将mnist数据分为测试数据与训练数据
X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels


tf.reset_default_graph()
keras.backend.clear_session()

#设置超参数
n_classes = 10  
n_width = 28
n_height = 28
n_depth = 1
n_inputs = n_height * n_width * n_depth  # total pixels
n_epochs = 10
batch_size = 100
n_batches = int(mnist.train.num_examples/batch_size)

#
n_filters=[32,64]
learning_rate = 0.01

#定义模型
model = Sequential()
#设置模型的输入数据
model.add(Reshape(target_shape=(n_width,n_height,n_depth), input_shape=(n_inputs,)))

#添加卷积层  4*4卷积核     生成32个特征图
model.add(Conv2D(filters=n_filters[0], kernel_size=4, padding='SAME', activation='relu' ) )
#添加池化层
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2) ) )

#添加卷积层  4*4卷积核     生成64个特征图
model.add(Conv2D(filters=n_filters[1], kernel_size=4, padding='SAME', activation='relu', ) )
#添加池化层
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#将特征图展开
model.add(Flatten())
#添加一个全连接层
model.add(Dense(units=1024, activation='relu'))

#最后softmax层
model.add(Dense(units=n_classes, activation='softmax'))

#查看模型的概要
model.summary()
#模型可视化
plot_model(model, to_file='01_cnn_model.png')

#编译模型
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=learning_rate),metrics=['accuracy'])

#训练模型
model.fit(X_train, Y_train,batch_size=batch_size,epochs=n_epochs)

#评估模型
score = model.evaluate(X_test, Y_test)
print('损失:', score[0])
print('准确率:', score[1])

