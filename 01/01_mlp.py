import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import datasets
from sklearn import model_selection
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import keras

tf.reset_default_graph()
keras.backend.clear_session()


#加载 Iris 数据集
iris = datasets.load_iris()
#查看数据
iris.data
#存储花尊长度作为目标值
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])
seed = 3
#划分训练集&测试集
x_train,x_test,y_train,y_test=model_selection.train_test_split(x_vals,y_vals,random_state=seed,test_size=0.2)
#通过 min-max归一化数据 0到1之间
mm = MinMaxScaler()
x_train = np.nan_to_num(mm.fit_transform(x_train))
x_test = np.nan_to_num(mm.fit_transform(x_test))

#定义模型
model = Sequential()
model.add(Dense(5, input_dim=3, activation='relu'))
#model.add(Dense(5, input_dim=3))
#model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
#查看模型
model.summary()
#模型可视化
plot_model(model, to_file='model.png')
#编译模型  定义损失函数&优化器&评估标
model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])
#训练模型
model.fit(x_train, y_train,epochs=400, batch_size=50)
#在测试集上的表现
score = model.evaluate(x_test, y_test, batch_size=50)

#*************************************************#
#开始使用 Keras 函数式 API 定义模型
#这部分返回一个张量
inputs = Input(shape=(3,))
x = Dense(5, activation='relu')(inputs)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=predictions)
model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])
model.fit(x_train, y_train,epochs=400, batch_size=50)
