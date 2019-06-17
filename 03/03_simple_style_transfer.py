from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
from keras.applications import vgg19
from keras import backend as K
import sys
 

#自定义超参数
#内容图片路径
#base_image_path = "../data/buildings.jpg"
base_image_path = "./rHuang1.jpeg"
#风格图片路径
#style_reference_image_path = "../data/starry-sky.jpg"
style_reference_image_path = "./fast-neural-style-keras-master/images/style/starry_night.jpg"
#生成图片名前缀
result_prefix ="starry_night"
#迭代次数
iterations = 10
#内容权重
content_weight = 0.025
#风格权重
style_weight = 1
#整体方差权重
total_variation_weight = 1

#获取内容图片的尺寸
width, height = load_img(base_image_path).size
#设定生成的图片的高度为400
img_nrows = 400
#与内容图片等比例，计算对应的宽度
img_ncols = int(width * img_nrows / height)
 
#图像预处理 使用keras导入图片，转为合适格式的Tensor
def preprocess_image(image_path):
    #读入图像，并转化为目标尺寸。
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) #3
    #vgg提供的预处理，主要完成（1）去均值（2）RGB转BGR（3）维度调换三个任务。
    img = vgg19.preprocess_input(img)
    return img
 
#图像后处理，将Tensor转换回图片
def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
 
 
#读入内容图和风格图，预处理，并包装成变量。这里把内容图和风格图都做成相同尺寸。
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))
#给目标图片定义占位符，目标图像与resize后的内容图大小相同。
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

#将三个张量串联到一起，形成一个形如(3,3,img_nrows,img_ncols)的张量
#三张图一同喂入网络中，以batch的形式
input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)
 
#加载vgg19预训练模型，模型由imagenet预训练.去掉模型的全连接层.
model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
#取出每一层的输出结果，字典的key为每层的名字，对应的value为该层输出的feature map
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
 
#计算特征图的格拉姆矩阵，格拉姆矩阵算两者的相关性，这里算的是一张特征图的自相关。

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
 


#计算风格图与生成图之间的格拉姆矩阵的距离，作为风格loss
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
 
#内容图和生成图之间的距离作为内容loss
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
 
#第三个loss函数总变异损失， 它鼓励生成的图像中的空间连续性，从而避免过度像素化的结果
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

#先计算content部分的loss
#loss初始化为0
loss = K.variable(0.)
#以第5卷积块第2个卷积层的特征图为输出。
layer_features = outputs_dict['block5_conv2']
#抽取内容特征图和生成特征图
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
#计算内容loss
loss += content_weight * content_loss(base_image_features,combination_features)

#计算style部分的loss
feature_layers = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1','block5_conv1']
#抽取风格图和生成图每个卷积块第一个卷积层输出的特征图
#并逐层计算风格loss，叠加在到loss中
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
    
#计算total_variation_loss 累加在总的loss上，得到最终的loss
loss += total_variation_weight * total_variation_loss(combination_image)
 
#计算生成图像的梯度
grads = K.gradients(loss, combination_image)
 
#output[0]为loss，剩下的是grad
outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

#相当于定义一个模型
f_outputs = K.function([combination_image], outputs)
 
#单独计算损失函数的值和梯度的值是低效的，因为这样做会导致两者之间的大量冗余计算;这个过程几乎是共同计算过程的两倍。定义函数同时计算损失值和梯度值

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values
 

class Evaluator(object):
 
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
 
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
 
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
 
evaluator = Evaluator()
 
#使用L-BFGS算法来求解非约束优化问题，scipy中提供了实现，使用fmin_l_bfgs_b函数来求解前面得到的总的loss的最小值
#x的初始化，初始化成内容图
x = preprocess_image(base_image_path)
#x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.
#迭代优化过程
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    #使用L-BFGS-B算法优化
    #不断被优化的是x，也就是生成图。
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    #图像后处理
    img = deprocess_image(x.copy())
    #保存图像
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))