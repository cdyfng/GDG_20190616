import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import patches

#获取标注的训练数据
train = pd.read_csv("keras-frcnn-master/train.txt",header=None)
train.columns = ['image_name','xmin','ymin','xmax','ymax','cell_type']
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
#读取一张图片
image = plt.imread("keras-frcnn-master/train_images/BloodImage_00001.jpg")
plt.imshow(image)
for _,row in train[train.image_name == "train_images/BloodImage_00001.jpg"].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax
    width = xmax - xmin
    height = ymax - ymin
    #用不同颜色的框表示不同细胞
    if row.cell_type == 'RBC':
        edgecolor = 'r'
        ax.annotate('RBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'WBC':
        edgecolor = 'b'
        ax.annotate('WBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'Platelets':
        edgecolor = 'g'
        ax.annotate('Platelets', xy=(xmax-40,ymin+20))
    #框出细胞
    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
    ax.add_patch(rect)