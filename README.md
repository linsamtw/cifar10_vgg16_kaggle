# CIFAR-10 ( 影像分類問題 10 類 )
**********************************************
我使用 keras - vgg16 ，最後上傳 Kaggle 準確率為 0.8744，DATA可以從 https://www.kaggle.com/c/cifar-10/data 下載

MY GPU is GTX-1070
**********************************************

首先，下圖是其中一張的 image <br><br>
 ![horse](https://github.com/f496328mm/cifar10_vgg16_kaggle/blob/master/horse1.png)

馬的圖片，大小為 32*32*3，我們將訓練 vgg 16 model ，學會辨識

## packages，其中安裝 CV2 可能會遇到問題，不少人遇到相同問題，可以 google 找找看解決方法<br>
```sh
import numpy as np
np.random.seed(100)
import os, cv2, random
import pandas as pd
%matplotlib inline 
import matplotlib.pyplot as plt
import keras.utils.np_utils as kutils
import re
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation, Conv2DTranspose
from keras.layers import Conv2D
import keras
import datetime
import math
import gc
```

## 抓取 all image 的檔案名稱，5萬張 train data， 30萬張 test data <br>
```sh
os.chdir('/home/linsam/kaggle/cifar_10') # 設定資料夾位置
train_labels = pd.read_csv('trainLabels.csv')# 讀進 labels
# 抓取 train 資料夾中所有檔案名稱
train_image_path = ['train/'+i for i in os.listdir('train/')] 
# 抓取 test 資料夾中所有檔案名稱
test_image_path = ['test/'+i for i in os.listdir('test/')]
# 印出 image 數量
print('train_image_amount : ' + str( len(train_image_path) ) )
print('test_image_amount : ' + str( len(test_image_path) ) )
```
## input image data by path <br>
```sh
# input train data
train_image,train_labels_int,train_labels_class = (
    input_train_data(train_image_path,train_labels) )
# input test data    
test_image,index = input_test_data(test_image_path)
```
## 做 cross_validation，最後比例改成 1，因為 data 越多結果越好<br>
```sh
print( train_image.shape )
n = int( len(train_image)*1 )
sub_train_x = train_image[:n]
sub_train_y = train_labels_int[:n]

sub_test_x = train_image[n:]
sub_test_y = train_labels_int[n:]
```

## 使用 VGG 16 model <br>
```sh
# function 雖然有寫 cnn two/three layer, 但是結果沒有 vgg 好
#====================== step 6 : build vgg16 model =============================
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD
```

## weights = None 代表重新訓練 weight ， weights='imagenet' 代表使用 keras vgg16 提供的 weight <br>
```sh
# VGG16 可以換成 VGG19, 但是不保證較好
model_vgg16_conv = VGG16(weights='imagenet', 
                         include_top=False)
#model_vgg16_conv = VGG16(weights = None, 
#                         include_top=False)                         
```

## 使用 vgg16 ， 需要設定 image 的 shape，該 image 的 shape is 32*32*3 <br>
```sh
# set input shape, 32*32*3
# 使用 vgg16 or 其他人的 model, 需要給定自己 image 的 shape
input = Input( shape=(32,32,3),name = 'image_input' )

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)
```

## 由於該問題是分 10 類，需要在最後修改 Dense(10)，softmax 代表 output 是機率  <br>
```sh
#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(10, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(input=input, output=x)
```

## lr is learn rate，momentum 越大代表 training 越積極，categorical_crossentropy 代表多元分類，accuracy 是一種 evaluation <br>
```sh
sgd = SGD(lr=1e-5, momentum=0.2, nesterov=True)
model.compile(optimizer = sgd, #sgd
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
```

## 設置 train_history， 可以看出訓練的過程 <br>
```sh
train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          validation_split = 0.2,# use 20% data to validation 
                          epochs = 10,# run 10 times
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)
```

## 分割完後的圖片<br>
```sh
img1 = my_plt_fun(x_split_start,x_split_end,0)
plt.imshow(img1)
```

## 分割完後的圖片<br>
```sh
img1 = my_plt_fun(x_split_start,x_split_end,0)
plt.imshow(img1)
```


## 分割完後的圖片<br>
```sh
img1 = my_plt_fun(x_split_start,x_split_end,0)
plt.imshow(img1)
```

## 儲存<br>
```sh
for i in range(len(x_split_start)):
    my_plt_fun(x_split_start,x_split_end,i)
```


