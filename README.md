# CIFAR-10 ( 影像分類問題 10 類 )
**********************************************
我使用 keras - vgg16，經過多次調整，最後上傳 Kaggle 準確率為 0.8744，DATA可以從 https://www.kaggle.com/c/cifar-10/data 下載，
結果不算好，如果有更好的方法，麻煩提供給我，謝謝

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

## 設置 train_history，利用 show_train_history 可以畫出訓練的過程 <br>
```sh
train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          validation_split = 0.2,# use 20% data to validation 
                          epochs = 10,# run 10 times
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)
show_train_history(train_history)
```

## 計算 train and test 的 confusion_matrix 正確率，model.save_weights 可以儲存訓練好的 weight <br>
```sh
train_table,tem11 = compare_corr_per(sub_train_x,sub_train_y)
# 如果沒有進行 cross_validation, 以下將不用執行
#test_table,tem21 = compare_corr_per(sub_test_x,sub_test_y)
model.save_weights('vgg16_temp_alldata.h5')
```

## 第二次訓練，與前面不同的是，lr : 1e-5 -> 1e-4，momentum : 0.2 -> 0.5，提高效率 <br>
```sh
sgd = SGD(lr=1e-4, momentum=0.5, nesterov=True)
model.compile(optimizer = sgd, 
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          validation_split = 0.2,# catch 20% data to validation 
                          epochs = 10,# run 10 times
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)

train_table,tem = compare_corr_per(sub_train_x,sub_train_y)# 計算正確率
test_table,tem2 = compare_corr_per(sub_test_x,sub_test_y)# 計算正確率
```

## 第三次訓練，再次提高效率，必須先進行小 lr 訓練，如果直接 lr = 1e-3，會訓練很爛 <br>
```sh
sgd = SGD(lr=1e-3, momentum=0.8, nesterov=True)
model.compile(optimizer = sgd, #sgd
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          #validation_split = 0.1,# catch 20% data to validation 
                          epochs = 10,# run 50 times
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)                          

train_table,tem = compare_corr_per(sub_train_x,sub_train_y)# 計算正確率
test_table,tem2 = compare_corr_per(sub_test_x,sub_test_y)# 計算正確率
```

## 以上並沒有更改 vgg 結構，下一步，我們將在 VGG16 最後端，加入以下結構

## 原始 VGG 結構，使用 model_vgg16_conv.summary() 函數，可以很清楚看出 VGG16 結構 <br>
```sh
model_vgg16_conv = VGG16(weights='imagenet', # 一樣使用 keras 提供的 weight
                         include_top=False)
model_vgg16_conv.summary()
#_______________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_20 (InputLayer)        (None, None, None, 3)     0         
#_______________________________________________________________
#block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
#_______________________________________________________________
#block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
#_______________________________________________________________
#block1_pool (MaxPooling2D)   (None, None, None, 64)    0         
#_______________________________________________________________
#block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
#_______________________________________________________________
#block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
#_______________________________________________________________
#block2_pool (MaxPooling2D)   (None, None, None, 128)   0         
#_______________________________________________________________
#block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
#_______________________________________________________________
#block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
#_______________________________________________________________
#block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
#_______________________________________________________________
#block3_pool (MaxPooling2D)   (None, None, None, 256)   0         
#_______________________________________________________________
#block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
#_______________________________________________________________
#block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
#_______________________________________________________________
#block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
#_______________________________________________________________
#block4_pool (MaxPooling2D)   (None, None, None, 512)   0         
#_______________________________________________________________
#block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
#_______________________________________________________________
#block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
#_______________________________________________________________
#block5_conv3 (Conv2D)        (None, None, None, 512)   2359808   
#_______________________________________________________________
#block5_pool (MaxPooling2D)   (None, None, None, 512)   0         
#=================================================================
#Total params: 14,714,688
#Trainable params: 14,714,688
#Non-trainable params: 0
```

## 以下更改 VGG16 結構，將提高準確率<br>
```sh
input = Input(shape=(32,
                     32,3),
                     name = 'image_input')
#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten()(output_vgg16_conv)
x = Dense(256,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='sigmoid')(x)
#Create your own model 
model = Model(input=input, output=x)
model.summary()# look my CNN architecture
```
## 更改結構後為<br>
```sh
#_______________________________________________________________
#
#_______________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#image_input (InputLayer)     (None, 32, 32, 3)         0         
#_______________________________________________________________
#   *** 這裡是原始 vgg16 結構, 可以由 Param 數量 14714688 得知
#vgg16 (Model)                multiple                  14714688
#_______________________________________________________________
#flatten_20 (Flatten)         (None, 512)               0         
#_______________________________________________________________
#dense_37 (Dense)             (None, 256)               131328    
#_______________________________________________________________
#dropout_18 (Dropout)         (None, 256)               0         
#_______________________________________________________________
#dense_38 (Dense)             (None, 10)                2570      
#=================================================================
#Total params: 14,848,586
#Trainable params: 14,848,586
#Non-trainable params: 0
#_______________________________________________________________
```

## 一樣先進行小 lr training，比較特別的是，這裡 loss 使用 binary_crossentropy，這裡提供我 train 好的 weight [vgg16_temp_alldata.h5](https://drive.google.com/file/d/0B4VP7a8ewj_2NG9NWkFFMmRKY2M/view) <br>
```sh
sgd = SGD(lr=1e-5, momentum=0.9)
model.compile(optimizer = sgd, #sgd
              loss='binary_crossentropy',
              metrics = ['accuracy'])

# use train_history, that can get model history
# 如果沒有GPU, 可以使用我 train 好的 weight
# model.load_weights('vgg16_temp_alldata.h5') 
train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          #validation_split = 0.2,# catch 20% data to validation 
						  # 不進行 validation_split, 由於 data 多, 可以 train 的更好
						  # 使用 validation_split 可以看出收斂性
                          epochs = 10,# run 20 times
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)
                          
#show_train_history(train_history)# plot training hisrtory
train_table,tem11 = compare_corr_per(sub_train_x,sub_train_y)
# 如果沒有進行 cross_validation, 以下將不用執行
#test_table,tem21 = compare_corr_per(sub_test_x,sub_test_y)
model.save_weights('vgg16_temp_alldata.h5')
```

## 再次訓練，150次，這裡提供我 train 好的 weight [vgg16_temp3_alldata.h5](https://drive.google.com/file/d/0B4VP7a8ewj_2aDAxcU1ZM0ZmQ2c/view)，可以藉由 model.load_weights 讀取<br>
```sh

model.load_weights('vgg16_temp_alldata.h5')
# 這是並沒有 1e-5 -> 1e-4 -> 1e-3, 直接 1e-5 -> 1e-3, loss 也改為 categorical_crossentropy
sgd = SGD(lr=1e-3, momentum=0.9)
model.compile(optimizer = sgd, #sgd
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
# 如果沒有GPU, 可以使用我 train 好的 weight
# model.load_weights('vgg16_temp3_alldata.h5')
train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          #validation_split = 0.2,# catch 20% data to validation 
						  # 不進行 validation_split, 由於 data 多, 可以 train 的更好
						  # 使用 validation_split 可以看出收斂性
                          epochs = 150,# run  times  150 
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)
                          
#show_train_history(train_history)# plot training hisrtory
train_table,tem13 = compare_corr_per(sub_train_x,sub_train_y)
#test_table,tem13 = compare_corr_per(sub_test_x,sub_test_y)
model.save_weights('vgg16_temp3_alldata.h5')#28
```

## 最後預測<br>
```sh
final_pred = model.predict(test_image,verbose=1)
```

## 預測結果為機率，轉成 cate，將 output 提交到 kaggle 上, 可以獲得 0.8744 準確率 <br>
```sh
cate_pred = change_prob_to_cate(final_pred)

output = {'id':index,
          'label':cate_pred}
output = pd.DataFrame(output)
output = output.sort_values('id')

output.to_csv('output.csv',index=False)
# 
```

