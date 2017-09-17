
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

# DATA可以從 https://www.kaggle.com/c/cifar-10/data 這裡下載

os.chdir('/home/linsam/kaggle/cifar_10') # 設定資料夾位置
train_labels = pd.read_csv('trainLabels.csv')# 讀進 labels
# 抓取 train 資料夾中所有檔案名稱
train_image_path = ['train/'+i for i in os.listdir('train/')] 
# 抓取 test 資料夾中所有檔案名稱
test_image_path = ['test/'+i for i in os.listdir('test/')]
# 印出 image 數量
print('train_image_amount : ' + str( len(train_image_path) ) )
print('test_image_amount : ' + str( len(test_image_path) ) )

#====================== step 1 : input data =============================
# train_image_path = train_image_path 
# input image data and change labels to onehot encoding (indicator matrix)
# input train data
train_image,train_labels_int,train_labels_class = (
    input_train_data(train_image_path,train_labels) )
# input test data    
test_image,index = input_test_data(test_image_path)

# 影像處理, 32*32 -> 36*36, 因為加邊界, 不過結果不好 
#train_image2 = np.ndarray((len(train_image),36, 36,3), dtype=np.uint8)
#for i in range(len(train_image)):
#    train_image2[i] = add_boundary(train_image[i])

# 影像處理, 32*32 -> 36*36, 因為加邊界, 不過結果不好 
#test_image2 = np.ndarray(( len(test_image) ,36, 36,3), dtype=np.uint8)
#for i in range(len(test_image)):
#    test_image2[i] = add_boundary(test_image[i])

#====================================================================
#plt.imshow(train_image[0])
# /255是 正規化, .0 代表浮點數, 不過結果不好
#train_image = train_image/255.0 
#test_image = test_image/255.0

#畫圖看看 
plt.imshow(train_image[0])

#====================================================================    

# plot image
#plot_images_labels(train_image,train_labels_class,20)
#----------------------------------------------------------------------
# 做 cross_validation, 最後改成1, 因為 data 越多結果越好
print( train_image.shape )
n = int( len(train_image)*1 )
sub_train_x = train_image[:n]
sub_train_y = train_labels_int[:n]

sub_test_x = train_image[n:]
sub_test_y = train_labels_int[n:]
#畫圖看看 
plt.imshow(sub_train_x[5612])
#----------------------------------------------------------------------
# function 雖然有寫 cnn two/three layer, 但是結果沒有 vgg 好
#====================== step 6 : build vgg16 model =============================
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD

# weights = None is retraining, weights='imagenet' is call weights by vgg16
# weights = None 代表重新訓練, weights='imagenet' 代表使用 keras vgg16 提供的 weight
# VGG16 可以換成 VGG19, 但是結果並不保證較好
model_vgg16_conv = VGG16(weights='imagenet', 
                         include_top=False)
#model_vgg16_conv = VGG16(weights = None, 
#                         include_top=False)                         

# set input shape, 32*32*3
# 使用 vgg16 or 其他人的 model, 需要給定自己 image 的 shape
input = Input(shape=(sub_train_x.shape[1],
                     sub_train_x.shape[2],3),
                     name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
# 使用 vgg, 需要在最後改成自己要的 output 類別數, 在這是做分 10 類
# softmax 代表 output 是 prob
x = Dense(10, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(input=input, output=x)

#model.compile(sgd, loss='binary_crossentropy')
# lr is learn rate, momentum 越大代表 training 越積極
sgd = SGD(lr=1e-5, momentum=0.2, nesterov=True)
model.compile(optimizer = sgd, #sgd
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()# look vgg structure

# use train_history, that can get model history
train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          validation_split = 0.2,# use 20% data to validation 
                          epochs = 10,# run 10 times
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)

train_table,tem = compare_corr_per(sub_train_x,sub_train_y)# 計算正確率
test_table,tem2 = compare_corr_per(sub_test_x,sub_test_y)# 計算正確率
#-----------------------------------------------------------------
# lr is learn rate, momentum 越大代表 training 越積極
sgd = SGD(lr=1e-4, momentum=0.5, nesterov=True)
model.compile(optimizer = sgd, #sgd
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
#model.compile(optimizer = sgd, loss='categorical_crossentropy',metrics = ['accuracy'])

# use train_history can get model history
#model.load_weights('vgg16_cifar_10.h5')# load training weights
train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          validation_split = 0.2,# catch 20% data to validation 
                          epochs = 10,# run 10 times
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)

train_table,tem = compare_corr_per(sub_train_x,sub_train_y)# 計算正確率
test_table,tem2 = compare_corr_per(sub_test_x,sub_test_y)# 計算正確率
#model.save_weights('vgg16_cifar_10.h5')# save training weights
#-----------------------------------------------------------------
# lr is learn rate, momentum 越大代表 training 越積極
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
#-----------------------------------------------------------------
# 需要注意的是, 需要先進行 learn rate = 1e-5, 較小的 training
# 如果直接 lr = 1e-3, 則 training 會很差

#-----------------------------------------------------------------
#====================== step 7 : final model =============================
# 以上並沒有更改 vgg 結構, 這一步, 我們將在 vgg 最後端, 加入以下結構

model_vgg16_conv = VGG16(weights='imagenet', # 一樣使用 keras 提供的 weight
                         include_top=False)
						 
# model_vgg16_conv.summary()
# 以下是原始 VGG 結構
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

# 以下更改 vgg 結構
input = Input(shape=(sub_train_x.shape[1],
                     sub_train_x.shape[2],3),
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
# 更改結構後為
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

#----------------------------------------------------------------------
# 一樣先進行小 lr training, 比較特別的是, 這裡 loss 使用 binary_crossentropy
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

#----------------------------------------------------------------------
#----------------------------------------------------------------------

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

#----------------------------------------------------------------------
#----------------------------------------------------------------------

# model.load_weights('vgg16_temp3_alldata.h5')#200
# 最後預測
final_pred = model.predict(test_image,verbose=1)

cate_pred = change_prob_to_cate(final_pred)

output = {'id':index,
          'label':cate_pred}
output = pd.DataFrame(output)
output = output.sort_values('id')

output.to_csv('output.csv',index=False)

# 將 output 提交到 kaggle 上, 可以獲得 0.8744














