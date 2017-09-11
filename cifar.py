

#===============================function===============================

def input_image(file_path,labels):
    # file_path = train_image_path[0]
    # labels = train_labels
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    
    #label = file_path.replace('train/','')
    label = re.split('/',file_path)[1]
    label = int( label.replace('.png','') )
    label = labels.iloc[label-1,1]
    labels.iloc[23179-1,:]

    return img,label    


def plot_images_labels(images,labels,num = 10):
    # images = img
    # labels = label
    idx = 0
    #num = 5
    fig = plt.gcf()
    fig.set_size_inches(15,30)
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx], cmap = 'binary')
        title = 'label = ' + str(labels[idx])
        
        ax.set_title(title,fontsize = 10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx = idx+1
        
    plt.show()   
 
 
def input_train_data(train_images ,train_labels ):
    # train_images = train_image_path
    images = []
    labels_class = []
    labels_int = []

    label_dick = {'airplane':0, 
                  'automobile':1, 
                  'bird':2, 
                  'cat':3, 
                  'deer':4, 
                  'dog':5, 
                  'frog':6, 
                  'horse':7, 
                  'ship':8, 
                  'truck':9}    
    
    count = len(train_images)
    train = np.ndarray((count, 32, 32,3), dtype=np.float32)
    for i in range(len(train_images)):# i=0
        
        file_path = train_images[i]
        img,label_class  = input_image(file_path,train_labels)
        img = img.astype('float32')/255.0
        #img = (img.astype('float32') - [125.3, 123.0, 113.9]) / [63.0, 62.1, 66.7]
        label_int = label_dick[label_class]
        train[i] = img
        
        labels_class.append(label_class)
        labels_int.append(label_int)
        
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
        
    labels_int = np.array(labels_int, dtype=np.uint8)
    labels_int = kutils.to_categorical(labels_int) 
    
    return train,labels_int,labels_class

def input_test_data(test_image_path):

    images = []
    count = len(test_image_path)
    test_image = np.ndarray((count, 32, 32,3), dtype=np.float32)
    index = []
    for i in range(len(test_image_path)):
        
        file_path = test_image_path[i]
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        #if( str( type(img) ) == "<class 'NoneType'>" )
        img = img.astype('float32')/255.0
        test_image[i] = img
        file_path = file_path.replace('.png','')
        file_path = file_path.replace('test/','')
        index.append(int(file_path))
        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    return test_image,index
   
def show_train_history(train_history):#(train = 'acc', validation = 'val_acc'):
    plt.figure(figsize = (10,10)) # change figure size
    plt.plot( train_history.history['acc'] )
    plt.plot( train_history.history['val_acc'] )
    plt.title('train history')
    plt.ylabel('acc')
    plt.xlabel('Epoch')

    plt.plot( train_history.history['loss'] )
    plt.plot( train_history.history['val_loss'] )
    plt.title('train history')
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.legend(['train_acc','validation_val_acc','train_loss','validation_val_loss'],loc = 'upper left')
    #plt.legend(['train','validation'],loc = 'upper left')
    plt.show    


def build_my_cnn_two_layer():
    np.random.seed(100)
    # build cnn model
    model = Sequential()
    #------------------------------------------------
    # build convolution 1
    model.add(Conv2D(
                    filters = 32,# random build 32 filter(濾波器) weight
                    kernel_size = (3,3),# 濾波器 is 3*3
                    input_shape = (32,32,3),# image.shape is 32*32*3
                    activation = 'relu',# 激活 function is relu
                    padding='same'# image size is same
                    ))
                     
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
    
    # build pool, (2,2) is dimension reduction, 
    # ex: shape 32*32 -> 16*16, but filter doesn't change, it is 32
    model.add(MaxPooling2D(pool_size = (2,2)))
    #------------------------------------------------
    # build convolution 2
    model.add(Conv2D(
                    filters = 64,# random build 32 filter(濾波器) weight
                    kernel_size = (3,3),# 濾波器 is 3*3
                    # this doesn't need input_shape, it is auto catch conv 1
                    activation = 'relu',# 激活 function is relu
                    padding='same'# image size is same
                    ))
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
    # build pool 2
    model.add(MaxPooling2D(pool_size = (2,2)))
    #------------------------------------------------
    # build flatten(平坦層), that is one dimension
    # 64 (filters) * 8 * 8 = 4096, 8 is 32(shape) -> pool -> 16 -> pool ->8
    model.add(Flatten())
    model.add(Dropout(rate = 0.25))
    # build Dense( 隱藏層 )
    model.add(Dense(1024,activation = 'relu'))
    model.add(Dropout(rate = 0.25))
    # build output, 10 is ten categories, softmax is output probability
    model.add(Dense(10,activation = 'softmax'))
    # print our DL model
    print(model.summary())
    
    # after training, we must compile
    model.compile(loss = 'categorical_crossentropy',# loss function
                  optimizer = 'adam', # optimizer ( 優化 )
                  metrics = ['accuracy'])# evaluation
    return model

def pred_plot_fun(model,sub_train_x):

    label_dick = {0:'airplane', 
                  1:'automobile', 
                  2:'bird', 
                  3:'cat', 
                  4:'deer', 
                  5:'dog', 
                  6:'frog', 
                  7:'horse', 
                  8:'ship', 
                  9:'truck'}  
    
    train_pred = model.predict_classes(sub_train_x)
    
    train_pred_class = []
    for i in range(len(train_pred)):
        train_pred_class.append( label_dick[ train_pred[i] ] )
    
    plot_images_labels(sub_train_x,train_pred_class,5)
    
    return train_pred,train_pred_class

def show_pred_prob_plot(sub_train_x,train_pred_prob,train_pred_class,j):

    label_dick = {0:'airplane', 
                  1:'automobile', 
                  2:'bird', 
                  3:'cat', 
                  4:'deer', 
                  5:'dog', 
                  6:'frog', 
                  7:'horse', 
                  8:'ship', 
                  9:'truck'} 
    plt.imshow(sub_train_x[j])
    plt.show()
    print( 'prediction is '+train_pred_class[j] + '\n')
    
    for i in range(10):
        if( train_pred_prob[j][i] == max(train_pred_prob[j]) ):
            print(label_dick[i] + ' probability :\t' , train_pred_prob[j][i],'*')
        else:
            print(label_dick[i] + ' probability :\t' , train_pred_prob[j][i])
        
def change_categorical_to_int(sub_train_y):
    int_matrix = []
    for j in range(len(sub_train_y)):
        tem = 0
        x = sub_train_y[j]==0
        for i in range(len(x)):
            if( x[i] == 1 ):
                tem = tem+1
            else:
                int_matrix.append(tem)
                break
    int_matrix = np.array(int_matrix)
    return int_matrix
    
def build_confusion_matrix(sub_train_y,train_pred):
    # train_pred = y_pred
    labels_int_matrix = change_categorical_to_int(sub_train_y)
    pred_int = charge_prob_to_int(train_pred)
    table = pd.crosstab(pred_int,labels_int_matrix,
                        rownames = ['label'],colnames = ['predict'])
    
    print(table)
    amount = 0
    for i in range(len(table)):
        amount = amount + table.iloc[i,i]
    correct_per = amount/len(pred_int)
    #print('\n correct =',correct_per)
    return table,correct_per
  
def compare_corr_per(sub_train_x,sub_train_y):

    train_pred = model.predict(sub_train_x,verbose=1)
    
    table,train_correct_per = build_confusion_matrix(sub_train_y,train_pred)
    
    #test_pred = model.predict(sub_test_x,verbose=1)
    
    #test_correct_per  = build_confusion_matrix(sub_test_y,test_pred)
    
    print('\ncorrect_per = ' + str( train_correct_per ) )
    #print('\ntest_correct_per = '  + str( test_correct_per ) )

    return table,train_correct_per


def build_my_cnn_three_layer():
    # build cnn model
    model = Sequential()
    #------------------------------------------------
    # build convolution 1
    model.add(Conv2D(
                    filters = 32,# random build 32 filter(濾波器) weight
                    kernel_size = (3,3),# 濾波器 is 3*3
                    input_shape = (32,32,3),# image.shape is 32*32*3
                    activation = 'relu',# 激活 function is relu
                    padding='same'# image size is same
                    ))
                     
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
    
    # build pool, (2,2) is dimension reduction, 
    # ex: shape 32*32 -> 16*16, but filter doesn't change, it is 32
    model.add(MaxPooling2D(pool_size = (2,2)))
    #------------------------------------------------
    # build convolution 2
    model.add(Conv2D(
                    filters = 64,# random build 32 filter(濾波器) weight
                    kernel_size = (3,3),# 濾波器 is 3*3
                    # this doesn't need input_shape, it is auto catch conv 1
                    activation = 'relu',# 激活 function is relu
                    padding='same'# image size is same
                    ))
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
    # build pool 2
    model.add(MaxPooling2D(pool_size = (2,2)))
    #------------------------------------------------
    # build convolution 3
    model.add(Conv2D(
                    filters = 128,# random build 32 filter(濾波器) weight
                    kernel_size = (3,3),# 濾波器 is 3*3
                    # this doesn't need input_shape, it is auto catch conv 1
                    activation = 'relu',# 激活 function is relu
                    padding='same'# image size is same
                    ))
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
    # build pool 2
    model.add(MaxPooling2D(pool_size = (2,2)))
    #------------------------------------------------    
    # build flatten(平坦層), that is one dimension
    # 64 (filters) * 4 * 4 = 4096, 8 is 32(shape) -> three pool -> 4
    model.add(Flatten())
    model.add(Dropout(rate = 0.3))
    # build Dense( 隱藏層 )
    model.add(Dense(2048,activation = 'relu'))
    model.add(Dropout(rate = 0.3))
    
    model.add(Dense(1024,activation = 'relu'))
    model.add(Dropout(rate = 0.3))    
    # build output, 10 is ten categories, softmax is output probability
    model.add(Dense(10,activation = 'softmax'))
    # print our DL model
    print(model.summary())
    
    # after training, we must compile
    model.compile(loss = 'categorical_crossentropy',# loss function
                  optimizer = 'adam', # optimizer ( 優化 )
                  metrics = ['accuracy'])# evaluation
    return model


def charge_prob_to_int(final_pred):
    # final_pred = y_pred
    int_matrix = []
    for j in range(len(final_pred)):
        tem = 0
        x = final_pred[j] == max(final_pred[j])
        for i in range(len(x)):
            if( x[i] == 0 ):
                tem = tem+1
            else:
                int_matrix.append(tem)
                break
    int_matrix = np.array(int_matrix)
    return int_matrix
    


def change_prob_to_cate(final_pred):
    # final_pred = y_pred
    cate_matrix = []
    for j in range(len(final_pred)):
        tem = 0
        x = final_pred[j] == max(final_pred[j])
        for i in range(len(x)):
            if( x[i] == 0 ):
                tem = tem+1
            else:
                cate_matrix.append(tem)
                break
    cate_matrix = np.array(cate_matrix)
    
    label_dick = {0:'airplane', 
                  1:'automobile', 
                  2:'bird', 
                  3:'cat', 
                  4:'deer', 
                  5:'dog', 
                  6:'frog', 
                  7:'horse', 
                  8:'ship', 
                  9:'truck'}     
    cate_pred = []
    for i in range(len(cate_matrix)):
        tem = label_dick[cate_matrix[i]]
        cate_pred.append(tem)
    cate_pred = np.array(cate_pred)
    
    return cate_pred


#===============================main===============================


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

os.chdir('/home/linsam/kaggle/cifar_10')
train_labels = pd.read_csv('trainLabels.csv')
train_image_path = ['train/'+i for i in os.listdir('train/')]
test_image_path = ['test/'+i for i in os.listdir('test/')]
print('train_image_amount : ' + str( len(train_image_path) ) )
print('test_image_amount : ' + str( len(test_image_path) ) )

#====================== step 1 : input data =============================
#train_image_path = train_image_path 
# input image data and change labels to onehot encoding (indicator matrix)
train_image,train_labels_int,train_labels_class = input_train_data(train_image_path,train_labels)

# plot image
#plot_images_labels(train_image,train_labels_class,20)
#----------------------------------------------------------------------
# cross validation
print( len(train_image) )
n = int( len(train_image)*0.9 )
sub_train_x = train_image[:n]
sub_train_y = train_labels_int[:n]

sub_test_x = train_image[n:]
sub_test_y = train_labels_int[n:]


#----------------------------------------------------------------------
#====================== step 6 : build vgg16 model =============================
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import SGD

sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
#adam = keras.optimizers.Adam(lr=0.001, 
#                             beta_1=0.9, 
#                             beta_2=0.999, 
#                             epsilon=1e-08, 
#                             decay=0.0)
                             
# weights = None is retraining, weights='imagenet' is call weights by vgg16
model_vgg16_conv = VGG16(weights='imagenet', 
                         include_top=False)
#model_vgg16_conv = VGG16(weights = None, 
#                         include_top=False)                         
#model_vgg16_conv.summary()

# set input shape 32*32*3
input = Input(shape=(sub_train_x.shape[1],
                     sub_train_x.shape[2],3),
                     name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
#x = Dropout(rate = 0.3)(x)
#x = Dense(2048, activation='relu')(x)
#x = Dropout(rate = 0.3)(x)
#x = Dense(1024, activation='relu')(x)
#x = Dropout(rate = 0.3)(x)
x = Dense(10, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(input=input, output=x)

#model.compile(sgd, loss='binary_crossentropy')
                         
model.compile(optimizer = sgd, #sgd
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
#model.compile(optimizer = sgd, loss='categorical_crossentropy',metrics = ['accuracy'])

# use train_history can get model history
#model.load_weights('2017-09-10_cifar_10.h5')
train_history = model.fit(sub_train_x,# train x ( feature )
                          sub_train_y,# train y ( label or target )
                          validation_split = 0.2,# catch 20% data to validation 
                          epochs = 10,# run 10 times
                          batch_size = 128,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)
model.save_weights('vgg16_cifar_10_2.h5')                          
#====================== step 7 : evaluate model =============================
show_train_history(train_history)

train_table,tem = compare_corr_per(sub_train_x,sub_train_y)

test_table,tem2 = compare_corr_per(sub_test_x,sub_test_y)


#--------------------------------------------------------------------------
# input test data
test_image,index = input_test_data(test_image_path)

final_pred = model.predict(test_image,verbose=1)

cate_pred = change_prob_to_cate(final_pred)

output = {'id':index,
          'label':cate_pred}
output = pd.DataFrame(output)
output = output.sort_values('id')

output.to_csv('output.csv',index=False)














































