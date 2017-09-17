

# 讀 image
def input_image(file_path,labels):
    # file_path = train_image_path[0]
    # labels = train_labels
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)# input image
    
    #label = file_path.replace('train/','')
    label = re.split('/',file_path)[1]# catch label
    label = int( label.replace('.png','') )# label translate int
    label = labels.iloc[label-1,1]# index - 1, because py start 0
    #labels.iloc[23179-1,:]

    return img,label    # 回傳 image and label

# 畫圖, 畫 image and 上 label
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
 
# input train data
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
    #圖片數量
    count = len(train_images)
	# 先初始化 train 空間, 圖片大小為32*32*3
    train = np.ndarray((count, 32, 32,3), dtype=np.uint8) 
    for i in range(len(train_images)):# i=0
        
        file_path = train_images[i]# catch image 位置
		# 利用 input_image 抓 image and label
        img,label_class  = input_image(file_path,train_labels)
        #img = img.astype('float32')#/255.0 # 效果不好
        #img = (img.astype('float32') - [125.3, 123.0, 113.9]) / [63.0, 62.1, 66.7]
        label_int = label_dick[label_class]# 原先是 class , 轉成數字便於建 model
        train[i] = img # 原先初始化的空間, 存放 image 矩陣
        
        labels_class.append(label_class)# label 的類別
        labels_int.append(label_int)# label 轉成 int
        # 印出讀檔過程
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
		
    # 轉 array, DL 需要的型態    
    labels_int = np.array(labels_int, dtype=np.uint8)
	# one hot encoding, 0~9 轉成 01 矩陣 
    labels_int = kutils.to_categorical(labels_int) 
    
    return train,labels_int,labels_class

# input test 跟 input train 一樣, 只差在沒有 label
def input_test_data(test_image_path):

    images = []
    count = len(test_image_path)
    test_image = np.ndarray((count, 32, 32,3), dtype=np.uint8)
    index = []
    for i in range(len(test_image_path)):
        
        file_path = test_image_path[i]
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        #if( str( type(img) ) == "<class 'NoneType'>" )
        #img = img.astype('float32')#/255.0
        test_image[i] = img
        file_path = file_path.replace('.png','')
        file_path = file_path.replace('test/','')
        index.append(int(file_path))
        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    return test_image,index
   
# plot training 過程   
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

# 建立 cnn two 層
def build_my_cnn_two_layer():
    np.random.seed(100)
    # build cnn model
    model = Sequential()
    #------------------------------------------------
    # build convolution 1
    model.add(Conv2D(
                    filters = 32,# random build 32 filter(濾波器) weight
                    kernel_size = (3,3),# 濾波器 is 3*3 size
                    input_shape = (32,32,3),# image.shape is 32*32*3
                    activation = 'relu',# 激活 function is relu
                    padding='same'# image size is same
                    ))
                     
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
    
    # build pool, (2,2) is dimension reduction, 
	# (2,2) --dimension reductioin ->1
    # ex: shape 32*32 -> 16*16, but filter doesn't change, it still is 32
    model.add(MaxPooling2D(pool_size = (2,2)))
    #------------------------------------------------
    # build convolution 2
    model.add(Conv2D(
					# filters 要*2, 32 -> 64, 如果沒*2, 那 1->1 沒有效果
                    filters = 64,# random build 64 filter(濾波器) weight
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
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
    # build Dense( 隱藏層 )
    model.add(Dense(1024,activation = 'relu'))
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
	
    # build output, 10 is ten categories, softmax is output probability
    model.add(Dense(10,activation = 'softmax'))
    # print our DL model structure
    print(model.summary())
    
    # after training, we must compile, 
	# you can choose categorical_crossentropy or binary_crossentropy
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
    # prediction to classes
    train_pred = model.predict_classes(sub_train_x)
    
    train_pred_class = []
    for i in range(len(train_pred)):
        train_pred_class.append( label_dick[ train_pred[i] ] )
    
    plot_images_labels(sub_train_x,train_pred_class,5)
    
    return train_pred,train_pred_class

# show all prob in one image
# 一張圖的機率可能是 airplane 0.9, bird 0.05 ...etc
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

# translate categorical to int, in order to confusion_matrix
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
	
# 計算correct正確率
def compare_corr_per(sub_train_x,sub_train_y):

    train_pred = model.predict(sub_train_x,verbose=1)
    
    table,train_correct_per = build_confusion_matrix(sub_train_y,train_pred)
    
    #test_pred = model.predict(sub_test_x,verbose=1)
    
    #test_correct_per  = build_confusion_matrix(sub_test_y,test_pred)
    
    print('\ncorrect_per = ' + str( train_correct_per ) )
    #print('\ntest_correct_per = '  + str( test_correct_per ) )

    return table,train_correct_per

# 建立三層 cnn
def build_my_cnn_three_layer(sgd):
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
					# filters 要*2, 32 -> 64, 如果沒*2, 那 1->1 沒有效果
                    filters = 64,# random build 64 filter(濾波器) weight
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
					# filters 要*2, 64 -> 128, 如果沒*2, 那 1->1 沒有效果
                    filters = 128,# random build 128 filter(濾波器) weight
                    kernel_size = (3,3),# 濾波器 is 3*3
                    # this doesn't need input_shape, it is auto catch conv 1
                    activation = 'relu',# 激活 function is relu
                    padding='same'# image size is same
                    ))
    model.add(Dropout(rate = 0.25))# drop 25% net, avoid overfitting
    # build pool 3
    model.add(MaxPooling2D(pool_size = (2,2)))
    #------------------------------------------------    
    # build flatten(平坦層), that is one dimension
    # 64 (filters) * 4 * 4 = 1024, 4 is 32(shape) -> three pool -> 4
    model.add(Flatten())
    model.add(Dropout(rate = 0.3))
    # build Dense( 隱藏層 )
    model.add(Dense(2048,activation = 'relu'))# 2048 可能有問題
    model.add(Dropout(rate = 0.3))
    
    model.add(Dense(1024,activation = 'relu'))
    model.add(Dropout(rate = 0.3))    
    # build output, 10 is ten categories, softmax is output probability
    model.add(Dense(10,activation = 'softmax'))
    # print our DL model
    print(model.summary())
    #sgd = SGD(lr=1e-3, momentum=0.2, nesterov=True)
    # after training, we must compile
    #model.compile(loss = 'categorical_crossentropy',# loss function
                  #optimizer = 'adam', # optimizer ( 優化 )
    #              optimizer = sgd, # optimizer ( 優化 )
    #              metrics = ['accuracy'])# evaluation
    return model


def build_my_cnn_three_and_zero_layer():
    # build cnn model
    model = Sequential()
    #------------------------------------------------
	#****************************   
	# add zero layer
    model.add(ZeroPadding2D((1, 1), input_shape=(32,32,3)))
	#****************************
    # build convolution 1
    model.add(Conv2D(
                    filters = 32,# random build 32 filter(濾波器) weight
                    kernel_size = (3,3),# 濾波器 is 3*3
                    #input_shape = (32,32,3),# image.shape is 32*32*3
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
    # 64 (filters) * 4 * 4 = 4096, 4 is 32(shape) -> three pool -> 4
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
    

# beacuse final output must be categorical
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


# =============================================
# work for image tran
# 因為 max pool 是 2*2 大小, 遇到邊界會有問題, filter 也是, 因此加邊界 
def add_boundary(x,row=36,col=36):
    # x = cv2.imread(train_image_path[0], cv2.IMREAD_COLOR)
    #x.astype(np.float32)
    #plt.imshow(x)
    # x = test_image[92969]
    
    x2 = np.ndarray((row, col,3), dtype=np.uint8)#, dtype=np.float32)
    #x2[:,:,:] = 0
    #plt.imshow(x2)
    #x.shape
    start = int( (row-x.shape[0])/2 )
    end = start+x.shape[0]
    x2[start:end,start:end,:3] = x[:,:,:]
    #x2[4:36,4:36,1] = x[:,:,1]
    #x2[4:36,4:36,2] = x[:,:,2]
    #plt.imshow(x2[4:36,4:36,:3])
    #plt.imshow(x[:32,:32,:3])
    x2[:start,:,:] = 0
    x2[end:,:,:] = 0
    x2[:,:start,:] = 0
    x2[:,end:,:] = 0
    return x2    

# 灰度化, del 雜點, 加邊界
def image_gray_del_mis_add_boundary(im2,row = 36,col=36):

    #im2 = tem[3000]
    #train_labels_class[2000]
    #plt.imshow(im2)
    #灰度化处理
    retval, im4 = cv2.threshold(im2, 115, 255, cv2.THRESH_BINARY_INV)
    #retval, im4 = cv2.threshold(im2, 100, 250, cv2.THRESH_BINARY_INV)
    #plt.imshow(im4)
    # 去雜點 
    im6 = del_mis_pt_by_threshold(im4)  
    #plt.imshow(im6)
    # 雜點去除後，我們必須對剩下的影像做強化
    im8 = cv2.dilate(im6, (2, 2), iterations=1)
    #plt.imshow(im8)
    # add boundary
    im10 = add_boundary(im8, row ,col)
    #plt.imshow(im10)

    return(im10)
# 
def del_mis_pt_by_threshold(im2):
    for col in range(3):
        count = 0
        for i in xrange(len(im2)):
            for j in xrange( len(im2[i]) ):
                if( im2[i,j,col] == 255):
                   count = 0 
                   for k in range(-2,3):
                       #print(k)
                       for l in range(-2,3):
                           try:
                               if im2[i + k,j + l,col] == 255:
                                   count += 1
                           except IndexError:
                               pass
        			# 這裡 threshold 設 4，當周遭小於 4 個點的話視為雜點
                if count <= 4:
                    im2[i,j,col] = 0
    return im2

def xrange(x):

    return iter(range(x))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



