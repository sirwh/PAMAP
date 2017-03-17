########################Building Networks#########################################
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Convolution1D, AveragePooling1D, BatchNormalization
from keras.optimizers import SGD

# declare the model
model1 = Sequential()
model1.add(Convolution1D(8, 5, input_shape=(256,3))) # input=256, output=252
#model1.add(BatchNormalization(axis=1))
model1.add(Activation('relu'))
model1.add(AveragePooling1D(pool_length=2, stride=2)) # input=252, output=126
model1.add(Dropout(0.25))
model1.add(Convolution1D(4, 5)) # input=126, output=122
#model1.add(BatchNormalization(axis=1))
model1.add(Activation('relu'))
model1.add(AveragePooling1D(pool_length=2, stride=2)) # input=122, output=61
model1.add(Dropout(0.25))

model2 = Sequential()
model2.add(Convolution1D(8, 5, input_shape=(256,3))) # input=256, output=252
#model2.add(BatchNormalization(axis=1))
model2.add(Activation('relu'))
model2.add(AveragePooling1D(pool_length=2, stride=2)) # input=252, output=126
model2.add(Dropout(0.25))
model2.add(Convolution1D(4, 5)) # input=126, output=122
#model2.add(BatchNormalization(axis=1))
model2.add(Activation('relu'))
model2.add(AveragePooling1D(pool_length=2, stride=2)) # input=122, output=61
model2.add(Dropout(0.25))

model3 = Sequential()
model3.add(Convolution1D(8, 5, input_shape=(256,3))) # input=256, output=252
#model3.add(BatchNormalization(axis=1))
model3.add(Activation('relu'))
model3.add(AveragePooling1D(pool_length=2, stride=2)) # input=252, output=126
model3.add(Dropout(0.25))
model3.add(Convolution1D(4, 5)) # input=126, output=122
#model3.add(BatchNormalization(axis=1))
model3.add(Activation('relu'))
model3.add(AveragePooling1D(pool_length=2, stride=2)) # input=122, output=61
model3.add(Dropout(0.25))

merged = Merge([model1, model2, model3], mode='concat', concat_axis=1)

final_model = Sequential()
final_model.add(merged)
final_model.add(Flatten()) # to reshaple the input to 1-dimensional
final_model.add(Dense(732)) # hidden layer, not sure about the activation
#final_model.add(BatchNormalization(axis=-1))
final_model.add(Dropout(0.5))
final_model.add(Dense(4, activation='softmax')) # output layer

sgd = SGD(lr=0.01, momentum=0.9, decay=0.05) # parameter of stochastic gradient descent 
final_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) # multi-class log loss

# save the model
final_model.save('PAMAP_new_alldim.h5') # save in the current folder


############################Training#####################################
import numpy as np
import scipy.io as sio
np.random.seed(123)
from keras.utils.np_utils import to_categorical
from keras.models import load_model

def data_concat(data, exception, dim): # make the data format fit the network input
    # data: the input data
    # exception: the index of test data,should be 1 number from 0~i-1 (there are totally i subjects)
    # dim: the index of acceleration data to be used, should be any 3 number from 0~8 (3 IMU * 3-axis acceleration)
    size = sum(len(i) for i in data) # get the number of subjects, should be 7 in our case
    data_train=[]
    for i in xrange(0,size): # loop for each subject
        if i!=exception: # assign data for training
            if len(data_train)==0:
                data_train = data[0,i]
            else:
                data_train=np.concatenate((data_train, data[0,i]),axis=0) # concat array in axis=0 (vertically)
        else: # assign data for test
            data_test=data[0,i]
    # now data_train has i-1 subjects and data_test has 1 subject
    data_train = data_train[:,dim]
    data_test = data_test[:,dim]
    # now data_train or data_test is (6L, 3L), and each [i,j] is (n obersavions, 1 label + 256 steps)
    x_train=[data_train[0,0][:,1:],data_train[0,1][:,1:],data_train[0,2][:,1:]] # the 1st element of x_train
    y_train=data_train[0,0][:,0].tolist() # the 1st element of y_train
    for j in xrange(1,len(data_train)): # x_train becomes (n obersavions for all training subjects, 256 steps)*3 dim
        for k in xrange(0,3):
            x_train[k]=np.concatenate(([x_train[k],data_train[j,k][:,1:]]),axis=0)
        y_train=y_train+data_train[j,0][:,0].tolist() # y_train becomes (n obersavions for all training subjects)
    x_test=[data_test[0,0][:,1:],data_test[0,1][:,1:],data_test[0,2][:,1:]]
    y_test=data_test[0,0][:,0].tolist()
    # preprocess the input data
    for j in xrange(3):
        x_train[j] = x_train[j].astype('float32').reshape(x_train[j].shape+(1,))
        x_test[j] = x_test[j].astype('float32').reshape(x_test[j].shape+(1,))
    x_train=np.concatenate((x_train[0],x_train[1],x_train[2]),axis=2)
    x_test=np.concatenate((x_test[0],x_test[1],x_test[2]),axis=2)
    y_train=np.asarray(y_train).astype('int')
    y_test=np.asarray(y_test).astype('int')
    # preprocess the data lable
    Y_train = to_categorical(y_train, 4)
    Y_test = to_categorical(y_test, 4)
    return x_train, x_test, Y_train, Y_test

mat_contents=sio.loadmat('input_data.mat') # load the data
input_data = mat_contents['input_data']
batch=32
ep=100
# train the model, using leave-one-out validation (we have 7 subjects)
for i in xrange(7): 
    final_model = load_model('PAMAP_new_alldim.h5')	# load the model
    x_train1, x_test1, Y_train, Y_test = data_concat(input_data, i, range(0,3))
    x_train2, x_test2, Y_train, Y_test = data_concat(input_data, i, range(3,6))
    x_train3, x_test3, Y_train, Y_test = data_concat(input_data, i, range(6,9))
    final_model.fit([x_train1,x_train2,x_train3], Y_train, batch_size=batch, nb_epoch=ep) # batch=32, epoch=100 trains the model fast
    final_model.save('PAMAP_new_alldim_b'+str(batch)+'e'+str(ep)+'_'+str(i)+'.h5')

#############################Test####################################
mat_contents=sio.loadmat('input_data.mat')
input_data = mat_contents['input_data']
batch=32
ep=100
# test the model, using leave-one-out validation
for i in xrange(7):
    model = load_model('PAMAP_new_alldim_b'+str(batch)+'e'+str(ep)+'_'+str(i)+'.h5')
    x_train1, x_test1, Y_train, Y_test = data_concat(input_data, i, range(0,3))
    x_train2, x_test2, Y_train, Y_test = data_concat(input_data, i, range(3,6))
    x_train3, x_test3, Y_train, Y_test = data_concat(input_data, i, range(6,9))
    score = model.evaluate([x_train1,x_train2,x_train3], Y_train, verbose=0)
    score # train acc
    score = model.evaluate([x_test1,x_test2,x_test3], Y_test, verbose=0)
    score # test acc
    
#############################Some Results####################################	
'''
when epoch=100, batchsize=32, use dropout, run on dim0~2 (saved as PAMAP_drop_b32e100_012_i.h5)
train acc: 0.78, 0.76, 0.77, 0.79, 0.81, 0.76, 0.79
test acc:  0.53, 0.77, 0.77, 0.79, 0.56, 0.74, 0.76 (avg 0.70)

when epoch=100, batchsize=32, use dropout, run on dim3~5 (saved as PAMAP_drop_b32e100_345_i.h5)
train acc: 0.87, 0.89, 0.88, 0.89, 0.88, 0.89, 0.87
test acc:  0.86, 0.74, 0.89, 0.86, 0.90, 0.90, 0.88 (avg 0.86)

when epoch=100, batchsize=32, use dropout, run on dim6~8 (saved as PAMAP_drop_b32e100_678_i.h5)
train acc: 0.91, 0.89, 0.87, 0.88, 0.89, 0.90, 0.90
test acc:  0.83, 0.89, 0.81, 0.90, 0.90, 0.85, 0.81 (avg 0.86)

as a reference, if we only use the MLP part of the CNN (line 33~41), the accuracy is:
when epoch=100, batchsize=32, use dropout, run on dim3~5 (saved as PAMAP_dense_drop_b32e100_345_i.h5)
train acc: 0.76, 0.79, 0.76, 0.76, 0.78, 0.77, 0.80
test acc:  0.77, 0.20, 0.75, 0.75, 0.54, 0.74, 0.39 (avg 0.59)

so the CNN model has a 0.27 accuracy boost compared to simple MLP

when epoch=100, batchsize=32, use dropout, run on dim6~8 as a single input (saved as PAMAP_new_b32e100_678_i.h5)
train acc: 0.88, 0.87, 0.87, 0.87, 0.88, 0.88, 0.89
test acc:  0.85, 0.88, 0.87, 0.86, 0.84, 0.86, 0.75 (avg 0.84)

when epoch=100, batchsize=32, use dropout, run on dimALL (0~2 as input1, 3~5 as input2, 6~8 as input 3)
train acc: 0.92, 0.94, 0.93, 0.92, 0.93, 0.94, 0.94
test acc:  0.80, 0.88, 0.92, 0.89, 0.92, 0.80, 0.87 (avg 0.87)

when epoch=100, batchsize=32, use dropout, run on dimALL, using ReLU (saved as PAMAP_new_alldim_b32e100_678_i.h5)
train acc: 0.96, 0.96, 0.97, 0.96, 0.96, 0.96, 0.96
test acc:  0.90, 0.93, 0.93, 0.95, 0.91, 0.95, 0.93 (avg 0.93)!
