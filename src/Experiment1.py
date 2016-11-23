#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 19:53:00 2016

@author: heslote1
"""

  # Assumption: if you are not seizing you are pre-seize
        
        # Prepare for Neural Network
        # Take last 10 seconds
        # Vectorize matrix 10*23 points
        # Predict
        
#import progressbar        
#from time import sleep

# LSTM for international airline passengers problem with regression framing
import numpy as np
import scipy as sp 
import os
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics

from keras.models import load_model


# convert an array of values into a dataset matrix
look_back = 10 # 10 seconds of data required
look_fwd  = 30 # 30 seconds into the future
s_period  = 1  # 1 second per sample 

def create_NN_data(data):
    dataX, dataY = [], []
    for j in range(data['seiz'][0].shape[0]-look_back):  
        dataX.append( np.array([np.concatenate((
            data['freq'][0][j:(j+look_back), :].flatten(),
            data['freq'][1][j:(j+look_back), :].flatten(),
            data['amp' ][0][j:(j+look_back), :].flatten(),
            data['amp' ][1][j:(j+look_back), :].flatten())).tolist()])
            )
        dataY.append(np.array([[
            data['seiz'][0][j,1],
            data['seiz'][1][j,1],
            data['seiz'][2][j,1]]]).astype(float)
            )

    return dataX, dataY
 
# https://github.com/fchollet/keras/issues/369
def halfsse(y_true, y_pred):
    '''1/2*Sum_Squared_Error'''
    from keras import backend as K
    E = K.square(y_pred - y_true)/2
    return E
    
# fix random seed for reproducibility
np.random.seed(7)

root  = "/Users/heslote1/www.physionet.org/pn6/chbmit/timeseries/" # end all folders with "/"
modelFileName = os.path.join(root, 'Expiriment1_20161122_1800.h5')
#modelFileName = os.path.join(root, 'Expiriment1_20161121_1715.h5')
#modelFileName = os.path.join(root, 'Expiriment1_20161120_1620.h5')
mfiles = [x for x in os.listdir(root) if x.endswith('.mat')]
createModel = True
freq = 0
amp  = 0
for mfile in mfiles:
    filename = os.path.join(root,mfile)
    # load the dataset
    data         = sp.io.loadmat(filename)
    data['amp']  = np.divide(data['amp'], 775)
    data['freq'] = np.divide(data['freq'],128)
    #    freq = max(freq, np.max(data['freq']))
    #    amp  = max(amp, np.max(data['amp']))
    
    dataX, dataY    = create_NN_data(data)    
    dataY = dataY
    del data # manage memory
    
    # split into train and test sets
    size_test  = int(len(dataX) * 0.67)

    np.random.seed(7) # fix random seed for reproducibility
    test  = np.random.choice(len(dataX), size_test)
    train = np.setdiff1d(range(len(dataX)), test) 
    
    x_train, y_train  = [dataX[i] for i in train], [dataY[i] for i in train]
    x_test,  y_test   = [dataX[i] for i in test],  [dataY[i] for i in test]
    
    nb_inputs = len(dataX[0][0])
    # reshape input to be [samples, time steps, features]
    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    #x_test  = np.reshape(x_test,  (x_test.shape[0],  1, x_test.shape[1]))
    del dataX # manage memory
    del dataY # manage memory
    
    print "Completed Deletion of Original Data"
    if createModel:
        # create model
        model = Sequential()
        np.random.seed(7) # fix random seed for reproducibility
        model.add(Dense(1000, input_dim=nb_inputs, activation='sigmoid', bias=True)) # Layer 1
        np.random.seed(7) # fix random seed for reproducibility
        model.add(Dense(50, activation='sigmoid', bias=True))             # Layer 2
        np.random.seed(7) # fix random seed for reproducibility
        model.add(Dense(10, activation='sigmoid', bias=True))             # Layer 3
        np.random.seed(7) # fix random seed for reproducibility
        model.add(Dense(3, activation='sigmoid', bias=True))             # Layer 3
        
        # Compile model
        eta = 0.001
        sgd = SGD(lr=eta, momentum=0.0, decay=0.0, nesterov=False)    
        model.compile(loss=halfsse, optimizer=sgd, metrics=['accuracy'])
        createModel = False
    elif not nb_inputs == max( model.get_input_shape_at(0) ):
        continue
    
    # Train the model for a fixed number of epochs
    ok = True
    for i in range(len(x_train)):
        if y_train[i].any():
            epochs = 200
        else:
            epochs = 1
                
        if ok:
            model.fit(x_train[i], y_train[i], batch_size=1, nb_epoch=epochs, verbose=0, callbacks=[], 
                  validation_split=0.0, validation_data=None, shuffle=False, 
                  class_weight=None, sample_weight=None)

    model.save(modelFileName) # save

del x_train, x_test, y_train, y_test
count = 0
FPR = []
TPR= []
Thresholds = []
NB_Inputs = []
if 0:
    modelFileName = os.path.join(root, 'Expiriment1_20161121_1715.h5')
    model       = load_model(modelFileName)

for mfile in mfiles:
    filename = os.path.join(root,mfile)
    # load the dataset
    data         = sp.io.loadmat(filename)
    data['amp']  = np.divide(data['amp'], 775)
    data['freq'] = np.divide(data['freq'],128)
    
    dataX, dataY    = create_NN_data(data)    
    dataY = dataY
    del data # manage memory
    
    # split into train and test sets
    size_test  = int(len(dataX) * 0.67)

    np.random.seed(7) # fix random seed for reproducibility
    test  = np.random.choice(len(dataX), size_test)
    train = np.setdiff1d(range(len(dataX)), test) 
    
    x_train, y_train  = [dataX[i] for i in train], [dataY[i] for i in train]
    x_test,  y_test   = [dataX[i] for i in test],  [dataY[i] for i in test] 

    nb_inputs = len(dataX[0][0])
    NB_Inputs.append(nb_inputs)
    
    if not nb_inputs == max( model.get_input_shape_at(0) ):
        continue
    # make predictions
    predict = []
    for i in range(len(y_test)):
        predict.append( np.round(model.predict(x_test[i])) )
    
    FPR.append([])
    TPR.append([])
    Thresholds.append([])
       
    for j in range(3):  
        fpr, tpr, thresholds = metrics.roc_curve(np.reshape(y_train, (len(y_train),3))[j,:],
                      np.round(np.reshape(predict, (len(predict), 3))[j,:]))
        FPR[count].append(fpr), TPR[count].append(tpr), Thresholds[count].append(thresholds)
        
    count += 1

exit
for i in range(len(dataY)):
#        print dataY[i].any()
    if dataY[i].any():
        break
    predict = model.predict(dataX[i])
    
for j in range(30):
    predict = model.predict(dataX[i+j])
    print predict
    
#    for layer in model.layers:
#        print(layer.get_weights())
    
#    # Computes the loss on some input data, batch by batch.
#    for i in range(len(x_train)):
#        model.evaluate(x_train[i], y_train[i], batch_size=1, verbose=2, sample_weight=None)

# make predictions
predict_train = model.predict(x_train[i])
predict_test  = model.predict(x_test[i])
    
## calculate root mean squared error
#score_train = math.sqrt(mean_squared_error(y_train, predict_train))
#print('Train Score: %.2f RMSE' % (score_train))
#score_test = math.sqrt(mean_squared_error(testY[0], predict_test[:,0]))
#print('Test Score: %.2f RMSE' % (score_test))
## shift train predictions for plotting
#trainPredictPlot = numpy.empty_like(dataset)
#trainPredictPlot[:, :] = numpy.nan
#trainPredictPlot[look_back:len(predict_train)+look_back, :] = predict_train
## shift test predictions for plotting
#testPredictPlot = numpy.empty_like(dataset)
#testPredictPlot[:, :] = numpy.nan
#testPredictPlot[len(predict_train)+(look_back*2)+1:len(dataset)-1, :] = predict_test
## plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()