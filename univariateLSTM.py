# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:09:07 2019

@author: Vishesh Biyani
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:50:28 2019

@author: Vishesh Biyani
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics

def splitdata(dataset):
    train, test = dataset[:75], dataset[75:]
    return train, test

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
       # self.h_0 = torch.zeros(1,1,8).type(torch.DoubleTensor)
       # self.c_0 = torch.zeros(1,1,8).type(torch.DoubleTensor)
        self.lstm = nn.LSTM(1,16,2)
        self.linear2 = nn.Linear(96,24)
        self.linear3 = nn.Linear(24,1)
    
    def forward(self,inp):
        out2, (hn, cn) = self.lstm(inp,(self.h_0,self.c_0))
       # self.h_0 = hn
       # self.c_0 = cn
     #   print("out2 of lstm: ",out2)
        out2 = out2.transpose(0,1)
     #   print("transposed output: ",out2)
       # print("input to linear: ",out2.view(1,-1))
        out3 = self.linear2(F.elu(out2.view(1,-1)))
        out = self.linear3(F.leaky_relu(out3))
     #   print("final out:",out)
        return out
    
    def initHidden(self):
        self.h_0 = torch.zeros(2,1,16).type(torch.DoubleTensor)
        self.c_0 = torch.zeros(2,1,16).type(torch.DoubleTensor)
    
   

def trainprep(train):
    std = statistics.stdev(train.values[:,0])
    men = statistics.mean(train.values[:,0])
    history = []
    actual = []
    
    for i in range(-6,69):
        col = []
        for p in range(6):
            col.append((train.values[i+p][0]-men)/std if(train.values[i+p][0]) else 0)
  #      col = [(train.values[i][0]-men)/std if(train.values[i][0]) else 0,(train.values[i+1][0]-men)/std if(train.values[i+1][0]) else 0,(train.values[i+2][0]-men)/std if(train.values[i+2][0]) else 0, (train.values[i+3][0]-men)/std if(train.values[i+3][0]) else 0, (train.values[i+4][0]-men)/std if(train.values[i+4][0]) else 0, (train.values[i+5][0]-men)/std if(train.values[i+5][0]) else 0]
  #      col = [train.values[i][1],train.values[i+1][1],train.values[i+2][1]]
        actual.append((train.values[i+6][0]-men)/std)
        history.append(col)
    for j in range(75):
        history[j] = torch.from_numpy(np.asarray(history[j]))
        history[j] = history[j].type(torch.DoubleTensor)
        history[j] = history[j].view((6,1,1))
    for j in range(75):
        actual[j] = torch.from_numpy(np.asarray(actual[j]))
        actual[j] = actual[j].type(torch.DoubleTensor)
        actual[j] = actual[j].view((1,1))
    return history, actual

def testprep(test):
    std = statistics.stdev(train.values[:,0])
    men = statistics.mean(train.values[:,0])
    test_history = []
    test_actual = []
    for i in range(-6,27):
        colnew=[]
        for p in range(6):
            colnew.append((test.values[i+p][0]-men)/std if(test.values[i+p][0]) else 0)
        test_actual.append((test.values[i+6][0]-men)/std)
        test_history.append(colnew)
    for j in range(33):
        test_history[j] = torch.from_numpy(np.asarray(test_history[j]))
        test_history[j] = test_history[j].type(torch.DoubleTensor)
        test_history[j] = test_history[j].view((6,1,1))
    for j in range(33):
        test_actual[j] = torch.from_numpy(np.asarray(test_actual[j]))
        test_actual[j] = test_actual[j].type(torch.DoubleTensor)
        test_actual[j] = test_actual[j].view((1,1))
    return test_history, test_actual

def model_train(m,hist,act):
    criterion = nn.MSELoss()
    predictions = []
    std = statistics.stdev(train.values[:,0])
    men = statistics.mean(train.values[:,0])
    m_optimizer = optim.Adagrad(m.parameters(),lr=0.01)
    for epoch in range(200):
        epochpredictions = []
        epochloss=0.0
        for i in range(75):
            loss = 0
            m.initHidden()
            m_optimizer.zero_grad()
            yhat = m(hist[i])
          #  print("yhat: ",yhat.item())
            epochpredictions.append(yhat.item()*std + men)
            loss = criterion(yhat,act[i])
            epochloss+=loss.item()
            loss.backward()
            m_optimizer.step()
        predictions.append(epochpredictions)
        print('epoch %d loss: %.5f' % (epoch+1,epochloss/75))
    return m, predictions

def model_test(tm, test_hist, test_act):
    criterion = nn.MSELoss()
    test_predictions = []
    std = statistics.stdev(train.values[:,0])
    men = statistics.mean(train.values[:,0]) 
    test_loss = 0.0
    for i in range(33):
        loss = 0
        yhat_test = tm(test_hist[i])
        test_predictions.append(yhat_test.item()*std + men)
        loss = criterion(yhat_test,test_act[i])
        test_loss+=loss.item()
    print("Testing loss is: ",test_loss/33)   
    return test_predictions

def testimshow(gt,pred):
    pyplot.figure(1)
    pyplot.plot(gt)
    pyplot.plot(pred)
    #pyplot.plot(pred)
    pyplot.show()

def trainimshow(gt,pred):
    pyplot.figure(1)
    pyplot.plot(gt)
   # for i in range(100):
    pyplot.plot(pred[199])
    pyplot.show()

dataset = pd.read_csv('C:/Users/Vishesh Biyani/Desktop/UNIVARIATE/monthly-car-sales.csv',header=0,index_col=0)
train, test = splitdata(dataset)
m = model().double()
hist, act = trainprep(train)
test_hist, test_act = testprep(test)
trained_model , train_predictions = model_train(m,hist,act)
path = 'C:/Users/Vishesh Biyani/Desktop/UNIVARIATE/UNImodelLSTM.pt'
torch.save(trained_model.state_dict(), path)
tst_predictions = model_test(trained_model,test_hist,test_act)
trainimshow(train,train_predictions)
testimshow(test,tst_predictions)
