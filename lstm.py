# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 10:35:11 2021

@author: DeLL
"""
import numpy as np
hidden_size = 100
vocab_size = 26

#parameters
Wxcu = np.random.randn(hidden_size,vocab_size)*0.01
Wscu = np.random.randn(hidden_size,hidden_size)*0.01
Wvcu = np.random.randn(hidden_size,hidden_size)*0.01
bcu  = np.zeros((hidden_size,1))

Wxcs = np.random.randn(hidden_size,vocab_size)*0.01
Wscs = np.random.randn(hidden_size,hidden_size)*0.01
Wvcs = np.random.randn(hidden_size,hidden_size)*0.01
bcs  = np.zeros((hidden_size,1)) 

Wxcr = np.random.randn(hidden_size,vocab_size)*0.01
Wscr = np.random.randn(hidden_size,hidden_size)*0.01
Wvcr = np.random.randn(hidden_size,hidden_size)*0.01
bcr  = np.zeros((hidden_size,1)) 

Wxdu = np.random.randn(hidden_size,vocab_size)*0.01
Wvdu = np.random.randn(hidden_size,hidden_size)*0.01
bdu  = np.zeros((hidden_size,1))


import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def forward(inputs,targets,hprev):
    loss=0
    xs,ss,vs,rs = {},{},{},{}
    ss[-1] = np.copy(hprev)
    vs[-1] = np.copy(np.tanh(hprev))
    acu,gcu,acs,gcs,acr,gcr,adu,u = {},{},{},{},{},{},{},{},{}
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]] = 1

        #update gate
        acu[t] = np.dot(Wxcu,xs[t])+np.dot(Wscu,ss[t-1])+np.dot(Wvcu,vs[t-1])+bcu
        gcu[t] = sigmoid(acu[t])

        #forget gate
        acs[t] = np.dot(Wxcs,xs[t])+np.dot(Wscs,ss[t-1])+np.dot(Wvcs,vs[t-1])+bcs
        gcs[t] = sigmoid(acs[t])

        #output gate
        acr[t] = np.dot(Wxcr,xs[t])+np.dot(Wscr,ss[t])+np.dot(Wvcr,vs[t-1])+bcr
        gcr[t] = sigmoid(acr[t])

        # update candidate
        adu[t] = np.dot(Wxdu,xs[t])+np.dot(Wvdu,vs[t-1])+bdu
        u[t] = np.tanh(adu[t])

        #calculating s[n]:
        ss[t] = np.multiply(gcu[t],u[t])+np.multiply(gcs[t],ss[t-1])

        #readout signals
        rs[t] = np.tanh(ss[t])
        vs[t] = np.multiply(gcr[t],rs[t])
    



    