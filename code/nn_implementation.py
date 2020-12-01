#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:10:56 2020

@author: glevinkonigsberg
"""

#%% Import libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
import random

#%% Create funcitons to be used

"""
Define a function that given a csv file and a metric, returns a dictionary
in which the keys are the OCC codes and each key has a single entry with
the value of the complexity metric
"""
def createOCCTagDict(ff,metric):
    df = pd.read_csv(ff[:-4]+".csv",usecols=['soc2010',metric])
    df = df.rename(columns={metric: "metricVar"})
    
    # Verify that there are no nan otherwise replace them
    if np.sum(np.isnan(df.metricVar)) > 0:
        print("Warning: there are entries with no value, replacing them with zero")
        df = df.fillna(0)
    
    # Get category list
    df.soc2010 = df.soc2010.astype('category')
    catList = df.soc2010.cat.categories
    
    # Creat dictionary
    dictionaryTags = {}
    for occ in catList:
        aux = float(df[df.soc2010.eq(occ)].metricVar)
        auxDict = {occ: aux}
        dictionaryTags.update(auxDict)
        
    return dictionaryTags


"""
# Define a function that takes a tags dictionary and a list of OCCs ans returns
# a tensor with the corresponging tags
"""
def getTags(tagDictionary,occList):
    
    tagList = [tagDictionary.get(occ) for occ in occList]
    
    tagList = [0 if tag is None else tag for tag in tagList]
    
    tagTensor = torch.FloatTensor(tagList)
    
    tagTensor = torch.reshape(tagTensor,(len(tagList),1))
    
    return tagTensor

"""
The folowing funcitons takes a list of directories that include three file:
    (i) bert_features.pt, (ii) occ.txt, (iii) .zip. It receives skill name
to look in the dictionary of occ.
It also receive a value p. It opens each the first two files and and jointly
samples a a propotion p of observations in each file. It assigns p to a vali-
dation sample and 1-p to a trainning sample.
It outputs four pytorch tensors: the bert features for the trainning sample,
the bert features for the validation sample, the target values for the training
sample and the targe values for the validation sample.
"""

def createSample(fileList,skill,p):
    
    first = True
    
    # Iterate across files
    for ff in fileList:
        
        # Load data
        loadedTensor = torch.load(ff + "/bert_features.pt")
        occFile = open(ff + "/occ.txt", 'rb')
        occList = pickle.load(occFile)        
        occFile.close()
        
        # Create indices for each type of sample
        size = loadedTensor.shape[0]
        indices = [i for i in range(size)]
        trainingIndices = random.sample(indices, int(np.ceil((1-p)*size)))
        valIndices = np.setdiff1d(indices,trainingIndices)
        
        trFeatures = loadedTensor[trainingIndices]
        valFeatures = loadedTensor[valIndices]
        trOCC = [occList[i] for i in trainingIndices]
        valOCC = [occList[i] for i in valIndices]
        
        if first:
            outTrFeatures = trFeatures
            outValFeatures = valFeatures
            outTrOCC = trOCC
            outValOCC = valOCC
            first = False
        else:
            outTrFeatures = torch.vstack((outTrFeatures,trFeatures))
            outValFeatures = torch.vstack((outValFeatures,valFeatures))
            outTrOCC.extend(trOCC)
            outValOCC.extend(valOCC)
        
    # Now get the dictionary for the relevant skill and get the values assigned
    # at each of our OCCs for that skill
    occDict = createOCCTagDict('occ_labels.csv',skill)
    trTarget = getTags(occDict,outTrOCC)
    valTarget = getTags(occDict,outValOCC)
    
    return outTrFeatures, trTarget, outValFeatures, valTarget

# Define trainning function which receive tensor X, tags Y, neural network
# model nn, loss function criterions, optimizing algorithm optimizer,
# number of epochs iterations, size of sample for SGD n, validation tensor
# valX, validation labels n, and boolean variable plot.
def train(X,Y,valX,valY,nnModel,criterion,optimizer,iterations,n,saveFile,plot):
    
    t = time.time()
    
    print("Just started working, time is: " + time.asctime())
    
    numObs = len(Y)
    groups = int(np.ceil(numObs/n))
    
    error = []
    valError = []
    
    for i in range(iterations):
        
        # Reshuffle data
        indices = list(range(len(Y)))
        random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        
        for j in range(groups):
            
            thisX = X[j*n:(j+1)*n]
            thisY = Y[j*n:(j+1)*n]
            
            # Forward pass
            yPred = nnModel(thisX)
            
            # Compute loss
            loss = criterion(yPred,thisY)
            
            # Restart gradient
            optimizer.zero_grad()
            
            # Back propagation
            loss.backward()
            
            # Update wights
            optimizer.step()
            
        # Store value of loss for plot
        error.append(loss.item())
        
        # Track validation error
        # valModel = nnModel
        # valCriterion = criterion
        # valOutput = valModel(valX)
        # valLoss = valCriterion(valOutput,valY)
        # valError.append(valLoss.item())
        with torch.no_grad():
            valOutput = nnModel(valX)
            valLoss = criterion(valOutput,valY)
            valError.append(valLoss.item())
        
        elapsed = time.time() - t
        
        if i%10 == 0:
            print("Just finished epoch: " + str(i) + ". Elapsed time is: " + str(elapsed))
            print("Time is: " + time.asctime())
        
        if i%100 == 0 and i != 0:
            torch.save(nnModel,saveFile)
    
    torch.save(nnModel,saveFile)
    
    # If requested plot error over iterations
    if plot:
        t = np.linspace(1,iterations,iterations)
        plt.plot(t,error,'b')
        plt.plot(t,valError,'r')
        plt.legend(["In sample error","Validation error"])
        plt.savefig('errorGraph.png')
        #plt.show()
    
    return error, valError


#%% Define the neural network and training elements

# Notice that BERT has 768 features
nnModel = nn.Sequential(nn.Linear(in_features = 768, out_features = 768), \
                        nn.ReLU(),nn.Linear(in_features = 768, out_features = 1))
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(nnModel.parameters(), lr=1e-3)

