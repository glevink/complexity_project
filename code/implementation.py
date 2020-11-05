#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:19:31 2020

@author: glevinkonigsberg
"""

# %% Importing libraries
import torch
import torch.nn as nn
import numpy as np
import zipfile
import pandas as pd
import os
import str
import matplotlib.pyplot as plt
import random
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import itertools


# %% Define auxiliary and processing functions 

#Define function to read posting DFs from files, include unziping and
# rezipping. Then creates a dictionary that maps an OCC code into a list
# of sentences coming from all the postings in the file. Input is file
def createOCCDesDict(ff):
    # Unzip file to temp folder
    
    if ff[-3:] == "zip":
    
        zipfile.ZipFile(ff,"r").extractall("temp")
    
        df = pd.read_csv(ff[:-4]+".csv",usecols=['ConsolidatedONET','JobText'])
        
    elif ff[-3:] == "csv":
        df = pd.read_csv(ff[:-4]+".csv",usecols=['ConsolidatedONET','JobText'])
        
    else:
        print("This is not a Zip file nor a csv file")
    
    # Transform ConsolidatedONET into six digit categoricacategorical
    df.ConsolidatedONET = np.floor(df.ConsolidatedONET/100)
    df.ConsolidatedONET = df.ConsolidatedONET.fillna(0)
    df['ConsolidatedONET'] = df['ConsolidatedONET'].astype('int').astype('category')
    
    # Get category list
    catList = df.ConsolidatedONET.cat.categories
    
    # Tranform JobText to string
    df['JobText'] = df['JobText'].astype('string')
    
    # Eliminate nan that confuses python later
    df.JobText = df.JobText.fillna("")
    
    
    dictionary = {}
    for occ in catList:
        aux = df[df.ConsolidatedONET.eq(occ)].JobText
        aux = ".".join(aux)
        aux = aux.replace("..",".")
        aux = aux.split(".")
        auxDict = {occ: aux}
        dictionary.update(auxDict)
        
    if ff[-3:] == "zip":
        os.remove('temp/' + ff[:-4] + '.csv')
        
    return dictionary
    

# Define a function that given a csv file and a metric, returns a dictionary
# in which the keys are the OCC codes and each key has a single entry with
# the value of the complexity metric
def createOCCTagDict(ff,metric):
    ff = "../label_creation/output/" + ff
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

# Define a function that given a word2vec model and a phrase returns and array
# in which each row is the vector representation of each token in the phrase
def phrase2Vec(model,phrase):
    
    tokens = word_tokenize(phrase.lower())
    
    # Eliminate tokens not in the model
    isInVocab = [(tt in model.vocab) for tt in tokens]
    
    tokens = list(itertools.compress(tokens, isInVocab))
    
    if len(tokens) > 0:
        vectors = model[tokens]
    else:
        aux = model['hello'].shape
        vectors = np.zeros(aux)
    
    vectors = torch.from_numpy(vectors)
    
    return vectors

# Define a function that given a dictionary of text and one of tags,
# randomly extracts from each key
# a subsamples in a way consistent with the overall distribution. N is the
# total sample size. Later we''l want to add tags to this process
def sampleDict(dictionaryText,dictionaryTags,n):
    topics = dictionaryText.keys()
    dist = np.array([len(dictionaryText.get(tt)) for tt in topics])
    dist = dist/sum(dist)
    sampleSize = np.round(dist*n).astype('int')
    X = []
    Y = []
    for i, topic in enumerate(topics):
        if topic not in dictionaryTags.keys():
            continue
        newList = random.sample(dictionaryText.get(topic),sampleSize[i])
        if newList == []:
            continue
        X.append(newList)
        newTags = [dictionaryTags.get(topic) for i in range(len(newList))]
        Y.append(newTags)
        # Here we also need Y.append and the right labels
    X = list(itertools.chain.from_iterable(X))
    Y = list(itertools.chain.from_iterable(Y))
    return X, Y
    
# Define function that takes a sample of text and an embedding model and gives
# back the packeded tensor that is the input of the RNN
def sample2Tensor(X,model):
    tensorList = [phrase2Vec(model,pp) for pp in X]
    
    return tensorList


#%% Data reading and processing, from files to tensor

# First create the dictionary of text

# Create list of files to iterate on

fileList = ["../processed_xml/US_XML_AddFeed_20071119_20071125.csv"]

textDictionary = {}
for ff in fileList:
    newEntries = createOCCDesDict(ff)
    textDictionary.update(newEntries)
    del newEntries
    
# Now create the tag dictionary for our specific metric
metric = 'leader'
tagFile = 'occ_labels.csv'

tagDictionary = createOCCTagDict(tagFile,metric)

# Now generate a sample
inputX,inputY = sampleDict(textDictionary,tagDictionary, 10000)

# Generate a validation sample
valX,valY = sampleDict(textDictionary,tagDictionary, 10000)


# Load pre-trained embedding matrix     
model = api.load('glove-wiki-gigaword-50')

# Transform X into a list of tensors
inputX = sample2Tensor(inputX,model)
valX = sample2Tensor(valX,model)



#%% Neural network definitions

# Create a class that takes the packed tensor, applies a RNN, then takes
# the last ouput, applies a linear unit a prediction. The RNN layer has
# by default a tanh activation and other things we might want to check.
class rnnFC(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(rnnFC, self).__init__()

        self.Phrase = nn.RNN(input_size,hidden_size)
        self.Pred = nn.Linear(hidden_size, 1)

    def forward(self, input):
        aa,bb = self.Phrase(input)
        output = self.Pred(bb)
        return torch.squeeze(output)

# Define trainning function
def train(X,Y,nnModel,criterion,optimizer,iterations,valX,valY,plot):
    
    error = []
    valError = []
    
    
    # Set dependent variable as tensor
    target = torch.FloatTensor(Y)
    valTarget = torch.FloatTensor(valY)
    
    # Pack tensor list        
    packedTensor = torch.nn.utils.rnn.pack_sequence(X, enforce_sorted=False)
    valPackedTensor = torch.nn.utils.rnn.pack_sequence(valX, enforce_sorted=False)
    
    for i in range(iterations):
                
        # Forward pass
        yPred = nnModel(packedTensor)
        
        # Compute loss
        loss = criterion(yPred,target)
        
        # Track validation error
        
        valModel = nnModel
        valCriterion = criterion
        valOutput = valModel(valPackedTensor)
        valLoss = valCriterion(valOutput,valTarget)
        valError.append(valLoss.item())        
        
        # Restart gradient
        optimizer.zero_grad()
        
        # Back propagation
        loss.backward()
        
        # Update wights
        optimizer.step()
            
        # Store value of loss for plot
        error.append(loss.item())
        
        
    
    # If requested plot error over iterations
    if plot:
        t = np.linspace(1,iterations,iterations)
        plt.plot(t,error,'b')
        plt.plot(t,valError,'r')
        plt.legend(["In sample error","Validation error"])
        plt.show()
    
    return error

# Set size for the model
d = model['hello'].shape[0]

# Define our nn model, loss function module and optimizer
nnModel = rnnFC(d,d)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(nnModel.parameters(), lr=1e-3)

# Train the model
train(inputX,inputY,nnModel,criterion,optimizer,200,valX,valY,True)
