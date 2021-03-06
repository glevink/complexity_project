#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:42:18 2020

@author: glevinkonigsberg
"""

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
import matplotlib.pyplot as plt
import random
import transformers as ppb
import time


# %% Define auxiliary and processing functions 

# Define funciton that receives a list of files with a data on internet postings
# take a sample of at most size n from each file and returns the embedding
# generated by distilBERT.
def posting2Bert(fileList,n):
    
    # Large parts of this function come from
    # https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb#scrollTo=C9t60At16PVs
    
    # Intialize dataFrame
    
    dfOut = pd.DataFrame(columns = ['JobText', 'ConsolidatedONET'])
    
    # Iterate across files in fileList
    
    for ff in fileList:

        if ff[-3:] == "zip":
        
            zipfile.ZipFile(ff,"r").extractall("temp")
            
            filename = ff.split('/')
            
            fileName = filename[-1:][0][:-4]
            
            df = pd.read_csv("temp/"+fileName+".csv",usecols=['ConsolidatedONET','JobText'])
            
            os.remove("temp/"+fileName+".csv")
            
            
        elif ff[-3:] == "csv":
            df = pd.read_csv(ff[:-4]+".csv",usecols=['ConsolidatedONET','JobText'])
            
            os.remove(ff[:-4]+".csv")
            
        else:
            print(ff + "is not a Zip file nor a csv file")
        
        # Take a sample of size n
        df = df.sample(n)
        
        # Transform ConsolidatedONET into six digit categoricacategorical
        # If the OCC is zero, drop the observation
        df.ConsolidatedONET = np.floor(df.ConsolidatedONET/100)
        df.ConsolidatedONET = df.ConsolidatedONET.fillna(0)
        df = df[df.ConsolidatedONET != 0]
        df['ConsolidatedONET'] = df['ConsolidatedONET'].astype('int').astype('category')
        
        # Tranform JobText to string
        df['JobText'] = df['JobText'].astype('string')
        
        # Eliminate nan that confuses python later
        df.JobText = df.JobText.fillna("")
        
        dfOut = dfOut.append(df)
    
    
    # Load pretrained BERT model
        
    t = time.time()
    
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    # Add tokens
    tokenized = dfOut['JobText'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    
    # Let's add padding
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
        

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    
    if padded.shape[1] > 512:
        print("Warning: some elements where larger than 512 and I had to cut them")
        padded = padded[:,0:512]

    
    attention_mask = np.where(padded != 0, 1, 0)
    
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)
    
    print("Applying Bert now")
    
    # Applying Bert to all observations at once takes lots of RAM, let's intead
    # apply it in groups of 100
    
    for j in range(int(np.ceil(input_ids.shape[0]/100))):
        
        if j%10 == 0:
            print("Currently working on observations " + str(100*j) +" to " + str((j+1)*100) +".")
            print("Time is: " + time.asctime() + ".")
        
        with torch.no_grad():
            last_hidden_states = model(input_ids[100*j:(j+1)*100], \
                                       attention_mask=attention_mask[100*j:(j+1)*100])
        
        if j == 0:
            features = last_hidden_states[0][:,0,:]
        else:
            features = torch.vstack((features,last_hidden_states[0][:,0,:]))
    
    elapsed = time.time() - t
    
    print('Applying Bert took: ' + str(elapsed) + 'seconds.')
        
    # Also output OCC
    occ = df.ConsolidatedONET.values.tolist()
    
    return features, occ
    

# Define a function that given a csv file and a metric, returns a dictionary
# in which the keys are the OCC codes and each key has a single entry with
# the value of the complexity metric
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

# Define a function that takes a tags dictionary and a list of OCCs ans returns
# a tensor with the corresponging tags

def getTags(tagDictionary,occList):
    
    tagList = [tagDictionary.get(occ) for occ in occList]
    
    tagList = [0 if tag is None else tag for tag in tagList]
    
    tagTensor = torch.FloatTensor(tagList)
    
    
    return tagTensor

# Define trainning function whic receive tensor X, tags Y, neeural network
# model nn, loss function criterions, optimizing algorithm optimizer,
# number of epochs iterations, size of sample for SGD n, validation tensor
# valX, validation labels n, and boolean variable plot.
def train(X,Y,nnModel,criterion,optimizer,iterations,n,valX,valY,plot):
    
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
        valModel = nnModel
        valCriterion = criterion
        valOutput = valModel(valX)
        valLoss = valCriterion(valOutput,valY)
        valError.append(valLoss.item())
        
        elapsed = time.time() - t
        
        print("Just finished epoch: " + str(i) + ". Elapsed time is: " + str(elapsed))
        print("Time is: " + time.asctime())
        
        
    
    # If requested plot error over iterations
    if plot:
        t = np.linspace(1,iterations,iterations)
        plt.plot(t,error,'b')
        plt.plot(t,valError,'r')
        plt.legend(["In sample error","Validation error"])
        plt.show()
    
    return error, valError

#%% Data reading and processing, from files to tensor

# First create the dictionary of text

# Create list of files to iterate on

fileList = os.listdir('../../processed_xml')[:11]
fileList.remove('.DS_Store')
fileList = ['../../processed_xml/' + ff for ff in fileList]

# Get a sample of postings (transformed into BERT features) and their corres
# ponding occ codes. Also get a validation sample
X, occList = posting2Bert(fileList,5000)
xVal, occListVal = posting2Bert(fileList,1000)

# Now create the tag dictionary for our specific metric
metric = 'leader'
tagFile = "../../label_creation/output/occ_labels.csv"

tagDictionary = createOCCTagDict(tagFile,metric)

# Get the tag that corresponds to our sampled OCC codes
Y = getTags(tagDictionary,occList)
yVal = getTags(tagDictionary,occListVal)



#%% Neural network definitions and trainning

# Set size for the model
d = X.shape[1]

# Define our nn model, loss function module and optimizer
# Define a fully connected neural network with one tanh layer
nnModel = nn.Sequential(nn.Linear(in_features = d, out_features = d), \
                        nn.ReLU(),nn.Linear(in_features = d, out_features = 1))
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(nnModel.parameters(), lr=1e-3)

# Train the model
error,valError = train(X,Y,nnModel,criterion,optimizer,100,1000,xVal,yVal,True)
