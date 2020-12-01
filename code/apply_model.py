#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:27:33 2020

@author: glevinkonigsberg
"""

# %% Importing libraries
import torch
import numpy as np
import zipfile
import pandas as pd
import os
#%% Define functions

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
"""

def applyModel(dirList,modelFile,skill):
    
    first = True
    # Iterate across directories
    for dd in dirList:
        
        zipfile.ZipFile(dd+"/data_frame.zip","r").extractall("temp")
        df = pd.read_csv("temp/data_frame.csv")
        os.remove("temp/data_frame.csv")
        bertFeatures = torch.load(dd + "/bert_features.pt")
        
        if first:
            outDF = df
            outFeatures = bertFeatures
        else:
            first = False
            outDF = outDF.append(df)
            outFeatures = torch.vstack((outFeatures,bertFeatures))
            
    # Load pytorch model
    nnModel = torch.load(modelFile)
    nnModel.eval()
    
    predVal = nnModel(bertFeatures).detach().numpy()
    
    df[skill + "Complexity"] = predVal
    
    # Let's get the OCC-based complexity measure
    
    tagDict = createOCCTagDict('occ_labels.csv',skill)
    
    occComplexity = getTags(tagDict,df.ConsolidatedONET.tolist()).detach().numpy()
    
    df[skill + 'ComplexityOCC'] = occComplexity
    
    return df
