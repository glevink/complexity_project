#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 07:54:58 2020

@author: glevinkonigsberg


"""

# %% Importing libraries
import torch
import numpy as np
import zipfile
import pandas as pd
import os
import pickle
import transformers as ppb
import time


# %% Define functions

# Define function that receives a list of files with a data on internet postings
# take a sample of at most size n from each file, applies DistilBert to it and
# creates a folder containing the original sample dataframe, the features tensor
# from DistilBert, and an OCC list associated with the Bert tensor.
def posting2BertFiles(fileList,outFolder,n):
    
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
        
    
    occ = df.ConsolidatedONET.values.tolist()
    
    # Save features tensor
    torch.save(features,outFolder + "/bert_features.pt")
    
    # Save occ list
    with open(outFolder + "/occ.txt", "wb") as fp:
        pickle.dump(occ, fp)
    
    # Save Dataframe
    compression_opts = dict(method='zip', archive_name = "data_frame" + ".csv")  
    df.to_csv(outFolder + "/data_frame" + ".zip", index=False,compression=compression_opts)
    
    
