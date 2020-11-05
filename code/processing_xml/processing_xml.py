#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:53:37 2020

@author: glevinkonigsberg
"""

# Packages
import pandas as pd
import dropbox
import xml.etree.ElementTree as ET
import zipfile
import os
import time


# Access dropbox

# The token expires every four hours so it needs to be updated before running
# the code
dbx = dropbox.Dropbox('DCjJ5aypffIAAAAAAAAAAYrbI90hi0QSIl_mg-RwJVUHrNeFS3XqAFjSfTdY1T18')


zipFiles = []
for entry in dbx.files_list_folder('/zip').entries:
    zipFiles.append(entry.name)


tags = ["JobID", "JobDate", "JobText", "MSA"]
df = pd.DataFrame(columns = tags)


j = 1
for ff in zipFiles:
    
    metadata, f = dbx.files_download("/zip/" + ff)
    out = open('temp.zip', 'wb')
    out.write(f.content)
    out.close()
    
    try:
        zipfile.ZipFile("temp.zip","r").extractall("temp")
        
        fileName = os.listdir("temp")
        fileName = fileName[0]
        tree = ET.parse("temp/" + fileName)
        root = tree.getroot()
        start = time.time()
        for i in range(len(root)):
            currentObs = root[i]
            values = []
            for t in tags:
                for x in currentObs.iter(tag = t):
                    values.append(x.text)
            newObs = pd.DataFrame(data = [values], columns = tags)
            df = df.append(newObs)
        os.remove("temp/" + fileName)
        end = time.time()
        print("File " + str(j) + " out of " + str(len(zipFiles)) + " done." + "Elapsed time is: " + str(end - start))
    except:
        print("Could not read file:" + zipFiles[j])
    os.remove("temp.zip")
    j = j + 1

compression_opts = dict(method='zip',
                        archive_name='out.csv')  
df.to_csv('output.zip', index=False,
          compression=compression_opts)
        