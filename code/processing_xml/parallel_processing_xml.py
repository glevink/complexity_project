#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:49:41 2020

@author: glevinkonigsberg
"""

# Packages
import pandas as pd
from multiprocessing import Pool
import dropbox
import xml.etree.ElementTree as ET
import zipfile
import os
import time
import gc
from contextlib import closing


# Access dropbox

# The token expires every four hours so it needs to be updated before running
# the code
dbx = dropbox.Dropbox('kKIt5A2UEP4AAAAAAAAAAT834PnNfIafOfwKJNq19ysL30VF0ezaEQIhmdexWFgM')



# Generate a list of files which are the input for the procss over the parallelization
# is done
zipFiles = []
for entry in dbx.files_list_folder('/zip').entries:
    zipFiles.append(entry.name)



def processing(ff):
    start = time.time()
    ff = ff[:-4]
    
    print("Started working on file: " + ff + " Time is " + time.asctime() + " EDT.")
    
    metadata, f = dbx.files_download("/zip/" + ff + ".zip")
    out = open(ff + ".zip", 'wb')
    out.write(f.content)
    out.close()
    
    tags = ["JobID", "JobDate", "JobText", "MSA"]
    df = pd.DataFrame(columns = tags)
    try:
        
        zipfile.ZipFile(ff + ".zip","r").extractall("temp")
        
        tree = ET.parse("temp/" + ff + ".xml")
        root = tree.getroot()
        del tree
        for i in range(len(root)):
            if i % 10000 == 0:
                print("Still working on file: " + ff + " currentyl on observation " + i + ". Time is " + time.asctime() + " EDT.")
            currentObs = root[i]
            values = []
            for t in tags:
                for x in currentObs.iter(tag = t):
                    values.append(x.text)
            newObs = pd.DataFrame(data = [values], columns = tags)
            df = df.append(newObs)
        del root
        compression_opts = dict(method='zip', archive_name = "processed_" + ff + ".csv")  
        df.to_csv("processed_" + ff + ".zip", index=False,compression=compression_opts)
        del df
        end = time.time()
        print("File " + ff + " has been succesfully processed." + " Elapsed time was: " + str(end - start))
        os.remove("temp/" + ff + ".xml")
    except:
        print("Could not read file: " + ff)
    os.remove(ff + ".zip")
    gc.collect()

with closing(Pool(processes = 6)) as pool:
    pool.map(processing,zipFiles[0:24])
    pool.terminate()

#if __name__ == '__main__':
    #with Pool(6) as p:
        #p.map(processing, zipFiles[0:6])