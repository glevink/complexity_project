#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:49:41 2020

@author: glevinkonigsberg
"""

# Packages
import pandas as pd
from multiprocessing import Pool
import xml.etree.ElementTree as ET
import zipfile
import os
import time
import gc
from contextlib import closing


# Access dropbox


# Generate a list of files which are the input for the procss over the parallelization
# is done

zipPath = "../../bgt_raw_data/Text Data/US/3.4"
processedPath = "/Users/glevinkonigsberg/Dropbox/complexity_project/processed_xml"

zipFiles = []
for yyyy in os.listdir(zipPath):
    if yyyy == '.DS_Store':
        continue
    for entry in os.listdir(zipPath + "/" + yyyy):
        zipFiles.append(yyyy+ "/" + entry)



def processing(ff):
    start = time.time()
    ff = ff[:-4]
    
    print("Started working on file: " + ff + " Time is " + time.asctime() + " EDT.")
    
    tags = (["JobID", "JobDate", "JobText", "MSA", "BGTOcc", "ConsolidatedONET",
             "ConsolidatedTitle", "CleanJobTitle", "CanonEmployer", "CanonCounty"])
    df = pd.DataFrame(columns = tags)
    try:
        
        
        zipfile.ZipFile(zipPath + "/" + ff + ".zip","r").extractall("temp")
        
        tree = ET.parse("temp/" + ff[5:] + ".xml")
        root = tree.getroot()
        del tree
        n = len(root)
        for i in range(n):
            if i % 50000 == 0:
                print("Still working on file: " + ff + " currently on observation " + str(i) + " out of " + str(n) + ". Time is " + time.asctime() + " EDT.")
            currentObs = root[i]
            values = []
            for t in tags:
                for x in currentObs.iter(tag = t):
                    values.append(x.text)
            newObs = pd.DataFrame(data = [values], columns = tags)
            df = df.append(newObs)
        del root
        compression_opts = dict(method='zip', archive_name = "processed_" + ff + ".csv")  
        df.to_csv(processedPath + "/" + "processed_" + ff[5:] + ".zip", index=False,compression=compression_opts)
        del df
        end = time.time()
        print("File " + ff + " has been succesfully processed." + " Elapsed time was: " + str(end - start))
        os.remove("temp/" + ff[5:] + ".xml")
    except:
        print("Could not read file: " + ff)
        try:
            os.remove("temp/" + ff[5:] + ".xml")
        except:
            print("Didn't even get the .xml")
    gc.collect()

with closing(Pool(processes = 6)) as pool:
    pool.map(processing,zipFiles[0:6])
    pool.terminate()

#if __name__ == '__main__':
 #   with Pool(1) as p:
  #      p.map(processing, zipFiles[0])