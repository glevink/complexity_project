# -*- coding: utf-8 -*-
"""
Creating Word Clouds for highest scoring job posts
"""

##### PREAMBLE ######


import pandas as pd
import string
from collections import defaultdict
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import numpy as np
from pprint import pprint

#Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from wordcloud import WordCloud

from yellowbrick.text.freqdist import FreqDistVisualizer



##### Load data #####


### Step 1. read in data
#read in text for job postings
df_text= pd.read_csv('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/applied_models/applied_model_socskills_batch_1-3.csv')

#Step 2. simple clean- remove punctuations & count number of words in each review
#first lets get rid of these punctuations
punctuations= ['#', '?', ':', '*', '!', '-', '.', ',', ';', "\\", "@", "&","|",")", "("]
#create a regular expression to get rid of the punctuations 
rx= '[' + re.escape(''.join(punctuations)) + ']'

#Convert the titles to lowercase
df_text['jobtext_processed']= df_text['JobText'].map(lambda x: str(x).lower())
#Remove punctuation
df_text['jobtext_processed']= df_text['jobtext_processed'].map(lambda x: re.sub(rx, '', str(x)))

# Remove "job description" and "united states" [[[ADD MORE]]]
#df_text['jobtext_processed']= df_text['jobtext_processed'].map(lambda x: x.replace('united states', ' '))
#df_text['jobtext_processed']= df_text['jobtext_processed'].map(lambda x: x.replace('job description', ' '))



# Drop "dummy" postings
df_text = df_text[df_text['jobtext_processed'] != 'jtext dummybgt' ]

#Print out the first rows of simple-cleaned reviews
df_text['jobtext_processed'].head()

#Count number of words in jobtext_processed using space as separator
df_text['num_words']= df_text['jobtext_processed'].map(lambda x: len(x.split()))


#Step 3: word tokenization with nltk's word_tokenize function
#first need to remove stopwords
stop_words= set(stopwords.words('english')) 

#Tokenize
def preprocess(text):
    '''tokenize and remove stopwords keep only tokens with > 2 characters'''
    result= []
    for token in word_tokenize(text):
        if token not in stop_words and len(token) > 2:
            result.append(token)
    return result

#Run the preprocess function above on review_text_processed1
df_text['jobtext_tokens']= df_text['jobtext_processed'].map(lambda x: preprocess(str(x)))
#Print out the first rows of tokenized reviews
df_text['jobtext_tokens'].head()



#Count number of tokens
df_text['num_tokens']= df_text['jobtext_tokens'].map(lambda x: len(x))

#write df to csv
jobtext_outf= open('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/word_clouds_jobtext_socskills.csv', 'w+', encoding= 'utf-8')
df_text.to_csv(jobtext_outf)



#Step 4: visualize the number of words & number of tokens
#freq distribution of word counts in docs
#how big the docs (job texts) are as a whole 
jobtext_lens_words= df_text['num_words']
jobtext_lens_tokens= df_text['num_tokens']

#plot 1- number of words 
plt.figure(figsize= (10,7), dpi= 160)
plt.hist(jobtext_lens_words, bins= 100, color= 'purple')
plt.text(600, 1250, "Mean   : " + str(round(np.mean(jobtext_lens_words))))
plt.text(600, 1150, "Median : " + str(round(np.median(jobtext_lens_words))))
plt.text(600, 1050, "Stdev   : " + str(round(np.std(jobtext_lens_words))))
plt.text(600, 950, "1%ile    : " + str(round(np.percentile(jobtext_lens_words, 1))))
plt.text(600, 850, "99%ile  : " + str(round(np.percentile(jobtext_lens_words, 99))))
plt.text(600, 750, "Max : " + str(round(np.max(jobtext_lens_words))))
plt.gca().set(xlim= (0, 800), ylabel= 'Number of Job Postings', xlabel= 'Job Posting Word Count')
plt.tick_params(size= 16)
plt.xticks(np.linspace(0, 800, 11))
#plt.title('Distribution of Job Posting Word Counts', fontdict= dict(size= 16))    
plt.savefig('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/distr_jobtext_words_socskills.png',  bbox_inches = "tight")    
plt.show()  

#plot 2- number of tokens
plt.figure(figsize= (10,7), dpi= 160)
plt.hist(jobtext_lens_tokens, bins= 100, color= 'purple')
plt.text(600, 1250, "Mean   : " + str(round(np.mean(jobtext_lens_tokens))))
plt.text(600, 1150, "Median : " + str(round(np.median(jobtext_lens_tokens))))
plt.text(600, 1050, "Stdev   : " + str(round(np.std(jobtext_lens_tokens))))
plt.text(600, 950, "1%ile    : " + str(round(np.percentile(jobtext_lens_tokens, 1))))
plt.text(600, 850, "99%ile  : " + str(round(np.percentile(jobtext_lens_tokens, 99))))
plt.text(600, 750, "Max : " + str(round(np.max(jobtext_lens_tokens))))
plt.gca().set(xlim= (0, 800), ylabel= 'Number of Job Postings', xlabel= 'Job Posting Token Count')
plt.tick_params(size= 16)
plt.xticks(np.linspace(0, 800, 11))
#plt.title('Distribution of Job Posting Token Counts', fontdict= dict(size= 16))  
plt.savefig('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/distr_jobtext_tokens_socskills.png', bbox_inches = "tight")    
plt.show()  

##### Wordcloud ######
lemmatizer = WordNetLemmatizer()



### Predicted top 10 %
criterion = 'socskillsComplexity'
cutoff = np.percentile(df_text[criterion], 90)

top10predict_smalldf = df_text[df_text[criterion] >= cutoff ]
top10predict_padded= [' '.join(w) for w in top10predict_smalldf['jobtext_tokens']]
top10predict_temp_string= ''
for t in top10predict_padded:
    top10predict_temp_string += "\n" + t

# LEMMATIZE
top10predict_lem_words = [lemmatizer.lemmatize(x) for x in top10predict_temp_string.split()]

top10predict_long_string =  '' 
for t in top10predict_lem_words:
    top10predict_long_string += "\n" + t

#Create a WordCloud object
wcloud= WordCloud(background_color="white", max_words= 5000, contour_width= 3, contour_color='steelblue')
#Generate a word cloud
wcloud.generate(top10predict_long_string)
#Visualize the word cloud
wcloud.to_image()
wcloud.to_file('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/wcloud_top10predict_socskills.png') 
     
    
### Actual Top 10%
criterion = 'socskillsComplexityOCC'
cutoff = np.percentile(df_text[criterion], 90)

top10act_smalldf = df_text[df_text[criterion] >= cutoff ]
top10act_padded= [' '.join(w) for w in top10act_smalldf['jobtext_tokens']]
top10act_temp_string= ''
for t in top10act_padded:
    top10act_temp_string += "\n" + t

# LEMMATIZE
top10act_lem_words = [lemmatizer.lemmatize(x) for x in top10act_temp_string.split()]

top10act_long_string =  '' 
for t in top10act_lem_words:
    top10act_long_string += "\n" + t  
    
#Create a WordCloud object
wcloud= WordCloud(background_color="white", max_words= 5000, contour_width= 3, contour_color='steelblue')
#Generate a word cloud
wcloud.generate(top10act_long_string)
#Visualize the word cloud
wcloud.to_image()
wcloud.to_file('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/wcloud_top10act_socskills.png') 
 

### Predicted Bottom 10 %
criterion = 'socskillsComplexity'
cutoff = np.percentile(df_text[criterion], 10)

low10predict_smalldf = df_text[df_text[criterion] < cutoff ]
low10predict_padded= [' '.join(w) for w in low10predict_smalldf['jobtext_tokens']]
low10predict_temp_string= ''
for t in low10predict_padded:
    low10predict_temp_string += "\n" + t

# LEMMATIZE
low10predict_lem_words = [lemmatizer.lemmatize(x) for x in low10predict_temp_string.split()]

low10predict_long_string =  '' 
for t in low10predict_lem_words:
    low10predict_long_string += "\n" + t
    
#Create a WordCloud object
wcloud= WordCloud(background_color="white", max_words= 5000, contour_width= 3, contour_color='steelblue')
#Generate a word cloud
wcloud.generate(low10predict_long_string)
#Visualize the word cloud
wcloud.to_image()
wcloud.to_file('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/wcloud_low10predict_socskills.png') 
     



### Actual Bottom 10 %
criterion = 'socskillsComplexityOCC'
cutoff = np.percentile(df_text[criterion], 10)

low10act_smalldf = df_text[df_text[criterion] < cutoff ]
low10act_padded= [' '.join(w) for w in low10act_smalldf['jobtext_tokens']]
low10act_temp_string= ''
for t in low10act_padded:
    low10act_temp_string += "\n" + t

# LEMMATIZE
low10act_lem_words = [lemmatizer.lemmatize(x) for x in low10act_temp_string.split()]

low10act_long_string =  '' 
for t in low10act_lem_words:
    low10act_long_string += "\n" + t  
    
#Create a WordCloud object
wcloud= WordCloud(background_color="white", max_words= 5000, contour_width= 3, contour_color='steelblue')
#Generate a word cloud
wcloud.generate(low10act_long_string)
#Visualize the word cloud
wcloud.to_image()
wcloud.to_file('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/wcloud_low10act_socskills.png') 
 


###  DISTINCT WORDS (difference top to bottom decile) ####

 
    ### Train VOcabulary on full data set ###
         
#Initialise the count vectorizer with the English stop words
count_vectorizer= CountVectorizer(stop_words='english')
#Fit and transform the processed job postings
#Fit= Tokenize the collection of reviews and form a vocabulary with it
#Transform= Encode all the job postings based on the vocabulary
#By default CountVectorizer removes punctuation and lowers text 


# Lemmatize and combine full data
all_padded= [' '.join(w) for w in df_text['jobtext_tokens']]
all_temp_string= ''
for t in all_padded:
    all_temp_string += "\n"  + t 

# LEMMATIZE
lem_words = [lemmatizer.lemmatize(x) for x in all_temp_string.split()]

all_long_string =  '' 
for t in lem_words:
    all_long_string += "\n" + t

all_vector= count_vectorizer.fit_transform(lem_words)
#features here is the built vocabulary words in a list

all_features= count_vectorizer.get_feature_names()




    ### Count frequencies in top and bottom postings using full vocabulary ###

# Top 10 predicted
test = [0]
test[0] = top10predict_long_string 
top10predict_vector = count_vectorizer.transform(test)
top10predict_mat = top10predict_vector.toarray()

# Bottom 10 predicted
test = [0]
test[0] = low10predict_long_string 
low10predict_vector = count_vectorizer.transform(test)
low10predict_mat = low10predict_vector.toarray()


# Top 10 actual
test = [0]
test[0] = top10act_long_string 
top10act_vector = count_vectorizer.transform(test)
top10act_mat = top10act_vector.toarray()

# Bottom 10 actual
test = [0]
test[0] = low10act_long_string 
low10act_vector = count_vectorizer.transform(test)
low10act_mat = low10act_vector.toarray()


    ### Difference count arrays top vs. bottom PREDICTED and plot ###

top_m_low_predict_mat = top10predict_mat - low10predict_mat
top_m_low_predict_mat = [x for x in top_m_low_predict_mat[0]]                            

count_dict= (zip(all_features, top_m_low_predict_mat))
count_dict= sorted(count_dict, key=lambda x:x[1], reverse=True)[0:15]
words= [w[0] for w in count_dict]
counts= [w[1] for w in count_dict]
x_pos= np.arange(len(words)) 
   
plt.figure(2, figsize=(15, 15/1.6180))
plt.title('15 most distinct words: top vs. bottom predicted score decile')
sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 2.5})
sns.barplot(x_pos, counts, palette='husl')
plt.xticks(x_pos, words, rotation=20) 
plt.xlabel('words')
plt.ylabel('counts')
plt.savefig('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/top_v_low_pred_freq_socskills.png')
plt.show()

    ### Difference count arrays top vs. bottom ACTUAL and plot ###

top_m_low_act_mat = top10act_mat - low10act_mat
top_m_low_act_mat = [x for x in top_m_low_act_mat[0]]                            

count_dict= (zip(all_features, top_m_low_act_mat))
count_dict= sorted(count_dict, key=lambda x:x[1], reverse=True)[0:15]
words= [w[0] for w in count_dict]
counts= [w[1] for w in count_dict]
x_pos= np.arange(len(words)) 
   
plt.figure(2, figsize=(15, 15/1.6180))
plt.title('15 most distinct words: top vs. bottom actual score decile')
sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 2.5})
sns.barplot(x_pos, counts, palette='husl')
plt.xticks(x_pos, words, rotation=20) 
plt.xlabel('words')
plt.ylabel('counts')
plt.savefig('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/top_v_low_act_freq_socskills.png')
plt.show()

  ### Difference count arrays bottom vs. top PREDICTED and plot ###

top_m_low_predict_mat = low10predict_mat - top10predict_mat 
top_m_low_predict_mat = [x for x in top_m_low_predict_mat[0]]                            

count_dict= (zip(all_features, top_m_low_predict_mat))
count_dict= sorted(count_dict, key=lambda x:x[1], reverse=True)[0:15]
words= [w[0] for w in count_dict]
counts= [w[1] for w in count_dict]
x_pos= np.arange(len(words)) 
   
plt.figure(2, figsize=(15, 15/1.6180))
plt.title('15 most distinct words: bottom vs. top predicted score decile')
sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 2.5})
sns.barplot(x_pos, counts, palette='husl')
plt.xticks(x_pos, words, rotation=20) 
plt.xlabel('words')
plt.ylabel('counts')
plt.savefig('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/low_v_top_pred_freq_socskills.png')
plt.show()

    ### Difference count arrays bottom vs. top ACTUAL and plot ###

top_m_low_act_mat = low10act_mat - top10act_mat
top_m_low_act_mat = [x for x in top_m_low_act_mat[0]]                            

count_dict= (zip(all_features, top_m_low_act_mat))
count_dict= sorted(count_dict, key=lambda x:x[1], reverse=True)[0:15]
words= [w[0] for w in count_dict]
counts= [w[1] for w in count_dict]
x_pos= np.arange(len(words)) 
   
plt.figure(2, figsize=(15, 15/1.6180))
plt.title('15 most distinct words: bottom vs. top actual score decile')
sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 2.5})
sns.barplot(x_pos, counts, palette='husl')
plt.xticks(x_pos, words, rotation=20) 
plt.xlabel('words')
plt.ylabel('counts')
plt.savefig('/Users/gschubert/Dropbox/PhD Course Notes/6.862 Machine Learning/complexity_project/Project proposal/low_v_top_act_freq_socskills.png')
plt.show()
