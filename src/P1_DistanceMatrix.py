################################################################################
################################################################################
#############   PART 1 --> getting the distance matrices
################################################################################
################################################################################


## packages
import os 
import csv
import pandas as pd
from collections import OrderedDict
import sys
import re
import random


## first, read in functions from 'distanceMatrices.py'
# os.chdir('')
import distanceMatrices_final



## Read in data

# reading in using command line
len_argv = len(sys.argv)

# python3 validation_synonyms.py [filepath] [tweet index in csv]
if len_argv == 1:
    path_in = "./data/Twitter_mani.csv"
    tweet_index = 4
elif len_argv == 3:
    path_in = sys.argv[1]
    tweet_index = int(sys.argv[2])
else:
    print("ERROR")
    print("Usage: python3 P1_DistanceMatrix.py [filepath to data] [tweet index in csv]")
    exit()

## Read in data
## Your data's filepath:
# os.chdir(r'C:\Users\smith\Google Drive\BDSI\grace-sandbox\Data')

## Your data's filename:
fileRead = csv.reader(open(path_in, encoding="utf8"))
tweets = list()
for row in fileRead:
	tweets.append(row[tweet_index])   ## change this depending on how your data is stored
tweets_dup=tweets[1:]


## Pre-processing
tweetsProcessed = [distanceMatrices_final.preProcessingFcn(tweet) for tweet in tweets_dup]
print("# tweets before removing duplicates: " + str(len(tweetsProcessed)))
tweetsProcessed = list(OrderedDict.fromkeys(tweetsProcessed))  # removing duplicates
print("# tweets after removing duplicates: " + str(len(tweetsProcessed)))


## Calculate Distance Matrix
minMentions = 20
wordDistMethod = 'condProb'
newMethod = distanceMatrices_final.makeMatrices(tweetsProcessed, minMentions=minMentions, preProcess=False, wordDistMethod=wordDistMethod)

## Printing information
print('Number of tweets originally: ' + str(newMethod['nOriginal']))
print('Number of original unique words: ' + str(newMethod['vOriginal']))
print('Number of tweets after removing "pointless babble": ' + str(newMethod['n']))
print('Number of words appearing at least ' + str(minMentions) + ' times: ' + str(newMethod['v']))


## Save Distance Matrix to a CSV
# os.chdir(r'C:\Users\smith\Google Drive\BDSI\final\Data')
d = newMethod['d']
ddf=pd.DataFrame(d)
ddf.to_csv('./data/d_hat.csv', header = True, index=False)
s = newMethod['s']
sdf=pd.DataFrame(s)
sdf.to_csv('./data/s.csv', header = False, index=False)

def write_data(filepath, tweets):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for tw in tweets:
            writer.writerow([tw])

path_out = re.sub(r'\.csv$', '_clean.csv', path_in)
write_data(path_out, tweetsProcessed)

## FROM HERE WE GO TO R FOR EVALUATION
    ## to choose the best number of clusters
print("Distance matrix saved. Go to R to find the optimal number of clusters. P2_OptimalClustNum.R")