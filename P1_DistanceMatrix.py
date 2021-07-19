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


## first, read in functions from 'distanceMatrices.py'
os.chdir(r'C:\Users\smith\Google Drive\BDSI\grace-sandbox\Data')
import distanceMatrices

## Read in data
## Change this to whatever your data's filepath is
os.chdir(r'C:\Users\smith\Google Drive\BDSI\sasha-sandbox')

## Change this to whatever your data's filename is
fileRead = csv.reader(open('VT_incentive_v2.csv', encoding="utf8"))
tweets = list()
for row in fileRead:
	tweets.append(row[1])
tweets_dup=tweets[1:]


## Pre-processing
tweetsProcessed = [distanceMatrices.preProcessingFcn(tweet) for tweet in tweets_dup]
print("# tweets before removing duplicates: " + str(len(tweetsProcessed)))
tweetsProcessed = list(OrderedDict.fromkeys(tweetsProcessed))  # removing duplicates
print("# tweets after removing duplicates: " + str(len(tweetsProcessed)))


## Calculate Distance Matrix
minMentions = 20
wordDistMethod = 'condProb'
newMethod = distanceMatrices.makeMatrices(tweetsProcessed, minMentions=minMentions, preProcess=False, wordDistMethod=wordDistMethod)

## Printing information
print('Number of tweets originally: ' + str(newMethod['nOriginal']))
print('Number of original unique words: ' + str(newMethod['vOriginal']))
print('Number of tweets after removing "pointless babble": ' + str(newMethod['n']))
print('Number of words appearing at least ' + str(minMentions) + ' times: ' + str(newMethod['v']))


## Save Distance Matrix to a CSV
os.chdir(r'C:\Users\smith\Google Drive\BDSI\grace-sandbox\Data')
d = newMethod['d']
ddf=pd.DataFrame(d)
ddf.to_csv('d_hat.csv', header = True, index=False)
s = newMethod['s']
sdf=pd.DataFrame(s)
sdf.to_csv('s.csv', header = False, index=False)

def write_data(filepath, tweets):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for tw in tweets:
            writer.writerow([tw])
write_data('./clean_data.csv', tweetsProcessed)


## FROM HERE WE GO TO R FOR EVALUATION
    ## to choose the best number of clusters
print("Distance matrix saved. Go to R to find the optimal number of clusters. P2_OptimalClustNum.R")