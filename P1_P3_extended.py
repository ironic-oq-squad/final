####################################################################################
####################################################################################
#############   PART 1 Extended --> getting the distance matrices, adding synonyms
####################################################################################
####################################################################################


## packages
import os 
import csv
import pandas as pd
from collections import OrderedDict
import numpy as np
import random
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import math
from nltk.stem import PorterStemmer


## first, read in functions from 'distanceMatrices.py'
# os.chdir('')
import distanceMatrices_final



## Read in data

## Read in data
## Change this to whatever your data's filepath is
os.chdir(r'C:\Users\smith\Google Drive\BDSI\grace-sandbox\Data')

## Change this to whatever your data's filename is
fileRead = csv.reader(open('Twitter_mani.csv', encoding="utf8"))
tweets = list()
for row in fileRead:
	tweets.append(row[4])   ## change this depending on how your data is stored
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
os.chdir(r'C:\Users\smith\Google Drive\BDSI\final\Data')
d = newMethod['d']
df=pd.DataFrame(d)
df.to_csv('d_hat.csv', header = True, index=False)


## FROM HERE WE GO TO R FOR EVALUATION to choose the best number of clusters
## With the optimal number of clusters chosen from R, run k-medoids
optimal = input("Go to R for cluster evaluation. Enter the optimal number of clusters: ")
numberTopics = int(optimal)


print("Starting k-medoids")
# tweets
random.seed(123)
s = newMethod['s']
kmedoids_instance_tweets = kmedoids(s, range(numberTopics), data_type='distance_matrix')

kmedoids_instance_tweets.process()
clusters_tweets = kmedoids_instance_tweets.get_clusters()
medoids_tweets = kmedoids_instance_tweets.get_medoids()


# TF-IDF - separating tweets by cluster (according to k-medoids)
clusters_tweets_combined = []  # list of strings, where string is concatenation of each tweet in a cluster
for c in clusters_tweets:
    cluster_tweets = []
    combined_tweets = ""
    for i in c:
        combined_tweets = combined_tweets + " " + tweetsProcessed[i]
    clusters_tweets_combined.append(combined_tweets)


# TF-IDF - getting scores
# higher score means more relevant to the document
vectorizer = TfidfVectorizer()
tfidf_sparse = vectorizer.fit_transform(clusters_tweets_combined)  # gives location of feature ((x,y) = (cluster, feature number)) and its tf-idf score
scores = (tfidf_sparse.toarray())
features = vectorizer.get_feature_names()
print('tfidf scores done')


# TF-IDF - top features for each cluster
# PROPORTION: maybe change number to be a proportion (10%?) of the cluster?
# CAP: currently capping at (# total synonyms we want) * (proportion of total words that are in cluster i)
clusters_num_words = [len(word_tokenize(cluster)) for cluster in clusters_tweets_combined]  # number of words per cluster
total_num_words = sum(clusters_num_words)  # total number of words in all tweets
cap_synonym = int(math.ceil(total_num_words * 0.0033))  # total number of synonyms we want
clusters_prop_words = [num / total_num_words for num in clusters_num_words]  # proportion of total words in each cluster
clusters_syn_cap = [math.ceil(cap_synonym * prop) for prop in clusters_prop_words]  # cap of how many base words can be taken from each cluster
print("Total number of words: " + str(sum(clusters_num_words)))
print("Number of total words per cluster: " + str(clusters_num_words))
# print("METHOD: * 0.0033")
print("Number of base words per cluster: " + str(clusters_syn_cap))


def top_tfidf_feats(row, features, top_n):
    topn_ids = np.argsort(row)[::-1][:top_n]  # sort indices by the tf-idf score
    top_feats = [features[i] for i in topn_ids]  # list of features
    # top_feats_tfidf = [(features[i], row[i]) for i in topn_ids]  # list of tuples (feature, score)
    # df = pd.DataFrame(top_feats_tfidf)
    # df.columns = ['feature', 'tfidf']
    # print(df)
    return top_feats

# get list of all top features, excluding those shared among clusters
topwords_all = set()
for i in range(0, scores.shape[0]):
    topwords_all.symmetric_difference_update(top_tfidf_feats(scores[i], features, int(clusters_syn_cap[i])))  # keeps everything but duplicates
topwords_all = list(topwords_all)  # converting to list

print("Total base word cap: " + str(cap_synonym))
print("Total base words collected, w/ overlap: " + str(sum(clusters_syn_cap)))
print("Total base words collected, w/out overlap: " + str(len(topwords_all)))


# SYNONYMS
# IN: list of "base" words to consider in hash map
# OUT: hash map including all inputted base words where key:pair = synonym:base word
def make_mapping(words):
    mapping = {}
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemma_names():
                if lemma not in mapping and lemma != word:
                    if '_' not in lemma:
                        mapping[lemma] = word
    return mapping

# IN: tweets = original corpus, mapping = synonym:base mapping
# OUT: altered tweets with all base words added to end of tweet if synonym found in tweet
def alter_tweets(tweets, mapping):
    count_added=0
    new_tweets = []
    for tweet in tweets:
        for word in word_tokenize(tweet):
            if word in mapping:
                tweet = tweet + " " + mapping[word]
                count_added = count_added + 1
        new_tweets.append(tweet)
    print("Total number of words added to tweets: " + str(count_added))
    return new_tweets


# IN: filepath of output file
# OUT: tweets to be exported
def write_data(filepath, tweets):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text'])
        for tw in tweets:
            writer.writerow([tw])


# IN: list of tweets (strings)
# OUT: list of stemmed tweets (strings)
def stem(tweets):
    ps = PorterStemmer()
    new_tweets = []
    for tw in tweets:
        tw = ' '.join([ps.stem(word) for word in tw.split()])
        new_tweets.append(tw)
    return new_tweets


# print("All top words: ")
# for word in topwords_all:
#     print(word)


## Adding base words to tweets
tweetsAltered = alter_tweets(tweetsProcessed, make_mapping(topwords_all))
print('altering tweets done')


## Stemming tweets
tweetsProcessed_stem = stem(tweetsProcessed)
tweetsAltered_stem = stem(tweetsAltered)
print('stemming tweets done')


## Writing data
write_data('./Twitter_mani_clean.csv', tweetsProcessed)
write_data('./Twitter_mani_altered.csv', tweetsAltered)
write_data('./Twitter_mani_clean_stem.csv', tweetsProcessed_stem)
write_data('./Twitter_mani_altered_stem.csv', tweetsAltered_stem)
print('data saved')