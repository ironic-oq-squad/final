################################################################################
################################################################################
#############   PART 3 --> finding and adding synonyms to tweets
################################################################################
################################################################################



## packages
import os 
import csv
import numpy as np
import random
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import math
from nltk.stem import PorterStemmer
import sys
import re


# reading in using command line
len_argv = len(sys.argv)

# python3 validation_synonyms.py [filepath] [tweet index in csv]
if len_argv == 1:
    path_in = "./data/VT_incentive_v2_clean.csv"
elif len_argv == 2:
    path_in = sys.argv[1]
else:
    print("ERROR")
    print("Usage: python3 P3_synonyms.py [filepath to clean data]")
    exit()

## Loading cleaned tweets and tweet distance matrix
# os.chdir(r'C:\Users\smith\Google Drive\BDSI\final\Data')
fileRead = csv.reader(open(path_in, encoding="utf8"))
tweetsProcessed = list()
for row in fileRead:
	tweetsProcessed.append(row[0])


fileRead2 = csv.reader(open('./data/s.csv', encoding="utf8"))
s = list()
for row in fileRead2:
 	s.append(row)


## Now with the optimal number of clusters chosen from R, run k-medoids
optimal = input("Go to R for cluster evaluation. Enter the optimal number of clusters: ")
numberTopics = int(optimal)



print("Starting k-medoids")
# tweets
random.seed(123)
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
path_out_altered = re.sub(r'\.csv$', '_altered.csv', path_in)
# path_out_clean_stem = re.sub(r'\.csv$', '_clean_stem.csv', path_in)
path_out_altered_stem = re.sub(r'\.csv$', '_altered_stem.csv', path_in)

write_data(path_out_altered, tweetsAltered)
write_data(path_out_clean_stem, tweetsProcessed_stem)
write_data(path_out_altered_stem, tweetsAltered_stem)
print('data saved')
