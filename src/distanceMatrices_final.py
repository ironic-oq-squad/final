################################################################################
################################################################################
#############   Functions for creating word- and tweet-level distance matrices
################################################################################
################################################################################

## packages needed for makeMatrices function
	# used for many things
import numpy as np
	# used in pre-processing function
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import demoji
demoji.download_codes()
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


### define functions needed for makeMatrices function
## pre-processing: stem, remove URLs and stop words
def preProcessingFcn(tweet, removeWords=list(), stem=False, removeURL=True, removeStopwords=True, 
	removeNumbers=True, removeHashtags=True, removeAt=True, removePunctuation=True, lem=True, demojify=True):
	ps = PorterStemmer()
	lm = WordNetLemmatizer()
	tweet = tweet.lower()
	if removeURL==True:
		tweet = re.sub(r"http\S+", " ", tweet)
	if removeHashtags==True:
		tweet = tweet.replace('#', ' ')
	if removeAt==True:
		tweet = tweet.replace('@', ' ')
	if removeNumbers==True:
		tweet=  ''.join(i for i in tweet if not i.isdigit())
	if demojify==True:
		tweet=demoji.replace(tweet)
	if removePunctuation==True:
		tweet = re.sub(r"[,.;@#?!&$:]+\ *", " ", tweet)
	if removeStopwords==True:
		tweet = ' '.join([word for word in tweet.split() if word not in stopwords.words('english')])
	if len(removeWords)>0:
		tweet = ' '.join([word for word in tweet.split() if word not in removeWords])
	if lem==True:
		tweet = ' '.join([lm.lemmatize(word) for word in tweet.split()])
	if stem==True:
		tweet = ' '.join([ps.stem(word) for word in tweet.split()])
	return tweet



def makeMatrices(tweets, minMentions=10, preProcess=True, verbose=True, wordDistMethod = 'condProb',
	makeTweetDistanceMatrix=True):
	## input tweets, output matrices w, c, t, d, s

	nOriginal = len(tweets)

	### pre-processing
	if preProcess==True:
		if verbose:
			print('Pre-processing')
		tweets = [preProcessingFcn(tweet) for tweet in tweets]

	### w = word matrix
	## rows=tweets, columns=words
	## w_ij=how many times word j appears in tweet i
	if verbose:
		print('Creating word matrix')
	vectorizer = CountVectorizer(strip_accents='unicode')
	tweetsNew = vectorizer.fit_transform(tweets)
	wOriginal1 = tweetsNew.toarray()
	wOriginal = np.where(wOriginal1>=1, 1, 0)
	colWords = vectorizer.get_feature_names()
	vOriginal = wOriginal.shape[1]

	# restrict to words with at least minMentions mentions
	numMentions = np.sum(wOriginal, axis=0)
	newWords = [colWords[i] for i in list(np.where(numMentions>=minMentions)[0])]
	w = wOriginal[:,list(np.where(numMentions>=minMentions)[0])]
	tweetsIndex = list(np.where(np.sum(w, axis=1)>0)[0])
	w = w[tweetsIndex,:]

	n = w.shape[0]
	v = w.shape[1]

	### c = word co-occurrence matrix
	## rows=words, columns=words
	## c_ij=number of times words i and j appear in the same tweet
	if verbose:
		print('Making co-occurrence matrix')
	c = w.T.dot(w)

	### t = word transition matrix
	## rows=words, columns=words
	## t_ij: if word i appears in a tweet, probability that word j also appears in that tweet
	if verbose:
		print('Making word transition matrix')
	t = np.divide(c, np.diag(c)).transpose()

	### d = word distance matrix
	## rows=words, cols=words
	## d_ij=minimum 'distance' from word i to word j
	if verbose:
		print('Making word distance matrix')
	if wordDistMethod=='condProb':
		# d(word i, word j) = 2 - [ P(word i|word j) + P(word j|word i) ]
		d = 2 - t - t.T
	if wordDistMethod=='Jaccard':
		# d(word i, word j) = 1- (# tweets w/ words i AND j)/(# tweets w/ words i OR j)
		colSumsMatrix = np.tile(np.sum(w, axis=0).transpose(), (v,1))
		d = 1 - c/(colSumsMatrix + colSumsMatrix.T-c)

	### s = tweet distance matrix
	## rows=tweets, cols=tweet
	## s_ij=mean d[words in tweet i & not in j, 
	##				words in tweet j & not in i]
	if makeTweetDistanceMatrix:
		if verbose:
			print('Making tweet distance matrix')
		s = np.zeros([n, n])
		for i in range(n):
			for j in range(n):
				if i<=j:
					iWords = np.where(w[i,:]>0)[0]
					jWords = np.where(w[j,:]>0)[0]
					ijIntersect = set(iWords) & set(jWords)
					iWordsUse = [word for word in iWords if word not in ijIntersect]
					jWordsUse = [word for word in jWords if word not in ijIntersect]
					dtweetsij = d[np.ix_(iWordsUse, jWordsUse)]
					entries = dtweetsij.shape[0]*dtweetsij.shape[1]
					if entries !=0:
						ij = np.sum(dtweetsij) / entries
						s[i,j] = ij
						s[j,i] = ij

	else:
		s = 'No tweet distance matrix'

	print(d.shape)
	print(w.shape)

	### everything to return
	output = dict()
	output['processedTweets'] = tweets
	output['n'] = n # number of tweets
	output['v'] = v # number of words (Vocabulary)
	output['words'] = newWords # words
	output['tweetIndex'] = tweetsIndex # tweets used
	output['nOriginal'] = nOriginal # original number of tweets
	output['vOriginal'] = vOriginal # original number of words
	output['w'] = w # Word matrix
	output['c'] = c # Co-occurrence matrix
	output['t'] = t # Transition matrix
	output['d'] = d # Distance matrix between words
	output['s'] = s # diStance matrix between tweets
	return output



