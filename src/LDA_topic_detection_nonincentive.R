library(tm)
library(readr)
library(quanteda)
library(dplyr)
library(textclean)
library(tidytext)
library(stringr)
library(textdata)
library(factoextra)
library(amap)
library(vader)
library(WeightedCluster)
library(cluster)
library(ggfortify)
library(factoextra)
library(ggplot2)
library(Matrix)
library(topicmodels)

## the data frame "topic" should only contain a column vector of tweets with header 'text'

setwd("/Users/smith/Desktop/sasha-sandbox/dm3_testing")  # CHANGE BACK TO ../data
topic = read_csv("VT_nonincentive_v2_altered_stem.csv") # your clean dataset
# e.g. VT_incentive_v2_clean, VT_incentive_v2_altered
dim(topic)
topic = unique(topic) # removing empty tweets
dim(topic)

tok <- topic %>% 
  corpus(text_field='text') %>% 
  tokens()

# head(tok)

dfmat_tweets<-dfm(tok) # creating term-doc mat
dim(dfmat_tweets)
### some features about tdm ######################################

# class(dfmat_tweets)
# ndoc(dfmat_tweets)
# nfeat(dfmat_tweets)
# topfeatures((dfmat_tweets), 20)
# head(featnames(dfmat_tweets), 50)
# head(dfmat_tweets)

####### expression ########

# nsim = 3000 # put nsim = nrow(topic) if corpus size is reasonable
#nsim = min(c(nrow(topic), 3000))
nsim = nrow(topic)
set.seed(2021)
sam = sample(1:nrow(dfmat_tweets),nsim)

samp_dat = dfmat_tweets[sam,]
print(dim(samp_dat))
# choos_cols = which(colSums(samp_dat) > 0)
# samp_dat = samp_dat[,choos_cols] # removing words that dont occur in any tweet
# choose_rows = which(rowSums(samp_dat) > 0)
# samp_dat = samp_dat[choose_rows,] # removing blank tweets
names = samp_dat@Dimnames[[2]]

word_expression = t(matrix(as.vector(samp_dat),nrow = nrow(samp_dat),ncol = ncol(samp_dat)))
rownames(word_expression) = names
colnames(word_expression) = paste0("d_",1:ncol(word_expression))
print(dim(word_expression))
word_expression[1:5, 1:5]


dtm = as(word_expression, "dgCMatrix") # converting tdm to dgC class for using it in LDA

#############################################################################################
# maximum likelihood estimate of optimal number of topics 
#############################################################################################

grid_search_topic_k = seq(5,80,by = 5) # possible topic sizes
mod_log_lik = NULL

for (i in grid_search_topic_k) 
{
  mod = LDA(t(dtm), k=i, method="Gibbs",
            control=list(alpha=0.5, iter=200, seed=12345, thin=1))
  mod_log_lik = c(mod_log_lik,mod@loglikelihood)
  print(mod@loglikelihood)
}

##### topic size (k) vs likelihood plot #######################

plot(grid_search_topic_k,mod_log_lik,ty = "b",
     xlab = "number of topics (k)",
     ylab = "Log likelihood",
     pch = 19)
abline(v = grid_search_topic_k[which.max(mod_log_lik)],lty = 2)

###############################################################

# MLE of optimal no. of topics

k_opt_lda = grid_search_topic_k[which.max(mod_log_lik)] 

# running LDA with optimal no. of topics 

text_lda = LDA(t(dtm), k = k_opt_lda, 
               method="Gibbs",
               control=list(alpha=0.5, iter=200, seed=12345))


# getting feature words for each topic 

text_topics = tidy(text_lda, matrix = "beta")
text_top_terms1 = text_topics %>%
  group_by(topic) %>%
  top_n(25, beta) %>% # can set up no. of feature words per topic
  ungroup() %>%
  arrange(topic, -beta)

key_words = list()
for(i in 1:k_opt_lda)
{
  key_words[[i]] = c(text_top_terms1[text_top_terms1$topic == i,2])
}

print(key_words) # feature words for all k_opt_lda many topics

# feature words per topic plotted as per ranked beta values 
# see before plotting k_opt_lda value, might crash R

# text_top_terms1 %>%
#   mutate(term = reorder(term, beta)) %>%
#   ggplot(aes(term, beta, fill = factor(topic))) +
#   geom_col(show.legend = FALSE) +
#   facet_wrap(~ topic, scales = "free") +
#   coord_flip()

#############################################################################################
# sentiment analysis on topics
#############################################################################################

# assigning topics to tweets based on maximum posterior probability of topics 

topic_assign_prob = text_lda@gamma # matrix of posterior probability of topics per document

tweet_clusters = vector(mode = "list", length = k_opt_lda)
for(x in 1:nrow(topic_assign_prob))
{
  prob_vec = topic_assign_prob[x,]
  t = which(prob_vec == max(prob_vec))
  for(i in t)
  {
    tweet_clusters[[i]] = c(tweet_clusters[[i]],x) 
  }
}

tweets_clustered = list() # tweets assigned to each topic
for(i in 1:k_opt_lda)
{
  tweets_clustered[[i]] = topic$text[sam[tweet_clusters[[i]]]]
}

# Running vader sentiment analysis on each of the tweet clusters

#mean_sentiment_per_topic = c() # mean sentiment score per topic

# for(i in 1:k_opt_lda)
# {
#   myDf <- data.frame(text = tweets_clustered[[i]], stringsAsFactors = FALSE) %>% 
#     filter(text!="") %>% #removing empty rows                                                              
#     vader_df()   #this code takes forever, so consider saving df when you run it
#   
#   mean_sentiment_per_topic[i] = mean(myDf$compound)
#   
#   plot = ggplot(myDf, aes(x=compound))+
#          geom_freqpoly(binwidth=0.1)+
#          labs(x="compound score",
#          title=paste0("Sentiment Analysis Plot: Vaccine Trust - topic ",i),
#          subtitle="vader (individual tweets) (bw=0.1)")
#   print(plot) # plot of sentiment score vs no of tweets with that sentiment value
# }

#############################################################################################

# have a look a some of the results (posterior distributions)
tmResult <- posterior(text_lda)
# format of the resulting object
attributes(tmResult)

# topics are probability distribtions over the entire vocabulary
beta <- tmResult$terms   # get beta from results
dim(beta)                # K distributions over ncol(DTM) terms

# for every document we have a probability distribution of its contained topics
theta <- tmResult$topics 
dim(theta)               # nDocs(DTM) distributions over K topics


library(LDAvis)
library(servr)


json <- createJSON(
  phi = beta, 
  theta = theta, 
  doc.length = rowSums(t(dtm)), 
  vocab = colnames(t(dtm)), 
  term.frequency = colSums(t(dtm)),
  R = 30,
  plot.opts = list(xlab="", ylab="")
)
serVis(json)


# order of json clusters (wrt key_words clusters)
new.order <- RJSONIO::fromJSON(json)$topic.order
print(new.order)


list_nonincentive <- list()
list_nonincentive[[1]] <- c(2, 10, 16, 34)
list_nonincentive[[2]] <-c(1, 3, 4, 5, 6, 7, 11, 12, 13, 17, 33, 50)
list_nonincentive[[3]] <- c(8, 9, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49)


# grouping clusters -- key_words indices

list2_nonincentive <- list()
for(i in 1:length(list_nonincentive)) {
  list2_nonincentive[[i]] <- new.order[list_nonincentive[[i]]]
}

# grabbing tweets
tweet_new_clust_nonincentive <- vector(mode = "list", length = 3)
for(i in 1:3) {
  vec <- list2_nonincentive[[i]]
  for(j in vec) {
    tweet_new_clust_nonincentive[[i]] <- c(tweet_new_clust_nonincentive[[i]], tweet_clusters[[j]])
  }
}



setwd("/Users/smith/Desktop/sasha-sandbox/data")  # CHANGE BACK TO ../data
sentiments_nonincentive <- read_csv("tweets_predicted_nonincentive.csv")
tweet_clust_sentiments_nonincentive <- list()
for(i in 1:3) {
  ids <- tweet_new_clust_nonincentive[[i]]
  tweet_clust_sentiments_nonincentive[[i]] <- sentiments_nonincentive[ids, 2]
}

table(tweet_clust_sentiments_nonincentive[[1]])
table(tweet_clust_sentiments_nonincentive[[2]])
table(tweet_clust_sentiments_nonincentive[[3]])


# getting sentiments for indiv. clusters
individual_noninc <- vector(mode = "list", length = k_opt_lda)
sents <- c()
for(i in 1:k_opt_lda) {
  individual_noninc[[i]] <-  sentiments_nonincentive[tweet_clusters[[i]], 2]  
  t <- table(individual_noninc[[i]])
  print(t)
  sents <- c(sents, ifelse(t[1] < t[3], 'positive', 'negative'))
}


  
  