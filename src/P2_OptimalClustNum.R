## Packages
install.packages('factoextra')
library('factoextra')
library('readr')


## Read in data
getwd()
setwd("../data")
d_hat <- read_csv("d_hat.csv")


## Evaluation methods for k-medoids

#Elbow method WSS: best k is at the elbow
fviz_nbclust(d_hat, cluster::pam, method = "wss", k.max=30)+
  labs(subtitle = "Elbow method")

#Silhouette method: labels best k
fviz_nbclust(d_hat, cluster::pam, method = "silhouette", k.max=30)+
  labs(subtitle = "Silhouette method")


## After choosing the optimal number of clusters based on these
  ## evaluation metrics, use that number as your input in P3_synonyms.py