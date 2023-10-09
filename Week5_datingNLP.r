#libraries
library(stringr)
library(dplyr)
library(ggplot2)
library(mosaic)
library(xtable)
library(gridExtra)
library(stopwords)
library(tokenizers)
library(quanteda)
library(SnowballC)
library(tidyverse)
library(caret)
library(mlbench)
library(cluster)
library(factoextra)

#loads the data
profiles <-
  data.frame(read.csv(
    file.path('dating', 'profiles.csv'),
    header = TRUE,
    stringsAsFactors = FALSE
  ))[,1:31]

#filter by age to improve performance
profiles <- filter(profiles, age>21&age<23)
#number of profiles
num <- nrow(profiles)
num

#prop between male and female
prop.table(table(profiles$sex))

essays <- select(profiles, starts_with("essay"))
essays <- apply(essays,
                MARGIN = 1,
                FUN = paste,
                collapse = " ")

#delete html chars
html <-
  c("<a", "class=.ilink.", "\n", "\\n", "<br ?/>", "/>", "/>\n<br")
html.pat <- paste0("(", paste(html, collapse = "|"), ")")
essays <- str_replace_all(essays, html.pat, "")


#delete url
url_pattern <-
  "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
essays <- str_replace_all(essays, url_pattern, "")


#delete numbers
num_patern <- ".[0-9]"
essays <- str_replace_all(essays, num_patern, "")

#delete hash tag
num_patern<-"(?:(?<=/s)|^)#(/w*[A-Za-z_]+/w*)"
essays <-str_replace_all(essays, num_patern, "")

#tokenizetion
tkns <-
  tokenize_word_stems(essays,
                      language = "english",
                      stopwords = stopwords::stopwords("en"))
#delete null values
tkns[is.na(tkns)] <- ""

#create dfm
d <-
  dfm(
    tokens(
      tkns,
      remove_url = TRUE,
      remove_separators = TRUE,
      remove_numbers = TRUE,
      remove_punct = TRUE,
      remove_symbols = TRUE
    )
  )
d<-dfm_remove(d,stopwords("english"))
#d=subset(or,select = -c("doc_id"))
d <- dfm_wordstem(d)

#filter the most significant terms
d <- dfm_trim(d, min_termfreq = 80, min_docfreq = 50)
nfeat(d)

#________________________________________________________DF-IDF
dfm.frame = convert(d, to = "data.frame")


#removing the doc_id column for calculations
dfm.frame <- select(dfm.frame,-doc_id)

#tf calculator
term.freq<-function(row){
  row/sum(as.numeric(row))
}

#idf calculator
inv.doc.freq<-function(col){
  num.of.texts<-length(col)
  word.count<-length(which(col>0))
  log10(num.of.texts/word.count)
}

#tf-idf calculator
tf.idf<-function(df, idf){
  df*idf
}

# Normalize all documents via TF.
dfm.tf <- apply(dfm.frame, 1, term.freq)


# Calculate the IDF vector 
dfm.idf <- apply(dfm.frame, 2, inv.doc.freq)


# Calculate TF-IDF for our corpus 
dfm.frame <-  apply(dfm.tf, 2, tf.idf, 
                    idf = dfm.idf)
#clean data
dfm.frame[is.nan(dfm.frame)] <- 0

#Transpose the matrix
dfm.frame <- t(dfm.frame)
dfm.frame = as.data.frame(dfm.frame)

#fix the names of the columns
colnames(dfm.frame) <- make.names(colnames(dfm.frame))

#________________________________________________________Train model

dfm.frame$biGender  <- ifelse(profiles$sex[1:num] == "f", 1, 0)

set.seed(110)
start.time<- Sys.time()
ind <- createMultiFolds(dfm.frame, k = 10, time = 3)
trctrl <-
  trainControl(
    method = "repeatedcv",
    number = 10,
    index = ind,
    allowParallel = TRUE
  )
fm_model <- train(
  factor(biGender) ~ .,
  data = dfm.frame,
  method = "rpart",
  trControl = trctrl,
  tuneLength = 0
)
train_time<-Sys.time()-start.time
fm_model$times
fm_model
fm_model$resample
confusionMatrix(data = fm_model)
#_______________________________________________________Remove Female/male words

essays <-str_replace_all(essays, head(stopwords::stopwords("english"), 40), "")
essays <-str_replace_all(essays, "[[:punct:]]", "")

m.top <- subset(essays, profiles$sex == "m") %>%
  str_split(" ") %>%
  unlist() %>%
  table() %>%
  sort(decreasing=TRUE) %>%
  names()
f.top <- subset(essays, profiles$sex == "f") %>%
  str_split(" ") %>%
  unlist() %>%
  table() %>%
  sort(decreasing=TRUE) %>%
  names()

# Male words
males= setdiff(m.top[1:500], f.top[1:500])
# Female words
females= setdiff(f.top[1:500], m.top[1:500])

new_d=dfm_remove(d,males)
new_d=dfm_remove(new_d,females)
#_________________________________________________________PCA
pca <- prcomp(new_d, scale = TRUE,
                 center = TRUE)
nd.frame<-convert(new_d, to = "data.frame")

#choose the top 150 words-pca
nd.frame<-nd.frame[,names(pca$center[1:150])]
#__________________________________________________________Clustering
#save to plots
pdf("Week5_datingNLP")
#2-Clusters
km2<-kmeans(nd.frame,2,nstart = 5)
fviz_cluster(km2,nd.frame,geom = "point")
#3-Clusters
km3<-kmeans(nd.frame,3,nstart = 30)
fviz_cluster(km3,nd.frame,geom = "point")
#4-Clusters
km4<-kmeans(nd.frame,4,nstart = 30)
fviz_cluster(km4,nd.frame,geom = "point")
#10-Clusters
km10<-kmeans(nd.frame,10,nstart = 30)
fviz_cluster(km10,nd.frame,geom = "point")

dev.off()
#___________________________________________________________SAVE
save(file = "Week5_datingNLP.rdata",fm_model,d,pca)

