---
layout: post
title: Simple Word2Vec using SVD
date: 2018-06-14
description: # add
img:  # Add image post (optional)
tags: [SVD,Word2Vec]
---
In this i am discussing about creating word vectors using SVD and co-occurrence matrix. For doing this used data base of amazon 
fine food reviews data set, downloded from [kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews/data). Cleaned the data
and after that took top 5000 top tfidf words because of time taken to calculate co-occurrence matrix is high. Created co-occurrence matrix
as below
~~~ python
def cooccurrence_matrix(list_words,distance,sentances):
'''
Returns co-occurrence matrix of words with in a distance of occurrrence
input:
list_words: list of words to get the co-occurrance matrix in order
distance: distance between words
sentances: documets to check ( a list )
output:
co-occurance matrix in te order of list_words order
'''
#length of matrix needed
l = len(list_words)
#creating a zero matrix
com = np.zeros((l,l))
#creating word and index dict
dict_idx = {v:i for i,v in enumerate(list_words)}
for sentence in sentances:
sentence=sentence.strip()
tokens= sentence.split()
for pos,token in enumerate(tokens):
#if eord is in required words
if token in list_words:
#start index to check any other word occure or not
start=max(0,pos-distance)
#end index
end=min(len(tokens),pos+distance+1)
for pos2 in range(start,end):
#if same position
if pos2==pos:
continue
# if same word
if token == tokens[pos2]:
continue
#if word found is in required words
if tokens[pos2] in list_words:
#index of word parent
row = dict_idx[token]
#index of occurance word
col = dict_idx[tokens[pos2]]
#adding value to that index
com[row,col] = com[row,col] + 1
return com
~~~
