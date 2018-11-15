---
layout: post
title: Simple Word2Vec using SVD
date: 2018-06-14
description: # add
type_post: same
url_post: # add
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
So co-occurrence matrix is (n x n) matrix where n is no of words. it gives how frequent those words exists in a given distance. performed Tuncated SVD on the co-occurrence matrix and selected no of components/dimensions using variance explained and for my data selected as 200 dimensions. Now i have 200 dimension vector for each word.  and converted into dict as below.
~~~ python
#word 2 vectors dict
W2V_200d = {}
j = 0
for i in Words:
    W2V_200d[i] = svd_out[j]
    j = j + 1
~~~
We have 200 dimension vectors for each word and checked similary of vectors using cosine similarity like below.
~~~ python
def top_similar_words(v,Wordvectors,n,list_words):
	"""
	Gives output angles between one vector(v1) and all other vectors as matrix(v2)
	input:
	v shape (d,1)
	Wordvectors shape (m,d)
	n - no of similarities needed
	list_words - list of all words in Wordvectors in same order
	output:
	list of n words which are similar
	"""
	v1 = np.array(v)
	v1 = v1.reshape(-1,1)
	v2 = np.array(Wordvectors)
	#dot product
	nr = np.dot(v2,v1)
	#vector lengths 2-norm
	dr = (np.linalg.norm(v1)*np.linalg.norm(v2,axis=1)).reshape(-1,1)
	# angle i.e cosine inverse and domine for cosine inverse is
	# [-1,1] so clipping to [-1,1] and claculating angle
	ang = np.arccos(np.clip(nr/dr,-1,1))
	ang = ang.ravel()
	#sorting and getting index of top n
	ang_10 = np.argsort(ang)[0:n+1]
	ang_n = np.delete(ang_10,0)
	dict_tfidf = {i:v for i,v in enumerate(list_words)}
	return [dict_tfidf[i] for i in ang_n]
~~~
found similar words like below
For 'Good' got top 10 as ('enjoy', 'enjoying', 'fantastic', 'amazing', 'fabulous', 'nice', 'delicious', 'incredible', 'outstanding', 'perk')  
For 'sep' - ('dec', 'aug', 'august', 'november', 'sept', 'february', 'expired', 'expiry', 'exp', 'expiration')  
For 'yum' - ('yummy', 'delish', 'awesome', 'org', 'everybody', 'addictive', 'wow', 'amazing', 'addicted')  
For 'gin' - ('cocktail', 'cola', 'fizzy', 'clot', 'kinda', 'ting', 'cytoma', 'sour', 'rootbeer', 'soda') 
  
References:
1. applied ai course - Matrix factorization methods
