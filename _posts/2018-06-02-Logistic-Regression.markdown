---
layout: post
title: Logistic Regression
date: 2018-06-04 
description: # add
img:  # Add image post (optional)
tags: [Logistic Regression,LR,Regression]
---
Logistic regression predicts the probability of an outcome that can only have two values. Let's take a data that having two classes, class -1 and class 1 as shown in below fig.  
![Logistic regression explination]({{site.baseurl}}/assets/img/log_reg.jpg)    

from above fig logiction regression function is to find the plane \\(\pi (W,b)\\) that best seperated the positive and negative classes. from above fig distance from point \\(x_i\\) to plane \\(\pi\\) i.e \\(d_i\\) = \\(\frac{W^T.x_i+b}{\||W||}\\) and W is unit vector so ||W|| = 1.  
so \\(d_i > 0\\) for all positive samples because w is in same direction and \\(d_i < 0\\) for all negative samples. if \\(y_i = 1\\) for positive points and -1 for negetive points then  for correctly classified points \\(y_iW^Tx_i > 0\\) and for all wrong classified points \\(y_iW^Tx_i < 0\\). Main objective of classification is to classify correctly so out objective in Logistic regression is to find the W and b such that  \\(\sum_{i=1}^n (y_iW^Tx_i)\\) is maximum. 
So optimization problem is \\[ W^* = argmax \sum_{i=1}^n (y_iW^Tx_i)\\]
