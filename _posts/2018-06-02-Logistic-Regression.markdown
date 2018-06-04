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
so \\(d_i > 0\\) for all positive samples because w is in same direction and \\(d_i < 0\\) for all negative samples. if \\(y_i = 1\\) for positive points and -1 for negetive points. let's assume plane is passing through origin i.e no intercept so b = 0. for correctly classified points \\(y_iW^Tx_i > 0\\) and for all wrong classified points \\(y_iW^Tx_i < 0\\). Main objective of classification is to classify correctly so out objective in Logistic regression is to find the W and b such that  \\(\sum_{i=1}^n (y_iW^Tx_i)\\) is maximum. 
So optimization problem is \\[ W^* = argmax \sum_{i=1}^n (y_iW^Tx_i)\\]  
But this will fails in some conditions one of them is listed below.   
![Logistic regression error case]({{site.baseurl}}/assets/img/lr_error_case.jpg)    
From above diagram  for \\(\pi_1\\), \\( \sum_{i=1}^n (y_iW_1^Tx_i)\\) = 1+1+1+1+1+1+1+1-50 = -42 and for \\(\pi_2\\), \\( \sum_{i=1}^n (y_iW_2^Tx_i)\\) = 1+2+3+4-1-2-3-4+1 = 1. so plane \\(\pi_2\\) is better for maximizing the sum but we can see that accuracy of \\(\pi_1\\) is higher than \\(\pi_2\\). it is because of single outlier in our train data. we can overcome this by squeezing the maximum distance points into min distance. this can be done by sigmoid function as shown below.  
![Sigmaoid]({{site.baseurl}}/assets/img/sigmoid.jpg)  
[sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) wil give values between [0,1] based on input real value. so we can decrese the outlier effect on the optimization problem.
so our optimization problem is \\[ W^* = argmax \sum_{i=1}^n \sigma(y_iW^Tx_i)\\]  
from operations that preserve Argmax, if g(x) is monotonic function then argmax f(x) = argmax g(f(x)). if we take g(x) as a log fuction then we can get some good properties as converting exponent as mutiplication, converting multiplication into addition. so our final optimization problem is \\[W^* = argmax \sum_{i=1}^n \log(\sigma(y_iW^Tx_i)) \\]
\\[W^* = argmax \sum_{i=1}^n \log(\frac{1}{1+e^(-y_iW^Tx_i)}) \\]  
\\[W^* = - argmax \sum_{i=1}^n \log(1+e^(-y_iW^Tx_i)) \\]  
\\[W^* =  argmin \sum_{i=1}^n \log(1+e^(-y_iW^Tx_i)) \\]    

