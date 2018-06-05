---
layout: post
title: Optimization 
date: 2018-06-05 
description: # add
img:  # Add image post (optional)
tags: [Optimization,lagrangian,gradient descent,Newtons method]
---
Optimization is a methodology for finding the maximum or minimum value a real function.More generally,optimization includes finding 
"best available" values of some objective function given a defined domain (or input), including a variety of different types of objective 
functions and different types of domains.  
#### Unconstrained Optimization:  
In an unconstrained optimization problem, the task is to locate the solution x* that maximizes or minimizes f(x) without imposing anycconstraints on x*. The solution x*, which is known as a stationary point, can be found by taking the first derivative of f(x) and setting it to zero. so for analytical steps are given below:
1. Find the f'(x) and set it to zero and then find x = x*  
2. if f''(x) > 0  then x* is local minimum   
3. if f''(x) < 0  then x* is local maximum    
4. If f''(x) = 0 then x* is inflection point of f(x) as shown in below fig  
![maximum and Minimum]({{site.baseurl}}/assets/img/max_min.jpg)  
eg: \\[[f(x) = x^3 − 3x^2 − 45x \\]
\\[f'(x) = 3^2 - 6x - 45 = 0 \\]
\\[x* = -3 or x* = 5\\]  
\\[f''(x) = 6x − 6\\]  
When x = -3, f ''(-3) = -24 and this means a MAXIMUM point.  
When x = 5, f ''(x) = 24 and this means a MINIMUM pont.  


