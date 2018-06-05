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
4. If f''(x) = 0 then x* is critical point of f(x)

