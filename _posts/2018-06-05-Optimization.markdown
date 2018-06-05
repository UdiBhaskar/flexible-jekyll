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
eg: \\[f(x) = x^3 − 3x^2 − 45x \\]
\\[f'(x) = 3^2 - 6x - 45 = 0 \\]
\\[x^* = -3 \text{ or } x^* = 5\\]
\\[f''(x) = 6x − 6\\]  
When x = -3, f ''(-3) = -24 and this means a MAXIMUM point.  
When x = 5, f ''(x) = 24 and this means a MINIMUM pont.   
This definition can be extended to a multivariate function\\(f(x_1,x_2...x_n)\\),where the condition for finding a stationary point is \\[\frac{\partial f}{\partial x_i} = 0 \text{ for all } x_1,x_2..x_n\\]
unlike univariate functions, it is more difficult to determine whether X* corresponds to a maximum or minimum stationary point.The difficulty arises because we need to consider the partial derivatives \\(\frac{\partial^2 f}{\partial x_i \partial x_j}\\) for all possible pairs of i,j. so this complete set is given by [hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix) H(x).  
1. X is minimum if H(x) is [positive definite](https://en.wikipedia.org/wiki/Positive-definite_matrix) i.e \\(X^THX > 0\\) for any non-zero column vector X.
2. X is maximum if H(x) is negative definite i.e \\(X^THX < 0\\) for every non zero column vector X.
3. X is saddle point if H(x) is indefinite i.e for some values of X \\(X^THX > 0\\) and for some values \\(X^THX < 0\\)  
eg: \\[f(x,y) = 3x^2+2y^3-2xy\\]
\\[\frac{\partial f}{\partial x} = 6x-2y = 0 \text{ and } \frac{\partial f}{\partial y} = 6y^2-2x = 0\\]
solution for above equation is \\(x^* = y^* = 0 \text{ and } x^* = 1/27, y^* = 1/9\\)  
Hessain of f is 
\begin{bmatrix}
    x_{11} & x_{12}  \\  
     x_{11} & x_{13}
 \end{bmatrix}
