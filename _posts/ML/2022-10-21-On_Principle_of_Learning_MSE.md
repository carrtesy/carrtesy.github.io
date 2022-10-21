---
title: "On the principle of learning MSE"
date: 2022-10-21
last_modified_at: 2022-10-21
categories:
 - Machine Learning
tags:
 - Machine Learning
 - Deep Learning
 
use_math: true
---

We usually use mean square error (MSE) as our machine learning softwares' objective. What makes it valid objective function?



### What machine learning does

Let our model be a function that maps an input $ X $ to an output $ Y $.

Then, this objective can be represented as modeling $ P(Y \| X ; \theta ) $, where $\theta$ is a parameter that we learn. 

To put it in another way, we are learning parameter $\theta$ that maximizes the probability of output $Y$ is provided given the input $X$.



This objective is often referred as Maximum Likelihood Estimate. Formally:

$\theta_{ML} = argmax  P(Y \| X ; \theta )$



### Where MSE comes from

So here, we would like to find $\theta$ that maximizes $ P(Y \| X ; \theta ) $.

Let's introduce other assumptions, as follows:

> Assumption 1. Samples are iid.

> Assumption 2. $ P(y|x) $ follows Normal distribution with the same variance, i.e., $ P(y\|x) = \frac{1}{sqrt{2\pi}\sigma} e^{-frac{y-yhat}{2\sigma^2}} $

From those assumptions, our objective becomes,

$ \theta_{ML} = argmax  P(Y \| X ; \theta ) $

$ = argmax  \prod_{i=1}^{m}P(Y \| X ; \theta ) $

$ = argmax  \sum_{i=1}^{m}logP(Y \| X ; \theta ) $

$ = argmax  -mlog\sigma -\frac{1}{2}log2\pi -\sum_{i=1}^{m} \frac{(yhat-y)^2}{2\sigma^2} $

$ = argmax  -\sum_{i=1}^{m} \frac{(yhat-y)^2}{2\sigma^2} $



To convert it into minimization objective,

$ = argmin  \sum_{i=1}^{m} \frac{(yhat-y)^2}{2\sigma^2} $



Hence, our conclusion here is that, 

Minimizing MSE Loss is equivalent to maximizing log likelihood.



 

