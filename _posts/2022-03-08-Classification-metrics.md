---
title: "Classification Metrics: Accuracy is not enough"
date: 2022-03-08
last_modified_at: 2022-03-08

categories:
 - Metrics

tags:
 - Classification
 - Metrics
 - Deep Learning
 - Machine Learning

use_math: true
---



Sometimes, accuracy isn't enough as a metric to evaluate classifier's performance. Are there any other metrics more than that? Let's discuss the ways to measure classifiers' performances.



## Accuracy is not enough

Accuracy is (a bit informally) defined as:

$ accuracy = \frac{How many correct answers have you made}{How many questions are there} $

Simple enough. What's the problem with this?



Consider a following situation.

>Tommy predicts tomorrow's weather every night (sunny or rainy). 
>
>His prediction accuracy is **90%**!

Seems great, but what if:

>Tommy predicts tomorrow's weather every night (sunny or rainy).
>
>**He predicts "sunny" every time.**
>
>His prediction accuracy is **90%**!



We wouldn't need model, if that always says "yes" to everything. 



## TP, TN, FP, FN

Before we get into details, Let's define *TP, TN, FP, FN* first.

It's usually helpful to get the meaning word for word, as follows:



| Model's prediction result | What model said | Meaning                                                      |
| ------------------------- | --------------- | ------------------------------------------------------------ |
| True                      | Positive        | Model said positive (true) and that was correct. (Actually it was true.) |
| True                      | Negative        | Model said positive (true) and that was wrong. (Actually it was false.) |
| False                     | Positive        | Model said negative (false) and that was correct. (Actually it was false.) |
| False                     | Negative        | Model said negative (false) and that was wrong. (Actually it was false.) |



## Confusion matrix

To see the values of *TP, TN, FP, FN* in concise manner, constructing confusion matrix helps.

(to be cont.)



## I'm still working on this content 

I haven't finished writing this content. Let me finish ASAP. Thanks!

---

This is the 1st draft written on Mar 08, 2022

1st draft: 2022-03-08
