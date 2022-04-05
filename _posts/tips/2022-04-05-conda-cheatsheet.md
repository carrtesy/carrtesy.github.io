---
title: "Conda cheatsheet"
date: 2022-04-05
categories:
 - programming tools and environments

tags:
 - conda
---

Here are some useful tips to set conda environments. 


## Renaming conda environment
```
conda create --name {new_name} --clone {old_name}
conda remove --name {old_name} --all
```