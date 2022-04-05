---
title: "Conda cheatsheet"
date: 2022-04-05
categories:
 - programming tools and environments

tags:
 - conda
---

Here are some useful tips to set conda environments. 

## List up
```
conda info --envs
```

## Renaming conda environment
```
conda create --name {new_name} --clone {old_name} # copy
conda remove --name {old_name} --all # and erase
```

## Removing conda environment
```
conda remove --name {old_name} --all
```