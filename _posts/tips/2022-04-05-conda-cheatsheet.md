---
title: "Conda cheatsheet"
date: 2022-04-05
categories:
 - programming tools and environments
last_modified_at: 2022-11-08
tags:
 - conda
---

Here are some useful tips to set conda environments. 

## Settings
```
conda create -n {env_name} python=3.9
```

```
conda install -y matplotlib
```

## List up
### The environments
```
conda info --envs
```
### Packages in the environments
```
conda list -n {env_name}
```



## Installations

```
conda install -y pip matplotlib seaborn pandas numpy notebook scikit-learn && conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch && pip install tensorboard
```



## Save conda environment as yaml

```
conda env export > {env_name.yaml}
```



## Create using yaml file

```
conda env create -f {env_name}.yaml
```



## Removing conda environment

```
conda remove --name {old_name} --all
```



## Renaming conda environment

```
conda create --name {new_name} --clone {old_name} # copy
conda remove --name {old_name} --all # and erase
```
