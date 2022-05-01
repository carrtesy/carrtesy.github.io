---
title: "Useful shellscripts"
date: 2022-05-01
categories:
 - programming tools and environments

tags:
 - linux
 - shell
---

Here are some useful shell scripts. 


## printing log file

```shell
watch 'head -n 2 $1; tail $1'
```
where $1: log file.