---
title: "How to setup ubuntu mirror page"
date: 2020-11-04
categories:
 - programming tools and environments 

tags:
 - linux
 - ubuntu

---
For faster apt update, you'll need mirror page, close to your own location.
setup can be done as follows.

1. file is in **/etc/apt/sources.list**.  Open it by:
```
$ sudo vi /etc/apt/sources.list
```

2. substitute. 
For me, i am in korea, so 
kr.archive.ubuntu.com -> mirror.kakao.com.
and in vi command, 
```
:%s/A/B 
```
substitutes A into B.
so for my case,
```
:%s/kr.archive.ubuntu.com/mirror.kakao.com
```

3. apply by
```
sudo apt get update
```

Done!