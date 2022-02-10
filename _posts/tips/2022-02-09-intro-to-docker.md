---
title: "Introduction to docker"
date: 2022-02-09
categories:
 - programming tools and environments

tags:
 - docker
---

Here are some useful tips to use docker. 



## Image

In docker's term, each program to execute is called "***image***". From [dockerhub](https://hub.docker.com/), it is possible to view available images that I need. 

```bash
docker pull httpd # Pulls images named "httpd"
```

To check images in my computer, run:

```bash
docker images # make container and run http image
```



## Container

In docker's term, environment that programs are executed is called "***container***".



### Run 

To make a container that runs image, type:

```bash
docker run http # make container and run http image
```

Running with customized name:

```bash
docker run --name tommy httpd # named "tommy"
```

To use docker container in web server, you need to run:

```bash
docker run --name tommy -p 8080:80 httpd # User access port => [IN]:[OUT] <= container access port
```

To synchronize file system,

```bash
docker run --name tommy -p 8080:80 -v ~/Desktop/repo:~/home/repo httpd # synchronize client and host folder
```



### Run in script: Docker-compose

If a command to run container is lengthy, such as,

```bash
docker \
run \
    #[very long command]
mysql:5.7
```

these commands can be formed into a file called "*docker-compose.yml*".

This file can be run using:

```bash
docker-compose up
```

and be shut down using:

```bash
docker-compose down
```



### Do something in container

Using *exec* command with *-it* option, command can be run iteratively. 

```bash
docker exec -it tommy /bin/bash # in container "tommy", exec bash interface (-i) tty (-t)
```



### Check Process

```
PS \home > docker ps -a
REPOSITORY        TAG                             IMAGE ID       CREATED         SIZE
httpd             latest                          a8ea074f4566   2 weeks ago     144MB
```



### Stop

To stop container, 

```
docker stop tommy
docker stop [containerid]
```



### Restart

To restart container that was stopped,

```
docker start tommy
```

This does not show logs. If you need logs, take:

```bash
docker logs [-f] tommy # showing logs, -f will show logs in live
```



### Remove

To remove containers,

```
docker rm tommy
```



## Customize your container and publish

### Pull base image and customize

As an example, Let's install git in ubuntu image. 

```bash
docker pull ubuntu # pull
docker run -it --name my-ubuntu ubuntu bash # run and bash
apt install git # install git
```



### Commit

To create new image, we can commit this. 

```
docker commit tommy my-ubuntu:my-ubuntu-git 
```



### Customize image using script : Dockerfile

However, customizing may have complicated procedures; That's why we have *Dockerfile*. 

In file named "*Dockerfile*", write something like:

```dockerfile
From ubuntu
RUN apt update && apt install -y git
```



### Build

Using Dockerfile, we can build a new image as follows:

```bash
docker build -t carrtesy:ubuntu-git-in-dockerfile . # [path to Dockerfile]
```



### Push

To make your image public, 

```bash
docker push carrtesy/python3:1.0 # push to [dockerhub account]/[repository name]:[version]
```





## References

### Basics

[korean] 생활코딩 / Docker 입구 수업 / https://www.youtube.com/playlist?list=PLuHgQVnccGMDeMJsGq2O-55Ymtx0IdKWf

### Advanced

[korean] 생활코딩 / 도커 : 이미지 만드는 법 - commit / https://www.youtube.com/watch?v=RMNOQXs-f68&list=RDCMUCvc8kv-i5fvFTJBFAk6n1SA&index=1

[korean] 생활코딩 / 도커 : 이미지 만드는 법 - Dockerfile & build / https://www.youtube.com/watch?v=0kQC19w0gTI&list=RDCMUCvc8kv-i5fvFTJBFAk6n1SA&index=2

[korean] 생활코딩 / Docker hub로 이미지 공유하기 (push) / https://www.youtube.com/watch?v=_38dU6GExDo

[korean] 생활코딩 / Docker compose 를 이용해서 복잡한 도커 컨테이너를 제어하기 /https://www.youtube.com/watch?v=EK6iYRCIjYs
