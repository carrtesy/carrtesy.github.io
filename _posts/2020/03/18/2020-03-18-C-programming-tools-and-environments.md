---
title: "C programming Tools and Environments"
date: 2020-03-18
categories:
 - programming tools and environments 

tags:
 - c
 - c++
 - gcc
 - gdb
 - vim

---


This post is a combination of useful tools in C, C++ programming

***
Index
### gcc
### gdb
### vim

***

### gcc

{% highlight c %}
#include <stdio.h>
// main.c

int main(void)
{
	printf("Hello World!\n");
}

{% endhighlight %}

- Normal Compile & execute 
```
gcc main.c 
```
```
./a.out
Hello World!
```

- Naming Option
```
gcc -o main main.c
```
```
./main
Hello World!
```

- gdb compile
```
gcc -g -o main main.c
```
```
gdb ./main

```

### gdb



### vim



[infix]: http://www.cs.man.ac.uk/~pjj/cs212/fix.html
[postfix]: http://www.cs.man.ac.uk/~pjj/cs212/fix.html
[geeksforgeeks]: https://www.geeksforgeeks.org/stack-set-2-infix-to-postfix/
[github]: https://github.com/dongminkim0220/Calculator

