---
title: "GCC cheatsheet"
date: 2020-03-18
categories:
 - programming tools and environments
 

tags:
 - c
 - c++
 - gcc
---

Something that I need for GCC

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

- redirection
```
./a.out < input.txt
```

- gdb compile
```
gcc -g -o main main.c
```
```
gdb ./main
```

