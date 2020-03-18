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

Index
- ### gcc
- ### gdb
- ### vim


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

{% highlight c %}
#include <stdio.h>
// gdbmain.c

int main(void)
{
	int times = 10;
	for(int i = 0; i < times; i++)
	{
		printf("line #%d\n", i+1);
	} 
	
}

{% endhighlight %}


Do compile using gdb compile first
```
gcc -g -o main main.c
```

Execute gdb
```
gdb ./main
```

prompt like:
```
GNU gdb (Ubuntu 8.2-0ubuntu1~18.04) 8.2
Copyright (C) 2018 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./gdbmain...done.
(gdb)
```

"l" to see codes
```
(gdb) l
1	#include <stdio.h>
2	
3	int main(void)
4	{
5	  int times = 10;
6	  for(int i = 0; i < times; i++)
7	  {
8	    printf("line #%d", i+1);
9	  }
10	}

```

"b number" to set breakpoint
```
(gdb) b 5
Breakpoint 1 at 0x652: file gdbmain.c, line 5.
```

"r" to run program
```
(gdb) r
Starting program: /home/dongmin/Desktop/Projects/current/dongminkim0220.github.io/_posts/2020/03/18/gdbmain 

Breakpoint 1, main () at gdbmain.c:5
5	  int times = 10;
```

"n" or "s" to proceed
"n" is for doing next line
"s" is to step into function, if any
```
(gdb) n
6	  for(int i = 0; i < times; i++)
```

"p" to see the value of a variable
```
(gdb) p times
$1 = 10
(gdb) p i
$2 = 0
```

"quit" to t(gdb) quit
urn off
```
(gdb) quit
```


### vim

{% highlight c %}
#include <stdio.h>
// main.c

int main(void)
{
	printf("Hello World!\n");
}

{% endhighlight %}

open
```
vim main.c
```

open and point particular line
```
vim +5 main.c
```

Vim modes are:
- normal
- input
- visual
- exit

normal mode: type "Esc"
![v1](/assets/images/post-2020-03-18-v1.png)

input mode: change into normal mode first, and,
type "a" to append
type "i" to insert
![v2](/assets/images/post-2020-03-18-v2.png)

exit mode: change into normal mode first, and, type ":"
":w" to write(save) file
":q" to quit
":wq" to write and quit
![v3](/assets/images/post-2020-03-18-v3.png)

To copy lines, 
change to normal mode by typing "Esc",
and type "yy" to copy single line,
Or to copy multiple lines, for example, 10 lines,  
type "10", and type "yy"
![v4](/assets/images/post-2020-03-18-v4.png)

