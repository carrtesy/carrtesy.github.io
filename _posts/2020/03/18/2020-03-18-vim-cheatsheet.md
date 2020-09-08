---
title: "Vim cheatsheet"
date: 2020-03-18
categories:
 - programming tools and environments 

tags:
 - c
 - c++
 - vim

---

Something that I need for VIM editor

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

"%" or (shift + 5) to go to parenthesis that is matched
![v5](/assets/images/post-2020-03-18-v5.png)
![v6](/assets/images/post-2020-03-18-v6.png)

