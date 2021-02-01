---
title: "Problem Solving Basis - Disjoint Set"
date: 2021-02-01
categories:
 - problem solving 

tags:
 - disjoint set
 - union find
---

How can we implement set without intersection?

# Idea
Suppose we have N elements: from 1 ~ N.

We are now trying to implement set without intersection, called "**disjoint set**".

Disjoint set does not have an intersection, so we can now pick up a representative of each set.
For this analogy, each set's representative is, let me call the '**ancestor**' of the set.

# HowTo
We are now trying to implement 2 operations.

- find(x): find the set of x
- Union(x, y): make x, y to be in the same set

## Initialization
At Initialization stage, each element is the member of each set. 

In other words, each element by itself is ancestor. So we have:
 
```
          1 2 3 4 5 6 7 8
ancestor |1 2 3 4 5 6 7 8| 
```

## Union
By Unioning the set, element x and y are being in the same set.

In other words, ancestor of x becomes ancestor of y.

Find() function haven't been introduced yet, but suppose this function can find the ancestor of each set.

then:

```cpp
x_ancestor = find(x);
y_ancestor = find(y);
ancestor[y] = x_ancestor;
```

will do the trick.

For example,
Union (1, 2) would do:

```
          1 2 3 4 5 6 7 8
ancestor |1 1 3 4 5 6 7 8| 
```

## Find
To find the ancestor of each, we need to go upstream.
Suppose our array is as follows:

```
          1 2 3 4 5 6 7 8
ancestor |1 1 3 2 5 6 7 4| 
```


and Let's do 2 examples. 

- find(1): Nothing matters, as 1's ancestor is 1.
- find(8): 8's ancestor is 4. So we call find(4). 4's ancestor is 2, so we call find(2), and likewise, find(1), and returns 1.

So in our code:
```cpp
if(ancestor[x] == x){
    return x;
}
return find(ancestor[x]);

```

But we can do more of tricks.
If we have to go upstream everytime, it is more like storing **parents** value, rather then **ancestor**, which takes a lot of time.

Therefore, let's update our ancestor as we find them.


So in our final code:
```cpp
if(ancestor[x] == x){
    return x;
}
return ancestor[x] = find(ancestor[x]);

```

# Complexity Analysis
- Time: Treat this problem as lineage tree, so tree traversal takes O(logn). But find(x) is defined recursively, so it is known to be O(a(N)). (ackermann function, almost constant complexity)
- Space: O(N) for array

# Template Code
cpp: <https://github.com/dongminkim0220/Problem-Solvings/blob/master/templates/cpp/union_find.cpp>

# Problems to Solve
- basics: <https://www.acmicpc.net/problem/1717>



