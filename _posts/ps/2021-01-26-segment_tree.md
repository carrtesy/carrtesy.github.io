---
title: "Problem Solving Basis - Segment Tree"
date: 2021-01-26
categories:
 - problem solving 

tags:
 - Segment Tree
---

How can we process aggregated query, in more efficient way?

# Idea
Suppose we have N data sequentially, and users frequently request the sum of data from *i*th data to *j*th data.
It is really nagging to sum all those up every time. Here's the idea: how about **caching** the sub-answers?

Segment Tree is one way to process such aggregated query in more efficient manner.
Segment Tree not only stores the data itself, but also some partial answers that are frequently referred.
Let's take an example.

We have dataset of 5 elements;
```
1 2 3 4 5
```

Then, we may set up a tree of:
```
            15
        10      5
      3   7   5   0
data 1 2 3 4 5 0 0 0
```

# HowTo
## Initialization
Suppose *N* data is available.

Segment Tree takes the form of binary tree.
Then those N data will be located at the leaf position.

So what we do first is to figure out how many leaf nodes will be.
Here, we need to find *S* such that:
$$ S = 2^k >= N $$
For example above when N = 5, 
we may need *k* to be 3, and *S* to be 8.

In this scheme, we may have *S-1* non-leaf nodes.

So if we allocate 2*S space, and index of 1 as root node, it is possible to set up our Segment Tree.

## Update
Suppose that we are trying to update 3rd element(which is 3 below), to 100.
```
       15
   10      5
 3   7   5   0
1 2 3 4 5 0 0 0
```

Firstly, we update the data:
```
        15
   10        5
 3    7    5   0
1 2 *100* 4 5 0 0 0
```

And also, parents too.
```
        15                       15                   *112*
   10        5             *107*     5            107        5

 3  *104*  5   0         3   104   5   0         3  104    5   0

1 2 100 4 5 0 0 0       1 2 100 4 5 0 0 0       1 2 100 4 5 0 0 0
```

## Query
How about our query?
Let's tag the tree first, by their role.

```
                  [1-8]
                   15
       [1-4]                [5-8]
         10                   5
   [1-2]    [3-4]      [5-6]    [7-8]
     3        7          5        0
[1-1][2-2][3-3][4-4][5-5][6-6][7-7][8-8]
  1    2    3    4    5    0    0    0

```
Suppose we want sum from 2nd element to 5th element.
Then we may send the query of (2,5) as follows.


```
                  (2, 5)
                  [1-8]
                   15
       [1-4]                [5-8]
         10                   5
   [1-2]    [3-4]      [5-6]    [7-8]
     3        7          5        0
[1-1][2-2][3-3][4-4][5-5][6-6][7-7][8-8]
  1    2    3    4    5    0    0    0

```

Since query [1-8] is not adequate, we need to split the query.
(2, 5) into (2, 4) and (5, 5). Then:


```
                  (2, 5)
                  [1-8]
                   15
       (2, 4)               (5, 5)
       [1-4]                [5-8]
         10                   5
   [1-2]    [3-4]      [5-6]    [7-8]
     3        7          5        0
[1-1][2-2][3-3][4-4][5-5][6-6][7-7][8-8]
  1    2    3    4    5    0    0    0

```

Likewise, we may end up doing:


```
                  (2, 5)
                  [1-8]
                   15
       (2, 4)               (5, 5)
       [1-4]                [5-8]
         10                   5
   (2, 2)  *(3, 4)     (5, 5)
   [1-2]    [3-4]      [5-6]    [7-8]
     3        7          5        0
    *(2,2)         *(5,5)
[1-1][2-2][3-3][4-4][5-5][6-6][7-7][8-8]
  1    2    3    4    5    0    0    0

```

# Complexity Analysis
- Time: O(logn) for queries, O(logn) for updates
- Space: O(2*S) or O(N)

# Template Code
cpp: https://github.com/dongminkim0220/Problem-Solvings/blob/master/templates/cpp/segment_tree.cpp

# Problems to Solve
- basics: https://www.acmicpc.net/problem/2042



