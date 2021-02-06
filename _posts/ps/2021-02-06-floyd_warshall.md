---
title: "Problem Solving Basis - Floyd Warshall"
date: 2021-02-01
categories:
 - problem solving 

tags:
 - floyd warshall
 - path finding
---

In [dijkstra algorithm](https://dongminkim0220.github.io/posts/dijkstra/), we have seen single source shortest path algorithm. How about multiple sources? In other words, what if we want to know every pairs of shortest paths?

# Idea
Suppose that we have V nodes, and want to figure out shortest path of any pair of those nodes, say distance from V<sub>i</sub> to V<sub>j</sub>. 

Then, there can be possibilities as follows:

- directly from V<sub>i</sub> to V<sub>j</sub>
- take a detour. Our path is V<sub>i</sub> to V<sub>k</sub>  and V<sub>k</sub> to V<sub>j</sub>

# How To

Let the detour node be k.

And we are now dealing with the shortest path from V<sub>i</sub> to V<sub>j</sub>.



Firstly, V<sub>1</sub> is our detour node. We compare the path distance: V<sub>i</sub> to V<sub>j</sub> directly and V<sub>i</sub> to V<sub>1</sub>, and V<sub>1</sub> to V<sub>j</sub>

This is our first answer, and will tell the path when V<sub>1</sub> as detour node being considered.



Secondly, V<sub>2</sub> is our detour node. We compare the path distance: minimum distance from the first calculation, and path detouring V<sub>2</sub>.

This is our second answer, and will tell the path when V<sub>1</sub>, V<sub>2</sub> as detour node being considered.



What would be our next step? V<sub>1</sub>, V<sub>2</sub>, ..., V<sub>k</sub> will be considered. 

# Complexity Analysis
Graph takes V vertices.
- Time:  There are $$ |V|^2 $$ pairs, and detouring node |V| times considered, so $$ |V|^3 $$ 
- Space: $$ |V|^2 $$ for adjacency matrix

# Template Code
cpp: <https://github.com/dongminkim0220/Problem-Solvings/blob/master/templates/cpp/floyd_warshall.cpp>

# Problems to Solve
- basics: <https://www.acmicpc.net/problem/11404>