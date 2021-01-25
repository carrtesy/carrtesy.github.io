---
title: "Problem Solving Basis - DFS, BFS"
date: 2021-01-25
categories:
 - problem solving 

tags:
 - DFS
 - BFS
---

# Intro
Basic of Problem Solving is to find all possible cases.
*DFS(Depth First Search)* and *BFS(Breadth First Search)* are good ways to do so.

# Idea
Suppose that we are trying to find a way out to maze.
To schemes can be summarized as follows:
- DFS: Go deep as far as I can. What if I got wrong? Go back to the recent point of your fork.
- BFS: Investigate every possible ways from your current point.

# Howto
When programming, I usally find following convetions.

## DFS
- check in
- destination?
- iterate
- can I go next?
- visit the point.
- check out

## BFS
- pop the queue
- destination?
- iterate
- can I go next?
- check in & enqueue

# Problems to Solve
- (bkj) basic implementation of BFS/DFS > https://www.acmicpc.net/problem/1260



