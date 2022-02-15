---
title: "N for Loops"
date: 2021-10-03
last_modified_at: 2021-10-03

categories:
 - Problem Solving

tags:
 - Brute Force
 - Recursion 
use_math: true
---



How can we set up n for loops for the problem solvings?



## For loops forever

As There are some cases that we need multiple for loops. For example, listing out the cases of picking m elements out of n elements. If m is small like 2, for loop would be the best solution.



What if 3? 4? Problem gets complicated. 

```cpp
for(int i = 0; i < n ; i++){
    for(int j = i; j < n; j++){
        for(int k = j; k < n; k++){
            // and so on...   
        }
    }
}
```



## Recursive approach

For this kind of problem, recursion would be the solution. That is, we are generating for loops in the process of for loops, until we get the result that we want.



## Implementation

So how can we do this?

```cpp
// pick m elements from n
void nCm(int n, vector<int> v, int m){
	
	if(m == 0) { printALL(v); return;}
	
	int start_point = v.empty() ? 0: v.back() + 1;
	
	for(int i = start_point; i < n ; i++){
		v.push_back(i);
		nCm(n, v, m-1);
		v.pop_back();
	}
}
```



In this code, *m* stands for the number that we should pick after. 

*start_point*, as its name suggests, indicates the starting index of the loop. 

At every loop, index is pushed into the memory (vector v), and get into another for loop. 

first line of this function states exit condition: if m elements are picked, print everything and return. 

 

## Finally

This recursion strategy is the best for implementing Brute-force algorithms.

Brute force is simplest, and powerful algorithm that utilizes computer's ability: iteration without exhaustion. 



---

This is the 1st draft written on Oct 3, 2021

1st draft: 2021-10-03
