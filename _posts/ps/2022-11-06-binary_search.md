---
title: "Problem Solving Basis - Binary Search"
date: 2022-11-06
last_modified_at: 2022-11-06
categories:
 - problem solving 

tags:
 - Binary Search
 - Search
---

Fast way to find the element in sorted data: binary search

# Template Code (Python)
## Recursive
```python
length, target = 10, 7
arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# recursive way
def binary_search(arr, s, e, target):
  if s > e:
    return -1
  mid = (s + e) // 2
  v = arr[mid]
  if v < target:
    return binary_search(arr, mid+1, e, target)
  elif v > target:
    return binary_search(arr, s, mid-1, target)
  else:
    return mid

print(binary_search(arr, 0, length - 1, target))
```

## Iterative
```python
length, target = 10, 7
arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# iterative way
s, e, idx = 0, length-1, -1
while s <= e:
  m = (s + e) // 2
  v = arr[m]
  if v < target:
    s = m + 1
  elif v > target:
    e = m - 1
  else:
    idx = m
    break
print(idx)
```



# Complexity Analysis
- Time: $O(logN)$

# link
python: https://github.com/carrtesy/CodingTest_python/blob/main/binary_search/binary_search_template.py





