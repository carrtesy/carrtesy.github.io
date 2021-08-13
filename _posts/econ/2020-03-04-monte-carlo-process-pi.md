---
title: "Monte Carlo Process: 𝜋 approximation"

date: 2020-03-04
last_modified_at: 2021-08-13
categories: 
 - monte carlo process
tags:
 - monte carlo process
 - quantecon
 - python
use_math: true
---


The task is to compute an approximation to 𝜋 using [Monte Carlo][monte-carlo].

*Python* is used for representation.

***
**Version 1**: Draw unit circle to the coordinate and plot the dots.

```python
import numpy as np
import matplotlib.pyplot as plt
```


let's setup the initial variables.

```python
trial = 1000
inCircle = 0
```

And draw the unit circle to the plt.

```python
circle = plt.Circle((0, 0), radius = 1, color='black', fill=False)
ax = plt.gca()
ax.add_artist(circle)
```

And then, plot the random coordinate.

```python
for i in range(trial):
	x = np.random.uniform(-1,1)
	y = np.random.uniform(-1,1)
    if (x**2 + y**2 < 1):
        inCircle += 1
        plt.scatter(x, y, s = 1, color = "black")
    else:
        plt.scatter(x, y, s = 1, color = "blue")   
```

Show the canvas and calculate the answer.        

```python
plt.axis([-1, 1, -1, 1])
plt.show()
ratio = inCircle/trial
π = ratio * 4 # result is here
```

The result comes out as:

![img1](/assets/images/post-2020-03-04-1.png)

***
**Version 2**: We want to see the converging process. 

Basic setup is as follows:

```python
import numpy as np
import matplotlib.pyplot as plt
import math
```

Firstly, lets define function that does montecarlo processes to estimate pi.

```python
def monteCarloPi(trial):
	cnt = 0
    for i in range(trial):
	x = np.random.uniform(-1,1)
	y = np.random.uniform(-1,1)
    
	if (x**2 + y**2 < 1):
		cnt = cnt+1

	return (cnt/trial)*4
```

And let's define another function to plot the result. 
if the value is within ϵ, let's plot the point in blue color.
else, let's do it in red. 

```python
def plotApproxProcess(x, ϵ, trial):
	plt.hlines(π, 1, trial, colors='black', linestyles='solid', label='π')
	for i in range(1, trial+1):
		result = monteCarloPi(i)
		if abs(result - x) < ϵ:
			plt.scatter(i, result, s = 1, color = "blue")
		else:
			plt.scatter(i, result, s = 1, color = "red")
```

And show the result. 
```python
π = math.pi
ϵ = 0.1
plotApproxProcess(π, ϵ, 1000)
plt.show()
```


The result comes out as:

![img2](/assets/images/post-2020-03-04-2.png)

***

Code in jupyter notebook is available at my [github][github].


[monte-carlo]: https://en.wikipedia.org/wiki/Monte_Carlo_method
[github]: https://github.com/dongminkim0220/QuantEconProjects/blob/master/monte_carlo.ipynb

