---
title: "Monte CarloProcess: PI approximation"
date: 2020-03-04
categories: montecarlo
---

# Monte Carlo Process to estimate 𝜋 with visualization

The task is to compute an approximation to $ \pi $ using [Monte Carlo][monte-carlo].

## Version 1
First we draw unit circle, which is $$ x^2 + y^2 = 1 $$ 
to the coordinate and plot the dots.

​```python
import numpy as np
import matplotlib.pyplot as plt

# Initial Setting
# trial is the number of trial
# inCircle is the number of points in the circle
trial = 1000
inCircle = 0

# Draw circle with radius 1 : x^2 + y^2 = 1
circle = plt.Circle((0, 0), radius = 1, color='black', fill=False)
ax = plt.gca()
ax.add_artist(circle)

# Plot with uniform dist
for i in range(trial):
    x = np.random.uniform(-1,1)
    y = np.random.uniform(-1,1)
    
    if (x**2 + y**2 < 1):
        inCircle += 1
        plt.scatter(x, y, s = 1, color = "black")
    else:
        plt.scatter(x, y, s = 1, color = "blue")
            
# Show the canvas and calculate the answer
plt.axis([-1, 1, -1, 1])
plt.show()

ratio = inCircle/trial
π = ratio * 4
print(π)
​```

The result comes out as:
![]

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[monte-carlo]: https://en.wikipedia.org/wiki/Monte_Carlo_method
[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
