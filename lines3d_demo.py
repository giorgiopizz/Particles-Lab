import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import radians
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = radians(60)
phi = radians(90)
t = np.linspace(-5, 10, 100)
x = t * np.sin(theta) * np.cos(phi) + 0
y = t * np.sin(theta) * np.sin(phi) -1
z = t * np.cos(theta) -2


rects  = [[[1, 3, 0], [1, 3, 1], [-1, 3, 1], [-1, 3, 0]]]
ax.add_collection3d(Poly3DCollection(rects))

vv = np.array(rects)
ax.plot(vv[0][:, 0], vv[0][:, 1], vv[0][:, 2], color='r')
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()
