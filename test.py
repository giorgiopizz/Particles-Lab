import time
import random
from math import sin, cos, pi
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import cuda, jit
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

@cuda.jit
def rand_theta(rng_states, theta):
    thread_id = cuda.grid(1)
    x = xoroshiro128p_uniform_float32(rng_states, thread_id) * pi/2
    y = xoroshiro128p_uniform_float32(rng_states, thread_id) * 0.33
    while(y>theta_distrib(x)):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id) * pi/2
        y = xoroshiro128p_uniform_float32(rng_states, thread_id) * 0.33
        print('p')
    theta[thread_id] = x



@jit(nopython=True)
def theta_distrib(x):
    return cos(x)**3 * sin(x)

# @jit(nopython=True)
# def rand_theta():
#     x = random.uniform(0,pi/2)
#     # max value of y doesn't compromise the distribution as long as it
#     # is above distrib. max. A choice of low max_y lowers comutation time
#     y = random.uniform(0,0.33)
#     while(y>theta_distrib(x)):
#         x = random.uniform(0,pi/2)
#         y = random.uniform(0,0.33)
#
#     return x
#
# x = []
# for i in range(10000):
#
#     x.append(rand_theta())


threads_per_block = 64
blocks = 40
theta = np.zeros(threads_per_block * blocks)
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=random.uniform(0,10000))
rand_theta[blocks, threads_per_block](rng_states, theta)


plt.hist(theta)
plt.show()
