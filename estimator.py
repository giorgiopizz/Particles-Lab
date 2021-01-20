import time
import random
from math import sin, cos, pi
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import cuda, jit
import numpy as np
from collections import namedtuple

start_time = time.time()



@jit(nopython=True)
def theta_distrib(x):
    return cos(x)**3 * sin(x)



double = np.zeros(1)
triple = np.zeros(1)

for i in range(1000):
    #theta = 0.49087549338874903
    # flux = mu_flux(theta)
    # phi = random.uniform(0, 2*pi)
    # # the flux and the surface are not perpendicular, cosine is needed for calculating the rate of particles
    # N = tempo*flux*S*cos(theta)*sin(theta)
    # massimo numero di cuda cores per rtx 2070 super = 2560 quindi 40*64
    threads_per_block = 64
    blocks = 40

    nIterations = 10000//(threads_per_block*blocks)
