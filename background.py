import time
import random
from math import sin, cos, pi
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import cuda, jit
import numpy as np


start_time = time.time()


@cuda.jit
def position(rng_states, theta, phi, z0, s1, s2, s3):
    thread_id = cuda.grid(1)

    x0 = xoroshiro128p_uniform_float32(rng_states, thread_id) * L_s
    y0 = xoroshiro128p_uniform_float32(rng_states, thread_id) * l_s

    scint_passed1 = passed(theta, phi, x0, y0, z0, 1)
    scint_passed2 = passed(theta, phi, x0, y0, z0, 2)
    scint_passed3 = passed(theta, phi, x0, y0, z0, 3)

    if scint_passed1:
        s1[thread_id] = 1
    if scint_passed2:
        s2[thread_id] = 1
    if scint_passed3:
        s3[thread_id] = 1

@jit(nopython=True)
def passed(theta, phi, x0, y0, z0, scint):
    if scint==1:
        z = z0-0.13
        h = 0.02
    elif scint==2:
        z = z0-0.09
        h = 0.04
    elif scint==3:
        z = z0-0.04
        h = 0.04
    else:
        print("No correct scintillator provided")
        return
    ingress = False
    path = 0
    # discr is the number of discretization of the z variable
    discr = 2000
    decrement = h/discr
    for i in range(discr):
        t = z / cos(theta)
        x = x0 + t * sin(theta) * cos(phi)
        y = y0 + t * sin(theta) * sin(phi)

        if((x > 0.85 and x < 1.15) and (y > 2.1 and y < 2.9) and not ingress):
            ingress = True
            path += decrement/cos(theta)

        elif((x > 0.85 and x < 1.15) and (y > 2.1 and y < 2.9) and ingress):
            path += decrement/cos(theta)

        elif(ingress):
            break

        z -= decrement
	# in order to be revealed by out experiment, the muon must release at leat 3 MeV
	# with a simple approximation, we suppose that most muons are MIP, which release 1 - 2 MeV/(g/cm^2)
	# assuming 1.032 g/cm^3 as the density of the scintillators
	# * 100 is m to cm conversion factor
    released_energy = 1 * 1.032 * path * 100
    #print(released_energy)
    if(released_energy > 3):
        return True
    else:
        return False


L = 0.80 #m
l = 0.30 #m
A = L*l #area sintillatori m^2
tempo = 600 #secondi (10 minuti)
#definisco superfice sopra lo scintillatore
L_s = 5 #m
l_s = 2 #m
S = L_s*l_s #area sopra sintillatori m^2
#coordinate iniziali dei muoni
z0 = 2 #m

results = []
tot1 = np.zeros(1)
tot2 = np.zeros(1)
tot3 = np.zeros(1)

for i in range(100):
    theta = random.uniform(0, pi/2)
    #theta = 0.49087549338874903
    flux = mu_flux(theta)
    phi = random.uniform(0, 2*pi)
    # the flux and the surface are not perpendicular, cosine is needed for calculating the rate of particles
    N = tempo*flux*S*cos(theta)*sin(theta)
    # massimo numero di cuda cores per rtx 2070 super = 2560 quindi 40*64
    threads_per_block = 64
    blocks = 40

    nIterations = N//(threads_per_block*blocks)
    for j in range(round(nIterations)):

        out1 = np.zeros(threads_per_block * blocks)
        out2 = np.zeros(threads_per_block * blocks)
        out3 = np.zeros(threads_per_block * blocks)

        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=random.uniform(0,10000))

        position[blocks, threads_per_block](rng_states, theta, phi, z0, out1, out2, out3)

        tot1 = np.concatenate((tot1,out1))
        tot2 = np.concatenate((tot2,out2))
        tot3 = np.concatenate((tot3,out3))


print(tot1.sum(), tot2.sum() , tot3.sum())



print("--- %s seconds ---" % (time.time() - start_time))
