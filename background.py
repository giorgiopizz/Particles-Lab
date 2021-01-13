import time
import random
from math import sin, cos, pi
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import cuda, jit
import numpy as np
from numba import float32
from numba.experimental import jitclass

start_time = time.time()


# class initialization standard:
# a = "x0, y0, z0, theta, phi"
# for i in a.split(", "):
#    print("self.{} = {}".format(i,i))


# the following code will only be available in future versions of numba which will support class on gpu
spec1 = [
    ('lenght',float32),
    ('width',float32),
    ('height',float32),
    ('x',float32),
    ('y',float32),
    ('z',float32),
]

spec2 = [
    ('x0',float32),
    ('y0',float32),
    ('z0',float32),
    ('theta',float32),
    ('phi',float32),
]

@jitclass(spec1)
class scintillator(object):
    def __init__(self, lenght, width, height, x, y, z):
        # position of scintillator is the position of its center
        self.lenght = lenght
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.z = z

    def top_surface(self):
        return self.z+self.height/2

@jitclass(spec2)
class muon(object):
    def __init__(self, x0, y0, z0, theta, phi):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.theta = theta
        self.phi = phi




@jit(nopython=True)
def mu_flux(x):
	# 70 m^-2 s^-1 sr^-1
	return 70 * cos(x)**2



@jit(nopython=True)
def passed(mu, scint):
    ingress = False
    path = 0
    # discr is the number of discretization of the z variable
    discr = 2000
    decrement = scint.height/discr
    # particle starts interacting at the top surface
    z = scint.top_surface()
    for i in range(discr):
        t = (mu.z0-z) / cos(mu.theta)
        x = mu.x0 + t * sin(mu.theta) * cos(mu.phi)
        y = mu.y0 + t * sin(mu.theta) * sin(mu.phi)


        # there are a few things the particle might do to interact
        # 1. the particle enters the scintillator at its top surface
        # 2. the particle enters the scintillator at a different point
        # 3. the particle is inside and loses its energy
        # 4. the particle exit the scintillator

        # the second case is taken into account because the first if doesn't require for the particle to be at the top surface
        # nor a break is called until: the particle has entered or z is below the lower surface

        # first case is first if
        # third case is second if
        # fourth case is the break
        if((abs(x)<scint.width/2) and (abs(y)<scint.lenght/2) and not ingress):
            ingress = True
            path += decrement/cos(theta)

        elif((abs(x)<scint.width/2) and (abs(y)<scint.lenght/2) and ingress):
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






@cuda.jit
def single_muon(rng_states, theta, phi, z0, s1, s2, s3):
    thread_id = cuda.grid(1)

    # x0 and y0 are generated between -Ls/2 and Ls/2
    x0 = xoroshiro128p_uniform_float32(rng_states, thread_id) * L_s-L_s/2
    y0 = xoroshiro128p_uniform_float32(rng_states, thread_id) * l_s-l_s/2

    mu = muon(x0, y0, z0, theta, phi)

    # normal configuration:
    #scint1 is at the top
    scint1 = scintillator(dimension(0.8,0.3,0.02), position(0,0,0.11))
    scint2 = scintillator(dimension(0.8,0.3,0.04), position(0,0,0.07))
    scint3 = scintillator(dimension(0.8,0.3,0.04), position(0,0,0.02))

    # old code
    # scint_passed1 = passed(theta, phi, x0, y0, z0, scintillatore)
    # scint_passed2 = passed(theta, phi, x0, y0, z0, 2)
    # scint_passed3 = passed(theta, phi, x0, y0, z0, 3)

    scint_passed1 = passed(mu, scint1)
    scint_passed1 = passed(mu, scint2)
    scint_passed1 = passed(mu, scint3)

    if scint_passed1:
        s1[thread_id] = 1
    if scint_passed2:
        s2[thread_id] = 1
    if scint_passed3:
        s3[thread_id] = 1



@cuda.jit
def geometrical_factor(rng_states, theta, phi, z0, double, triple):
    thread_id = cuda.grid(1)

    # x0 and y0 are generated between -Ls/2 and Ls/2
    x0 = xoroshiro128p_uniform_float32(rng_states, thread_id) * L_s-L_s/2
    y0 = xoroshiro128p_uniform_float32(rng_states, thread_id) * l_s-l_s/2

    mu = muon(x0, y0, z0, theta, phi)


    # code for geometrical factor:
    scint1 = scintillator(dimension(0.8,0.3,0.02), position(0, 0, 0.11))
    scint2 = scintillator(dimension(0.8,0.3,0.04), position(0, 0, 0.07))
    scint3 = scintillator(dimension(0.3,0.8,0.04), position(0, 0.25, 0.02))

    scint_passed1 = scint1.passed(mu)
    scint_passed1 = scint1.passed(mu)
    scint_passed1 = scint1.passed(mu)

    if scint_passed1 and scint_passed3:
        double[thread_id] = 1

    if scint_passed1 and scint_passed2 and scint_passed3:
        triple[thread_id] = 1


# old code
# @jit(nopython=True)
# def passed(theta, phi, x0, y0, z0, scint):
#     if scint==1:
#         z = z0-0.13
#         h = 0.02
#     elif scint==2:
#         z = z0-0.09
#         h = 0.04
#     elif scint==3:
#         z = z0-0.04
#         h = 0.04
#     else:
#         print("No correct scintillator provided")
#         return
#     ingress = False
#     path = 0
#     # discr is the number of discretization of the z variable
#     discr = 2000
#     decrement = h/discr
#     for i in range(discr):
#         t = z / cos(theta)
#         x = x0 + t * sin(theta) * cos(phi)
#         y = y0 + t * sin(theta) * sin(phi)
#
#         if((x > 0.85 and x < 1.15) and (y > 2.1 and y < 2.9) and not ingress):
#             ingress = True
#             path += decrement/cos(theta)
#
#         elif((x > 0.85 and x < 1.15) and (y > 2.1 and y < 2.9) and ingress):
#             path += decrement/cos(theta)
#
#         elif(ingress):
#             break
#
#         z -= decrement
# 	# in order to be revealed by out experiment, the muon must release at leat 3 MeV
# 	# with a simple approximation, we suppose that most muons are MIP, which release 1 - 2 MeV/(g/cm^2)
# 	# assuming 1.032 g/cm^3 as the density of the scintillators
# 	# * 100 is m to cm conversion factor
#     released_energy = 1 * 1.032 * path * 100
#     #print(released_energy)
#     if(released_energy > 3):
#         return True
#     else:
#         return False


L = 0.80 #m
l = 0.30 #m

tempo = 600 #secondi (10 minuti)
#definisco superfice sopra lo scintillatore
L_s = 5 #m
l_s = 2 #m
S = L_s*l_s #area sopra sintillatori m^2
#coordinate iniziali dei muoni
z0 = 2 #m

# tot1 = np.zeros(1)
# tot2 = np.zeros(1)
# tot3 = np.zeros(1)

double = np.zeros(1)
triple = np.zeros(1)

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
        # out3 = np.zeros(threads_per_block * blocks)

        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=random.uniform(0,10000))

        geometrical_factor[blocks, threads_per_block](rng_states, theta, phi, z0, out1, out2)
        # single_muon[blocks, threads_per_block](rng_states, theta, phi, z0, out1, out2, out3)
        double = np.concatenate((double,out1))
        triple = np.concatenate((triple,out2))
        # tot3 = np.concatenate((tot3,out3))


print(double.sum(), triple.sum() ,triple.sum()/double.sum())



print("--- %s seconds ---" % (time.time() - start_time))
