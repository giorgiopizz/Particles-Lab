import time
import random
from math import sin, cos, pi, exp
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import cuda, jit
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson

start_time = time.time()

started = False
#last_time = 0
start_time = 0




def factorial(N):
    if N==1 or N==0:
        return 1
    else:
        p = 1
        for i in range(N,1,-1):
            p *= i
        return p


@jit(nopython=True)
def interval(rate, t):
    return exp(-rate*t)*rate

@jit(nopython=True)
def interval_multiple_events(rate, N, t):
    return pow((rate*t),(N-1))*exp(-rate*t)*rate/factorial(N-1)

# interval_multiple_events(100,90,0.011)

@jit(nopython=True)
def rand_Time(rate, final_time):
    time = random.uniform(0,final_time)
    y = random.uniform(0,rate)
    while(y>interval(rate, time)):
        time = random.uniform(0,final_time)
        y = random.uniform(0,rate)
    return time




@cuda.jit
def generation(rng_states, final_time, start_stop):
    thread_id = cuda.grid(1)




    time = xoroshiro128p_uniform_float32(rng_states, thread_id) * T_measure
    y = xoroshiro128p_uniform_float32(rng_states, thread_id) * rate
    while(y>interval(rate, time)):
        time = xoroshiro128p_uniform_float32(rng_states, thread_id) * T_measure
        y = xoroshiro128p_uniform_float32(rng_states, thread_id) * rate
    coin =  xoroshiro128p_uniform_float32(rng_states, thread_id)
    if coin>=0.9:
        # it's a stop
        start_stop = 2
    elif coin>0.1 and coin<0.9:
        # it's not a start nor a stop
        start_stop = 3
    else:
        #it's a start
        start_stop = 1


def matrix_population(start_stop):
    M = np.array([])
    i=0
    while i<len(start_stop):
        if start_stop[i] == 1:
            l = [0]
            for j in range(i+1, len(start_stop)):
                if start_stop[j] == 1:
                    i = j
                    l = [0]
                elif start_stop[j] == 3:
                    l.append(1)
                elif start_stop[j] == 2:
                    l.append(1)
                    row = np.concatenate((np.zeros(i), np.array(l), np.zeros(len(start_stop)-(j+1))))
                    if M.size!=0:
                        M = np.vstack([M,row])
                    else:
                        M = np.array([row])
                    i = j+1
                    break
        i+=1
    return M

@cuda.jit
def matmul(M, times, result):
    """Perform square matrix multiplication of C = A * B
    """
    i = cuda.grid(1)
    if i < result.shape[0]:
        tmp = 0.
        for j in range(M.shape[1]):
            tmp += M[i, j] * times[j]
        result[i] = tmp


#
# muon = namedtuple('muon', ['x0','y0','z0','theta','phi'])
#
# scintillator = namedtuple('scintillator', ['lenght', 'width', 'height', 'x0','y0','z0', 'th'])





# @cuda.jit
# def geometrical_factor(rng_states, z0, double, triple):
#     thread_id = cuda.grid(1)
#
#     # x0 and y0 are generated between -Ls/2 and Ls/2
#     y0 = xoroshiro128p_uniform_float32(rng_states, thread_id) * L_s-L_s/2
#     x0 = xoroshiro128p_uniform_float32(rng_states, thread_id) * l_s-l_s/2
#
#     phi = xoroshiro128p_uniform_float32(rng_states, thread_id) * 2 * pi
#
#     # theta = np.zeros(1)
#     # #rng_states2 = create_xoroshiro128p_states(1, seed=random.uniform(0,10000))
#     # rand_theta[1,1](rng_states, theta)
#     theta = xoroshiro128p_uniform_float32(rng_states, thread_id) * pi/2
#     y = xoroshiro128p_uniform_float32(rng_states, thread_id) * 0.33
#     while(y>theta_distrib(theta)):
#         theta = xoroshiro128p_uniform_float32(rng_states, thread_id) * pi/2
#         y = xoroshiro128p_uniform_float32(rng_states, thread_id) * 0.33
#
#
#     time = xoroshiro128p_uniform_float32(rng_states, thread_id) * T_measure
#     y = xoroshiro128p_uniform_float32(rng_states, thread_id) * rate
#     while(y>theta_distrib(theta)):
#         theta = xoroshiro128p_uniform_float32(rng_states, thread_id) * pi/2
#         y = xoroshiro128p_uniform_float32(rng_states, thread_id) * 0.33
#
#     # mu = {'x0': x0, 'y0': y0, 'z0': z0, 'theta': theta, 'phi': phi}
#     mu = muon(x0,y0,z0,theta,phi)
#
#     # code for geometrical factor:
#     #scint1 is at the top
#     # zona = 2
#     # scint1 = scintillator(0.8,0.3,0.02,0,0,0.11)
#     # scint2 = scintillator(0.8,0.3,0.04,0,0,0.07)
#     # scint3 = scintillator(0.3,0.8,0.04,0,0.25-0.3*zona,0.02)
#
#     scint1 = scintillator(0.8,0.3,0.02,0,0,0.11, 3)
#     scint2 = scintillator(0.8,0.3,0.04,0,0,0.07, 3)
#     scint3 = scintillator(0.8,0.3,0.04,0,0,0.02, 3)
#
#     # #scint1 = {'lenght': 0.8, 'width': 0.3, 'height': 0.02, 'x0': 0, 'y0': 0, 'z0': 0.11}
#     # scint2 = {'lenght': 0.8, 'width': 0.3, 'height': 0.04, 'x0': 0, 'y0': 0, 'z0': 0.07}
#     # scint3 = {'lenght': 0.3, 'width': 0.8, 'height': 0.04, 'x0': 0, 'y0': 0.25, 'z0': 0.02}
#
#
#
#     scint_passed1 = passed(mu, scint1)
#     scint_passed2 = passed(mu, scint2)
#     scint_passed3 = passed(mu, scint3)
#
#     if scint_passed1 and scint_passed3:
#         double[thread_id] = 1
#
#     if scint_passed1 and scint_passed2 and scint_passed3:
#         triple[thread_id] = 1
#
#
# @jit(nopython=True)
# def passed(mu, scint):
#     ingress = False
#     path = 0
#     # discr is the number of discretization of the z variable
#     discr = 2000
#     decrement = scint.height/discr
#
#     # particle starts interacting at the top surface
#     z = scint.z0+scint.height/2
#
#     for i in range(discr):
#         t = (mu.z0-z) / cos(mu.theta)
#         x = mu.x0 + t * sin(mu.theta) * cos(mu.phi)
#         y = mu.y0 + t * sin(mu.theta) * sin(mu.phi)
#
#
#         # there are a few things the particle might do to interact
#         # 1. the particle enters the scintillator at its top surface
#         # 2. the particle enters the scintillator at a different point
#         # 3. the particle is inside and loses its energy
#         # 4. the particle exit the scintillator
#
#         # the second case is taken into account because the first if doesn't require for the particle to be at the top surface
#         # nor a break is called until: the particle has entered or z is below the lower surface
#
#         # first case is first if
#         # third case is second if
#         # fourth case is the break
#         if((abs(x-scint.x0)<scint.width/2) and (abs(y-scint.y0)<scint.lenght/2) and not ingress):
#             ingress = True
#             path += decrement/cos(mu.theta)
#
#         elif((abs(x-scint.x0)<scint.width/2) and (abs(y-scint.y0)<scint.lenght/2) and ingress):
#             path += decrement/cos(mu.theta)
#
#         elif(ingress):
#             break
#
#         z -= decrement
#     # in order to be revealed by out experiment, the muon must release at leat 3 MeV
#     # with a simple approximation, we suppose that most muons are MIP, which release 1 - 2 MeV/(g/cm^2)
#     # assuming 1.032 g/cm^3 as the density of the scintillators
#     # * 100 is m to cm conversion factor
#     released_energy = 1 * 1.032 * path * 100
#     #print(released_energy)
#     if(released_energy > scint.th):
#         return True
#     else:
#         return False


#
# L = 0.80 #m
# l = 0.30 #m
#
# tempo = 600 #secondi (10 minuti)
# #definisco superfice sopra lo scintillatore
# L_s = 5 #m
# l_s = 2 #m
# S = L_s*l_s #area sopra sintillatori m^2
# #coordinate iniziali dei muoni
# z0 = 2 #m
#
# # tot1 = np.zeros(1)
# # tot2 = np.zeros(1)
# # tot3 = np.zeros(1)
#
# double = np.zeros(1)
# triple = np.zeros(1)

# as rate for the muons to be generated we will use the rate of muons that would pass through the upper surface(5m x 2m)
# we use the rate integrated over angles = 1 cm^-2 min^-1, which means 1666.7 muons/second

rate = 1666.7
# T_measure is 100 minutes(~an hour and half)
T_measure = 100 * 60
N = poisson.rvs( rate*T_measure, size = 1)
#
# times = []
# last_time = 0
# for i in range(1,1000):
#     final_time = T_measure - last_time
#     last_time += rand_Time(rate, final_time)
#     if started:
#         delta = last_time - start_time
#         if delta < 11e-6:
#             times.append(delta)
#             started = False
#     else:
#         started = True
#         start_time = last_time
#
# print(N)
# print(times)
# plt.hist(times)
# plt.show()

result = np.zeros(1)


threads_per_block = 64
blocks = 40

nIterations = 3000//(threads_per_block*blocks)
#nIterations = 1
for j in range(round(nIterations)):

    out1 = np.zeros(threads_per_block * blocks)
    out2 = np.zeros(threads_per_block * blocks)
    # out3 = np.zeros(threads_per_block * blocks)

    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=random.uniform(0,10000))

    generation[blocks, threads_per_block](rng_states, out1, out2 )

    M = matrix_population(out2)

    # r = np.zeros(M.shape[0])
    # matmul[blocks, threads_per_block](M, out1, r)
    r = np.dot(M, out1)
    result = np.concatenate((result,r))




# y = xoroshiro128p_uniform_float32(rng_states, thread_id) * 0.33
# while(y>theta_distrib(theta)):
#     theta = xoroshiro128p_uniform_float32(rng_states, thread_id) * pi/2
#     y = xoroshiro128p_uniform_float32(rng_states, thread_id) * 0.33
#
#
#
# for i in range(1000):
#     #theta = 0.49087549338874903
#     # flux = mu_flux(theta)
#     # phi = random.uniform(0, 2*pi)
#     # # the flux and the surface are not perpendicular, cosine is needed for calculating the rate of particles
#     # N = tempo*flux*S*cos(theta)*sin(theta)
#     # massimo numero di cuda cores per rtx 2070 super = 2560 quindi 40*64
#     threads_per_block = 64
#     blocks = 40
#
#     nIterations = 10000//(threads_per_block*blocks)
#     #nIterations = 1
#     for j in range(round(nIterations)):
#
#         out1 = np.zeros(threads_per_block * blocks)
#         out2 = np.zeros(threads_per_block * blocks)
#         # out3 = np.zeros(threads_per_block * blocks)
#
#         rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=random.uniform(0,10000))
#
#         geometrical_factor[blocks, threads_per_block](rng_states, z0, out1, out2)
#         # single_muon[blocks, threads_per_block](rng_states, theta, phi, z0, out1, out2, out3)
#         double = np.concatenate((double,out1))
#         triple = np.concatenate((triple,out2))
#         # tot3 = np.concatenate((tot3,out3))
#
#
# print(double.sum(), triple.sum() ,triple.sum()/double.sum())
#
#
#
# print("--- %s seconds ---" % (time.time() - start_time))
