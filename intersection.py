import numpy as np
from math import sin, cos, radians
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sys import maxsize
from collections import namedtuple

rectangle = namedtuple('rectangle', ['lenght', 'center','direction'])
ray = namedtuple('ray', ['origin','theta','phi'])
point = namedtuple('point', ['x','y','z'])


def parallel(rectangle, ray):
    x = sin(ray.theta)*cos(ray.phi) - ray.origin[0]
    y = sin(ray.theta)*sin(ray.phi) - ray.origin[1]
    z = cos(ray.theta) - ray.origin[2]
    if np.dot(np.array([x, y, z]), np.array(rectangle.direction))==0:
        # this might be misleading but remember that rectangle direction
        # is perpendicular to the rectangle
        return True
    else:
        return False

def intersection(rectangle, ray):
    # this function finds intersection point and also distance
    # since for our scope it's only useful to return only the distance
    if not parallel(rectangle, ray):
        if rectangle.direction[0] == 1:
            # parallel to yz
            dir = 0
            x = rectangle.center[dir]
            t = (x-ray.origin[dir])/(sin(ray.theta)*cos(ray.phi))
            y = t * sin(ray.theta)*sin(ray.phi) + ray.origin[1]
            z = t * cos(ray.theta) + ray.origin[2]
            if (y < (rectangle.center[1] + rectangle.lenght[1]/2) and \
            y>(rectangle.center[1] - rectangle.lenght[1]/2)) and \
            (z < (rectangle.center[2] + rectangle.lenght[2]/2) and \
            z>(rectangle.center[2] - rectangle.lenght[2]/2)):
                return abs(t), x, y, z
            else:
                print('not contained')
                return maxsize, maxsize, maxsize, maxsize
        elif rectangle.direction[1] == 1:
            # parallel to xz
            dir = 1
            y = rectangle.center[dir]
            t = (y-ray.origin[dir])/(sin(ray.theta)*sin(ray.phi))
            print(t)
            x = t * sin(ray.theta)*cos(ray.phi) + ray.origin[0]
            z = t * cos(ray.theta) + ray.origin[2]
            if (x < (rectangle.center[0] + rectangle.lenght[0]/2) and \
            x>(rectangle.center[0] - rectangle.lenght[0]/2)) and \
            (z < (rectangle.center[2] + rectangle.lenght[2]/2) and \
            z>(rectangle.center[2] - rectangle.lenght[2]/2)):
                return abs(t), x, y, z
            else:
                print('not contained')
                return maxsize, maxsize, maxsize, maxsize
        elif rectangle.direction[2] == 1:
            # parallel to xy
            dir = 2
            z = rectangle.center[dir]
            t = (z-ray.origin[dir])/(cos(ray.theta))
            x = t * sin(ray.theta)*cos(ray.phi) + ray.origin[0]
            y = t * sin(ray.theta)*sin(ray.phi) + ray.origin[1]
            if (x < (rectangle.center[0] + rectangle.lenght[0]/2) and \
            x>(rectangle.center[0] - rectangle.lenght[0]/2)) and \
            (y < (rectangle.center[1] + rectangle.lenght[1]/2) and \
            y>(rectangle.center[1] - rectangle.lenght[1]/2)):
                return abs(t), x, y, z
            else:
                print('not contained')
                return maxsize, maxsize, maxsize, maxsize
        else:
            print("problem")
    else:
        # rectangle and ray are parallel so they cannot intersect
        return maxsize, maxsize, maxsize, maxsize


def intersection_solid(solid, ray):
    # solid is a list of rectangles
    l = []
    for rec in solid:
        l.append(intersection(rec, ray))
    return sorted(l)[0]


def rect_vertices(rect):
    if rect.direction[0] == 1:
        # we work with y and z
        vertices = []
        l = [-1,1]
        for i in l:
            for j in l[::i]:
                vertex = []
                vertex.append(rect.center[0])
                vertex.append(rect.center[1]+i*rect.lenght[1]/2)
                vertex.append(rect.center[2]+j*rect.lenght[2]/2)
                vertices.append(vertex)
        return vertices
    elif rect.direction[1] == 1:
        # we work with x and z
        vertices = []
        l = [-1,1]
        for i in l:
            for j in l[::i]:
                vertex = []
                vertex.append(rect.center[0]+i*rect.lenght[0]/2)
                vertex.append(rect.center[1])
                vertex.append(rect.center[2]+j*rect.lenght[2]/2)
                vertices.append(vertex)
        return vertices
    elif rect.direction[2] == 1:
        # we work with x and y
        vertices = []
        l = [-1,1]
        for i in l:
            for j in l[::i]:
                vertex = []
                vertex.append(rect.center[0]+i*rect.lenght[0]/2)
                vertex.append(rect.center[1]+j*rect.lenght[1]/2)
                vertex.append(rect.center[2])
                vertices.append(vertex)
        return vertices
    else:
        print('error in calculating rectangle vertices')

def plot(rects, lines):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    i = 0
    for line in lines:
        t = np.linspace(-10, 10, 100)
        x = t * np.sin(line.theta) * np.cos(line.phi) + line.origin[0]
        y = t * np.sin(line.theta) * np.sin(line.phi) + line.origin[1]
        z = t * np.cos(line.theta) + line.origin[2]
        i+=1
        ax.plot(x, y, z, label='ray '+str(i))

    for rect in rects:
        rect_v  = [rect_vertices(rect)]
        ax.add_collection3d(Poly3DCollection(rect_v))

        vv = np.array(rect_v)
        ax.plot(vv[0][:, 0], vv[0][:, 1], vv[0][:, 2], color='r')

    ax.legend()

    plt.show()


def parallelepiped(center, lenghts):
    # a parallelepiped is gust a list of six rectangles
    # given it's center and its lenghts, this functions calculates
    # the rectangles
    rectangles = []
    # for the x axis (rectangles perpendicular to x)
    # direction will be [1, 0, 0], center will be the center of the
    # parallelepiped traslated back and forth of a lenght = x_lenght/2

    for i in range(3):
        # i is the direction
        dir = [1 if x==i else 0 for x in range(3)]
        dim = [lenghts[x] if x!=i else 0 for x in range(3)]
        cent = [center[x] if x!=i else center[x]+lenghts[x]/2 for x in range(3)]
        r_pos = rectangle(dim, cent, dir)
        rectangles.append(r_pos)
        dir = [1 if x==i else 0 for x in range(3)]
        dim = [lenghts[x] if x!=i else 0 for x in range(3)]
        cent = [center[x] if x!=i else center[x]-lenghts[x]/2 for x in range(3)]
        r_neg = rectangle(dim, cent, dir)
        rectangles.append(r_neg)
    return rectangles

if __name__=="__main__":
	#Define plane
	# planeNormal = np.array([0, 0, 1])
	# planePoint = np.array([0, 0, 5]) #Any point on the plane
    #
	# #Define ray
	# rayDirection = np.array([0, -1, -1])
	# rayPoint = np.array([0, 0, 10]) #Any point along the ray
    #
	# Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
	# print ("intersection at", Psi)

    # r = ray([0,-1,-2],radians(60),radians(90))
    # rec = rectangle(2,0,1,[0,3,0.5],[0,1,0])
    # print(intersection(rec,r))
    # r = ray([0,0,0],radians(60),radians(0))
    # rec = rectangle([0,8,10],[0,3,0.5],[1,0,0])
    #
    # # print(rect_vertices(rec))
    # print(intersection(rec,r))
    # plot(r, rec)
    line = ray([0,0,0],radians(90),radians(0))
    r = parallelepiped([0,0,0], [2,2,2])
    #r = rectangle([2,2,0],[0,0,0],[0,0,1])
    plot(r, [line])
    print(intersection_solid(r, line))
