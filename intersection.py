import numpy as np
from math import sin, cos, radians

#
# class rectangle(object):
#     def __init__(self, lenght, width, center, direction):
#         self.lenght = lenght
#         self.width = width
#         self.center = center
#         self.direction = direction


from collections import namedtuple

rectangle = namedtuple('rectangle', ['x_lenght','y_lenght','z_lenght','center','direction'])
ray = namedtuple('ray', ['origin','theta','phi'])
point = namedtuple('point', ['x','y','z'])


def intersection(rectangle, ray):
    if rectangle.direction[0] == 1:
        # parallel to yz
        dir = 0
        x = rectangle.center[dir]
        t = (x-ray.origin[dir])/(sin(ray.theta)*cos(ray.phi))
        y = t * sin(ray.theta)*sin(ray.phi) + ray.origin[1]
        z = t * cos(ray.theta) + ray.origin[2]
        if (y < (rectangle.center[1] + rectangle.y_lenght/2) and \
        y>(rectangle.center[1] + rectangle.y_lenght/2)) and \
        (z < (rectangle.center[2] + rectangle.z_lenght/2) and \
        z>(rectangle.center[2] + rectangle.z_lenght/2)):
            return x, y, z
        else:
            print('not contained')
    elif rectangle.direction[1] == 1:
        # parallel to xz
        dir = 1
        y = rectangle.center[dir]
        t = (y-ray.origin[dir])/(sin(ray.theta)*sin(ray.phi))
        print(t)
        x = t * sin(ray.theta)*cos(ray.phi) + ray.origin[0]
        z = t * cos(ray.theta) + ray.origin[2]
        if (x < (rectangle.center[0] + rectangle.x_lenght/2) and \
        x>(rectangle.center[0] - rectangle.x_lenght/2)) and \
        (z < (rectangle.center[2] + rectangle.z_lenght/2) and \
        z>(rectangle.center[2] - rectangle.z_lenght/2)):
            return x, y, z
        else:
            print('not contained')
    elif rectangle.direction[2] == 1:
        # parallel to xy
        dir = 2
        z = rectangle.center[dir]
        t = (z-ray.origin[dir])/(cos(ray.theta))
        x = t * sin(ray.theta)*cos(ray.phi) + ray.origin[0]
        y = t * sin(ray.theta)*sin(ray.phi) + ray.origin[1]
        if (x < (rectangle.center[0] + rectangle.x_lenght/2) and \
        x>(rectangle.center[0] - rectangle.x_lenght/2)) and \
        (y < (rectangle.center[1] + rectangle.y_lenght/2) and \
        y>(rectangle.center[1] - rectangle.y_lenght/2)):
            return x, y, z
        else:
            print('not contained')
    else:
        print("problem")





# def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
#
# 	ndotu = planeNormal.dot(rayDirection)
# 	if abs(ndotu) < epsilon:
# 		raise RuntimeError("no intersection or line is within plane")
#
# 	w = rayPoint - planePoint
# 	si = -planeNormal.dot(w) / ndotu
# 	Psi = w + si * rayDirection + planePoint
# 	return Psi


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

    r = ray([0,-1,-2],radians(60),radians(90))
    rec = rectangle(2,0,1,[0,3,0.5],[0,1,0])
    print(intersection(rec,r))
