import numpy as np
from math import sin, cos

#
# class rectangle(object):
#     def __init__(self, lenght, width, center, direction):
#         self.lenght = lenght
#         self.width = width
#         self.center = center
#         self.direction = direction


from collections import namedtuple

rectangle = namedtuple('rectangle', ['lenght','width','center','direction'])
ray = namedtuple('rectangle', ['origin','theta','phi'])

def intersection(rectangle, ray):
    if rectangle.direction[0] != 1:
        # parallel to yz
        dir = 0
        x_0 = rectangle.center[dir]
        t = (x_0-ray.origin[dir])/(sin(ray.theta)*cos(ray.phi))
        y = t * sin(ray.theta)*sin(ray.phi)
        z = t * cos(ray.theta)
    elif rectangle.direction[1] != 1:
        # parallel to xz
        dir = 1
        y_0 = rectangle.center[dir]
        t = (y_0-ray.origin[dir])/(sin(ray.theta)*sin(ray.phi))
        x = t * sin(ray.theta)*cos(ray.phi)
        z = t * cos(ray.theta)
    elif rectangle.direction[1] != 1:
        # parallel to xy
        dir = 2
        z_0 = rectangle.center[dir]
        t = (z_0-ray.origin[dir])/(cos(ray.theta))
        x = t * sin(ray.theta)*cos(ray.phi)
        y = t * sin(ray.theta)*sin(ray.phi)
    else:
        print("problem")

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")

	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi


if __name__=="__main__":
	#Define plane
	planeNormal = np.array([0, 0, 1])
	planePoint = np.array([0, 0, 5]) #Any point on the plane

	#Define ray
	rayDirection = np.array([0, -1, -1])
	rayPoint = np.array([0, 0, 10]) #Any point along the ray

	Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
	print ("intersection at", Psi)
