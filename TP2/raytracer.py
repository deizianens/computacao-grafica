import sys
import numpy as np
from math import sqrt, pow, pi

# stores 3 values 
class Vec3( object ):
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

    # vector dot product
	def dot(self, number): 
		return self.x*number.x + self.y*number.y + self.z*number.z

    # vector cross product
	def cross(self, number):  
		return (self.y*number.z - self.z*number.y, self.z*number.x - self.x*number.z, self.x*number.y - self.y*number.x)

    # vector magnitude
	def magnitude(self): 
		return sqrt(self.x**2+self.y**2+self.z**2)

    # compute a normalized vector
	def normalize(self): 
		mag = self.magnitude()
		return Vec3(self.x/mag, self.y/mag, self.z/mag)

    # sum vectors
	def __add__(self, number):  
		return Vec3(self.x + number.x, self.y+number.y, self.z+number.z)

    # subtract vectors
	def __sub__(self, number):  
		return Vec3(self.x - number.x, self.y - number.y, self.z - number.z)

    # multiply vector by scalar
	def __mul__(self, number):  
		assert type(number) == float or type(number) == int
		return Vec3(self.x*number, self.y*number, self.z*number)


'''
At the core of a ray tracer is to send rays through pixels and 
compute what color is seen in the direction of those rays.
'''
class Ray:
    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction

    def point_at_parameter(self, t):
        return self.origin + self.direction * t


'''
Spheres, because spheres are cool! (and easy)
'''
class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color


def color(r):
    u_direction = Vec3()
    u_direction = u_direction.normalize(r)


'''
Trace rays 
'''
def trace(ray_orig, ray_dest):
    print("Trace!")


'''
Calculate whether a ray intersects or not an item on scene 
'''
def intersect(ray, shape):
    oc = ray.origin - shape.center
    a = ray.direction.dot(ray.direction)
    b = 2 * oc.dot(ray.direction)
    c = oc.dot(oc) - shape.radius * shape.radius

    # delta
    discriminant = b * b - 4 * a * c 

    if(discriminant >= 0):
        return 1
    else:
        return -1


def main():
    # default image size
    width = 340
    height = 480

    # reads output file name
    if(sys.argv < 1):
        print("Forneça o nome do arquivo de saída!")
    else:
        output = sys.argv[1] 

    # user can optionally specify width and height
    if(sys.argv > 2):
        width = sys.argv[2]
        height = sys.argv[3]



if __name__ == '__main__':
    main()