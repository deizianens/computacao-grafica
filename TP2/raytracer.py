import sys
import numpy as np
import array
from math import sqrt, pow, pi

# stores 3 values
class Vec3(object):
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

    @staticmethod
    def normalize(self):
        # compute a normalized vector
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

    def __str__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ',' + str(self.z) + ')'


'''
At the core of a ray tracer is to send rays through pixels and 
compute what color is seen in the direction of those rays.
'''
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

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


'''
Trace rays 
'''
def trace(ray_orig, ray_dest):
    print("Trace!")


'''
Calculate whether a ray intersects or not an item on scene 
'''
def intersect(center, radius, ray):
    oc = ray.origin - center
    a = ray.direction.dot(ray.direction)
    b = 2 * oc.dot(ray.direction)
    c = oc.dot(oc) - radius * radius

    # delta
    discriminant = b * b - 4 * a * c 

    return(discriminant > 0)


def color(r):
    if(intersect(Vec3(0,0,-1), 0.5, r)):
        return Vec3(1, 0, 0)
    u_direction = Vec3.normalize(r.direction)
    t = 0.1*(u_direction.y + 1.0)
    # print(u_direction.__str__())
    return Vec3(1.0, 1.0, 1.0).__mul__((1.0*t)).__add__(Vec3(0.5, 0.7, 1.0).__mul__(t))


def main():
    # default image size
    width = 480
    height = 340
    output = None

    # reads output file name
    if(len(sys.argv) < 2):
        print("Forneça o nome do arquivo de saída!")
        sys.exit()
    else:
        output = sys.argv[1] 

    # user can optionally specify width and height
    if(len(sys.argv) > 2):
        width = sys.argv[2]
        height = sys.argv[3]

    ppm_header = "P6 "+ str(width) +" "+ str(height) +" 255\n"
    image = array.array('B', [0, 0, 0] * width * height)

    lower_left_corner = Vec3(-2.0, -1.0, -1.0)
    horizontal = Vec3(4.0, 0.0, 0.0)
    vertical = Vec3(0.0, 2.0, 0.0)
    origin = Vec3(.0, .0, .0)

    for j in range(height-1, -1, -1):
        for i in range(0, width):
            u = i/width
            v = j/height
            r = Ray(origin, lower_left_corner.__add__(horizontal.__mul__(u).__add__(vertical.__mul__(v))))
            col = Vec3(color(r).x, color(r).y, color(r).z)
            index = 3 * (j * width + i)
            image[index]        = int(255*col.x)  # red channel
            image[index + 1]    = int(255*col.y)  # green channel
            image[index + 2]    = int(255*col.z)  # blue channel
          


    # Save the PPM image as a binary file
    with open(output, 'wb') as f:
        f.write(bytearray(ppm_header, 'ascii'))
        image.tofile(f)


if __name__ == '__main__':
    main()
