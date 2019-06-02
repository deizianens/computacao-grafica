import sys
import numpy as np
import random
import math 
import numbers
import multiprocessing
from collections import namedtuple

# stores 3 values
class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


    def __getitem__(self, i):
        if i == 0:
            return self.x

        if i == 1:
            return self.y

        if i == 2:
            return self.z

    # u + v
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    # u - v
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    # u * v
    def __mul__(self, other):
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    # u / k or u / v
    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return Vec3(self.x / other, self.y / other, self.z / other)

        return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    # k * u
    def __rmul__(self, other):
        return Vec3(other * self.x, other * self.y, other * self.z)

    # length of u
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    # N.B. Can't use __len__ since we need to return a floating-point value

    # squared length of u
    def squared_length(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    # dot product
    # ref: https://en.wikipedia.org/wiki/Dot_product
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    # cross product
    # ref: https://en.wikipedia.org/wiki/Cross_product
    def cross(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x

        return Vec3(x, y, z)

    # unit vector
    def unit(self):
        return self / self.length()

    # positive vec3
    def __pos__(self):
        return self

    # negative vec3
    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)


    def __repr__(self):
        return '%s(%.3f, %.3f, %.3f)' % (__class__.__name__, self.x, self.y, self.z)


'''
At the core of a ray tracer is to send rays through pixels and 
compute what color is seen in the direction of those rays.
'''
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def __call__(self, t):
        return self.origin + t * self.direction


class Hitable:
    def hit(self, ray, t_min, t_max, hit_info):
        raise NotImplementedError


class HitableList(list, Hitable):
    def hit(self, ray, t_min, t_max):
        hit_anything = False
        closest_so_far = t_max

        for obj in self:
            hit_info = obj.hit(ray, t_min, closest_so_far)
            if hit_info:
                hit_anything = True
                closest_so_far = hit_info.t
                result_info = hit_info

        if hit_anything:
            return result_info

        return None


'''
Spheres, because spheres are cool! (and easy)
'''
class Sphere(Hitable):
    def __init__(self, center, radius, material=None):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, t_min, t_max):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2 * ray.direction.dot(oc)
        c = oc.dot(oc) - self.radius*self.radius
        discriminant = b*b - 4*a*c

        if discriminant > 0:
            temp = (-b - math.sqrt(discriminant)) / (2*a)
            if t_min < temp < t_max:
                t = temp
                p = ray(t)
                normal = (p - self.center) / self.radius

                return HitInfo(t, p, normal, self.material)

            temp = (-b + math.sqrt(discriminant)) / (2*a)
            if t_min < temp < t_max:
                t = temp
                p = ray(t)
                normal = (p - self.center) / self.radius

                return HitInfo(t, p, normal, self.material)

        return None


'''
Camera classes
'''
class Camera:
    def __init__(self):
        self._lower_left_corner = Vec3(-2, -1, -1)
        self._horizontal = Vec3(x=4)
        self._vertical = Vec3(y=2)
        self._origin = Vec3()

    def ray(self, u, v):
        # 0 <= u, v <= 1
        return Ray(self._origin, self._lower_left_corner + u*self._horizontal + v*self._vertical)


class FOVCamera:
    def __init__(self, vfov, aspect):
        # vfov is top to bottom in degrees
        theta = math.radians(vfov)
        half_height = math.tan(theta / 2)
        half_width = aspect * half_height

        self._lower_left_corner = Vec3(-half_width, -half_height, -1)
        self._horizontal = Vec3(x=2 * half_width)
        self._vertical = Vec3(y=2 * half_height)
        self._origin = Vec3()

    def ray(self, u, v):
        return Ray(self._origin, self._lower_left_corner + u*self._horizontal + v*self._vertical)


class PositionalCamera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect):
        theta = math.radians(vfov)
        half_height = math.tan(theta / 2)
        half_width = aspect * half_height

        w = (lookfrom - lookat).unit()
        u = vup.cross(w).unit()
        v = w.cross(u)

        self._lower_left_corner = Vec3(-half_width, -half_height, -1)
        self._lower_left_corner = lookfrom - half_width*u - half_height*v - w
        self._horizontal = 2 * half_width * u
        self._vertical = 2 * half_height * v
        self._origin = lookfrom

    def ray(self, u, v):
        return Ray(self._origin, self._lower_left_corner + u*self._horizontal + v*self._vertical - self._origin)


'''
Material Classes
'''
class Material:
    def scatter(self, ray, hit_info):
        raise NotImplementedError


class Lambertian(Material):
    def __init__(self, albedo):
        super().__init__()
        self.albedo = albedo

    def scatter(self, ray, hit_info):
        target = hit_info.p + hit_info.normal + random_in_unit_radius_sphere()
        scattered = Ray(hit_info.p, target - hit_info.p)
        attenuation = self.albedo

        return ScatterInfo(scattered, attenuation)


class Metal(Material):
    def __init__(self, albedo, fuzz=0):
        super().__init__()
        self.albedo = albedo

        # fuzz is the fuzziness or perturbation parameter
        # it determines the fuzziness of the reflections
        self.fuzz = min(1, max(0, fuzz)) # ensures 0 <= self.fuzz <= 1

    def scatter(self, ray, hit_info):
        # Idea: Make ray.direction a computed property that returns the
        # unit direction vector when first accessed
        # maybe call it unit_direction and leave direction unchanged
        reflected = reflect(ray.direction.unit(), hit_info.normal)
        scattered = Ray(hit_info.p, reflected + self.fuzz*random_in_unit_radius_sphere())
        attenuation = self.albedo

        if scattered.direction.dot(hit_info.normal) > 0:
            return ScatterInfo(scattered, attenuation)

        return None


class Dielectric(Material):
    def __init__(self, refractive_index):
        super().__init__()
        self.refractive_index = refractive_index

    def scatter(self, ray, hit_info):
        reflected = reflect(ray.direction, hit_info.normal)
        attenuation = Vec3(1, 1, 1)

        if ray.direction.dot(hit_info.normal) > 0:
            outward_normal = -hit_info.normal
            ni_over_nt = self.refractive_index
            cosine = self.refractive_index * ray.direction.dot(hit_info.normal) / ray.direction.length()
        else:
            outward_normal = hit_info.normal
            ni_over_nt = 1 / self.refractive_index
            cosine = -ray.direction.dot(hit_info.normal) / ray.direction.length()

        refracted = refract(ray.direction, outward_normal, ni_over_nt)
        if refracted:
            reflect_prob = schlick(cosine, self.refractive_index)
        else:
            reflect_prob = 1

        if random.random() < reflect_prob:
            return ScatterInfo(Ray(hit_info.p, reflected), attenuation)

        return ScatterInfo(Ray(hit_info.p, refracted), attenuation)

'''
Methods
'''
def random_in_unit_disc():
    while True:
        p = 2 * Vec3(random.random(), random.random()) - Vec3(1, 1)
        if p.dot(p) < 1:
            return p


def random_in_unit_radius_sphere():
    one = Vec3(1, 1, 1)

    while True:
        s = 2*Vec3(random.random(), random.random(), random.random()) - one
        if s.squared_length() < 1:
            return s


def reflect(v, n):
    return v - 2*v.dot(n)*n


def refract(v, n, ni_over_nt):
    uv = v.unit()
    dt = uv.dot(n)
    discriminant = 1 - ni_over_nt*ni_over_nt*(1 - dt*dt)
    if discriminant > 0:
        return ni_over_nt*(uv - dt*n) - math.sqrt(discriminant)*n

    return None


def schlick(cosine, refractive_index):
    r0 = (1 - refractive_index) / (1 + refractive_index)
    r0 *= r0
    return r0 + (1 - r0) * pow(1 - cosine, 5)


'''
Calculate whether a ray intersects or not an item on scene 
'''
def intersect(center, radius, ray):
    oc = ray.origin - center
    a = ray.direction.dot(ray.direction)
    b = 2 * oc.dot(ray.direction)
    c = oc.dot(oc) - (radius * radius)

    # delta
    discriminant = b * b - 4 * a * c 

    if(discriminant < 0):
        return -1
    else:
        # bhaskara!
        return(-b - math.sqrt(discriminant))/ (2.0*a)


def color(ray, world, depth):
    hit_info = world.hit(ray, 0.001, sys.float_info.max)

    if hit_info:
        zero = Vec3()

        if depth < 50:
            scatter_info = hit_info.material.scatter(ray, hit_info)
            if scatter_info:
                return scatter_info.attenuation * color(scatter_info.scattered, world, depth+1)
            else:
                return zero
        else:
            return zero

    unit_direction = ray.direction.unit()
    t = 0.5 * (unit_direction.y + 1)
    white = Vec3(1, 1, 1)
    blue = Vec3(0.5, 0.7, 1)

    return (1-t)*white + t*blue


ScatterInfo = namedtuple('ScatterInfo', ['scattered', 'attenuation'])
HitInfo = namedtuple('HitInfo', ['t', 'p', 'normal', 'material'])


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
        width = int(sys.argv[2])
        height = int(sys.argv[3])

    R = math.cos(math.pi / 4)

    # world = HitableList([
    #     Sphere(Vec3(0, 0, -1), 0.5, Lambertian(Vec3(0.1, 0.2, 0.5))),
    #     Sphere(Vec3(0, -100.5, -1), 100, Lambertian(Vec3(0.8, 0.8, 0))),
    #     Sphere(Vec3(1, 0, -1), 0.5, Metal(Vec3(0.8, 0.6, 0.2))),
    #     Sphere(Vec3(-1, 0, -1), 0.5, Dielectric(1.5)),
    #     Sphere(Vec3(-1, 0, -1), -0.45, Dielectric(1.5))
    # ])

    world = HitableList([
        Sphere(Vec3(1, 0, -1), 0.5, Metal(Vec3(0.8, 0.6, 0.2))),
    ])

    cam = PositionalCamera(Vec3(-2, 2, 1), Vec3(0, 0, -1), Vec3(0, 1, 0), 90, width / height)
    ns = height


    # Save the PPM image as a binary file
    with open(output, 'w') as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for j in range(height-1, -1, -1):
            for i in range(0, width):
                col = Vec3(0, 0, 0)
                for s in range(0, ns):
                    u = (i + random.random()) / width
                    v = (j + random.random()) / height
                    r = cam.ray(u, v)
                    col += color(r, world, 0)

                col /= ns
                col = Vec3(math.sqrt(col.x), math.sqrt(col.y), math.sqrt(col.z))
                # print(col.__str__())
                index = 3 * (j * width + i)
                ir  = int(255.99 * col.x)  # red channel
                ig  = int(255.99 * col.y)  # green channel
                ib  = int(255.99 * col.z)  # blue channel
                f.write(f"{ir} {ig} {ib}\n")

if __name__ == '__main__':
    main()
