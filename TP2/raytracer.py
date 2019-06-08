import sys
import time
import numpy as np
import random
import math 
import numbers
import multiprocessing
import struct
from collections import namedtuple

# Obj File loader
# class FileLoader:
#     def __init__(self, path, material=0):
        # with open(path, 'r') as f:
        #     triangle = -1
        #     vertices = Vec3()
        #     for line in f.readlines():
        #         e = line.split()
        #         if(len(e)):
        #             if(triangle != -1):
		# 				vertices[triangle] = Vec3(float(e[1]), float(e[2]), float(e[3]))
		# 				triangle = triangle - 1
                    
        #             if(e[0] == "outer"):
		# 				triangle = 2
					
        #             if(e[0] == "endloop"):
		# 				list_.add(Triangle(vertices.z,vertices.y,vertices.x,material))
					
        #             print(list.size() + " triangles loaded.")



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

    # sum vectors
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    # subtract vectors
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    # multiply vectors
    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return Vec3(self.x * other, self.y * other, self.z * other)

        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    # divide vector by scalar or other vector
    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return Vec3(self.x / other, self.y / other, self.z / other)

        return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    # k * u
    def __rmul__(self, other):
        return Vec3(other * self.x, other * self.y, other * self.z)

    # vector magnitude
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    # squared length of vec
    def squared_length(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    # dot product
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    # cross product
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

    def set_(self, v):
        self.x = v.x
        self.y = v.y
        self.z = v.z

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
        self.time = 0

    def __call__(self, t):
        return self.origin + t * self.direction

    def point_at_parameter(self, t):
        return self.origin - (self.direction*t)


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
Shapes

Spheres, because spheres are cool! (and easy)
'''
class Sphere(Hitable):
    def __init__(self, center, radius, material=None):
        self.center = center
        self.radius = radius
        self.material = material

    '''
    Calculate whether a ray intersects or not an item on scene 
    '''
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


class MovingSphere(Hitable):
    def __init__(self, center0, center1, time0, time1, radius, material=None):
        self.center0 = center0
        self.center1 = center1
        self.time0 = time0
        self.time1 = time1
        self.radius = radius
        self.material = material

    '''
    Calculate whether a ray intersects or not an item on scene 
    '''
    def hit(self, ray, t_min, t_max):
        oc = ray.origin - self.center(ray.time)
        a = ray.direction.dot(ray.direction)
        b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius*self.radius
        discriminant = b*b - a*c

        if discriminant > 0:
            temp = (-b - math.sqrt(discriminant)) / (a)
            if t_min < temp < t_max:
                t = temp
                p = ray(t)
                normal = (p - self.center(ray.time)) / self.radius

                return HitInfo(t, p, normal, self.material)

            temp = (-b + math.sqrt(discriminant)) / (a)
            if t_min < temp < t_max:
                t = temp
                p = ray(t)
                normal = (p - self.center(ray.time)) / self.radius

                return HitInfo(t, p, normal, self.material)

        return None
    
    def center(self, time):
        return self.center0 + ((self.center1 - self.center0) * ((time-self.time0)/(self.time1 - time)))


class Triangle(Hitable):
    def __init__(self, v1, v2, v3, material=None):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.material = material  

    '''
    Calculate whether a ray intersects or not an item on scene 
    '''
    def hit(self, ray, t_min, t_max):
        # triangle normal
        tri_n = ((self.v1 - self.v2).cross(self.v2-self.v3)).unit()

        n_dot_dir = (tri_n.dot(ray.direction))

        if(n_dot_dir > 0):
            return None

        # ray parallel to triangle can't hit
        if(abs(n_dot_dir) < 0.00001):
            return None

        # find the ray plane intersection first
        d = -(tri_n.dot(self.v1))
        t = -(tri_n.dot(ray.origin) + d)/n_dot_dir

        if(t < t_min or t > t_max):
            return None

        p = ray.__call__(t)

        s1 = self.v2 - self.v1 # get first side
        v_to_p = p - self.v1 # get vector from one vertex to intersection point

        # if the cross product of the side and line to point isn't in the same 
        # direction as the normal, we are outside the triangle
        if(s1.cross(v_to_p).dot(tri_n) < 0):
            return None

        s2 = self.v3 - self.v2 
        v_to_p = p - self.v2 

        if(s2.cross(v_to_p).dot(tri_n) < 0):
            return None

        s3 = self.v1 - self.v3 
        v_to_p = p - self.v3 

        if(s3.cross(v_to_p).dot(tri_n) < 0):
            return None

        normal = tri_n

        return HitInfo(t, p, normal, self.material)



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


# lookfrom  = the point in space where the camera is
# lookat    = the point in space where the camera is looking at
# vup       = the upward direction
# vfov      = the vertical field of view 
# aspect    = the aspect ratio of the output
# aperture  = the aperture of the camera (measures like in a real lens)
# focus_dist= the distance to focus at
class PositionalCamera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist):
        theta = math.radians(vfov)
        half_height = math.tan(theta / 2)
        half_width = aspect * half_height

        w = (lookfrom - lookat).unit()
        u = vup.cross(w).unit()
        v = w.cross(u)

        # self._lower_left_corner = Vec3(-half_width, -half_height, -1)
        self._lower_left_corner = lookfrom - (half_width * focus_dist)*u - (half_height * focus_dist)*v - w * focus_dist
        self._horizontal = 2 * half_width * focus_dist * u
        self._vertical = 2 * half_height * focus_dist * v
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

    def scatter(self, ray, hit_info, attenuation):
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

    def scatter(self, ray, hit_info, attenuation):
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
    def __init__(self, a, refractive_index):
        super().__init__()
        self.transparency = a
        self.refractive_index = refractive_index

    def scatter(self, ray, hit_info, attenuation):
        reflected = reflect(ray.direction, hit_info.normal)
        attenuation.set_(self.transparency)

        # if within hemisphere of normal
        if ray.direction.dot(hit_info.normal) > 0:
            outward_normal = -hit_info.normal # flip normal inwards
            ni_over_nt = self.refractive_index
            cosine = ray.direction.dot(hit_info.normal)/ray.direction.length()
            # cosine = math.sqrt(1 - self.refractive_index * self.refractive_index*(1 - cosine * cosine))
        # we are inside the object
        else:
            outward_normal = hit_info.normal
            ni_over_nt = 1 / self.refractive_index
            cosine = -ray.direction.dot(hit_info.normal) / ray.direction.length()

        refracted = refract(ray.direction, outward_normal, ni_over_nt)
        # only sends out refraction or reflection ray, never both
        if refracted:
            reflect_prob = schlick(cosine, self.refractive_index) # using fresnel approximation
        else:
            reflect_prob = 1

        if random.random() < reflect_prob:
            return ScatterInfo(Ray(hit_info.p, reflected), attenuation)

        return ScatterInfo(Ray(hit_info.p, refracted), attenuation)



'''
Random Scenes for testing
'''
class Scene():
    def random_scene(self, n):
        list_ = []

        # ground
        list_.append(Sphere(Vec3(0, -1000, 0), 1000, Lambertian(Vec3(0.5, 0.5, 0.5))))

        # inserting other items on scene
        for _ in range(n):
            material = random.random()
            center = Vec3(random.random() + 0.9 * random.random(), 0.2, random.random() + 0.9 * random.random())
            if (center - Vec3(4, 0.2, 0)).length() > 0.9:
                # lambertian material
                if(material < 0.8):
                    list_.append(MovingSphere(center, center + Vec3(0, 0.5*random.random(), 0),
                    0.0, 1.0, 0.2,
                    Lambertian(Vec3(random.random(), random.random(), random.random()))))
                
                # Metalic material
                elif(material < 0.95):
                    list_.append(Sphere(center, 0.2, 
                    Metal(Vec3(1+random.random(), 0.5*(1+random.random()), 0.5*(1+random.random())), 0.5*random.random())))

                # Dieletric material
                else:
                    list_.append(Sphere(center, 0.2, 
                    Dielectric(Vec3(random.random()/2+0.5, random.random()/2+0.5, random.random()/2+0.5), 1.5)))

        # three centered spheres
        list_.append(Sphere(Vec3(0, 1, 0), 1, Dielectric(Vec3(0.95, 0.95, 0.95), 1.5)))
        list_.append(Sphere(Vec3(-4, 1, 0), 1, Lambertian(Vec3(0.4, 0.2, 0.1))))
        list_.append(Sphere(Vec3(4, 1, 0), 1, Metal(Vec3(0.7, 0.6, 0.5), 0.0)))

        return list_


    def triangles(self):
        list_ = []
        i = 0

        list_.append(Triangle(Vec3(0,0,0),Vec3(0,0,1),Vec3(0,1,1), 
                    Lambertian(Vec3(0.1, 0.1, 0.1))))
        list_.append(Triangle(Vec3(0,0,0),Vec3(0,1,1),Vec3(0,1,0), 
                    Lambertian(Vec3(0.5, 0.5, 0.5))))            

        list_.append(Sphere(Vec3(0, 0, 0), 0.1, Dielectric(Vec3(0.95, 0.95, 0.95), 1.5)))
        list_.append(Sphere(Vec3(-4, 1, 0), 0.1, Lambertian(Vec3(0.4, 0.2, 0.1))))
        list_.append(Sphere(Vec3(4, 1, 0), 0.1, Metal(Vec3(0.7, 0.6, 0.5), 0.0)))

        return list_


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


# Approximation of fresnel equation using schlick's approximation
def schlick(cosine, refractive_index):
    r0 = (1 - refractive_index) / (1 + refractive_index)
    r0 *= r0
    return r0 + (1 - r0) * pow(1 - cosine, 5)



def color(ray, world, depth):
    hit_info = world.hit(ray, 0.001, sys.float_info.max)

    if hit_info:
        zero = Vec3()

        if depth < 50:
            scatter_info = hit_info.material.scatter(ray, hit_info, zero)
            if scatter_info:
                return scatter_info.attenuation * color(scatter_info.scattered, world, depth+1)
            else:
                return zero
        else:
            return zero

    unit_direction = ray.direction.unit()
    t = 0.5 * (unit_direction.y + 1)
    white = Vec3(1, 1, 1)
    some_color =  Vec3(0.5, 0.7, 1)

    return (1-t)*white + t*some_color


ScatterInfo = namedtuple('ScatterInfo', ['scattered', 'attenuation'])
HitInfo = namedtuple('HitInfo', ['t', 'p', 'normal', 'material'])


def main():
    #timer to count program execution
    random.seed(time.time())

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

    # file_ = FileLoader("teapot.stl")

    world = HitableList(Scene().random_scene(10))
    # world = HitableList(Scene().random_scene(500))

    # book scene
    lookfrom = Vec3(13.5, 4, 3)
    lookat = Vec3(0.0, 1, 0)
    dist_to_focus = (lookfrom - lookat).length()
    aperture = 7.1

    # lookfrom = Vec3(-5,0,0)
    # lookat = Vec3(0,0,0)
    # dist_to_focus = (lookfrom - lookat).length()
    # aperture = 14
    
    cam = PositionalCamera(lookfrom, lookat, Vec3(0, 1, 0), 40, width / height, aperture, dist_to_focus)
    ns = height # number of samples


    # Save the PPM image as a binary file
    with open(output, 'w') as f:
        start_time = time.time()
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
    
    end_time = time.time() - start_time
    print('Renderizado em ' + str(end_time) + ' segundos')
    f.close()

if __name__ == '__main__':
    main()
