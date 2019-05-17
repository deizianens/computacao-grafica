# faz o setup do ambiente
import sys
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
import pyrr
import glfw
import numpy

# shaders
vertex_shader = '''
#version 330
uniform mat4 model_transform;
uniform mat4 camera;
uniform mat4 projection;
uniform vec3 camera_pos_input;

uniform float albedo_r;
uniform float albedo_g;
uniform float albedo_b;
uniform float ks;

in vec3 position;
in vec3 vertex_normal;

out vec3 view_position;
out vec3 pixel_position;
out vec3 vertex_color;
out vec3 camera_position;
out vec3 normal;
out vec3 light_source;
out vec4 ambient_light;

void main()
{
    gl_Position = projection * camera * model_transform * vec4(position, 1.0);
    view_position = gl_Position.xyz;
    pixel_position = position;
    camera_position = camera_pos_input;
    if(vertex_normal == vec3(0.0, 0.0, 0.0))
    {
        normal = normalize(position);
    }
    else
    {
        normal = normalize(vertex_normal);
    }
    
    light_source = vec3(5.0, 20.0, 10.0);
    float diffuse_red = albedo_r / 3.14 * dot(normal, light_source);
    float diffuse_green = albedo_g / 3.14 * dot(normal, light_source);
    float diffuse_blue = albedo_b / 3.14 * dot(normal, light_source);
    ambient_light = vec4(albedo_r/3.14, albedo_g/3.14, albedo_b/3.14, 0.0);

    vec3 reflection_vec = 2.0 * dot(light_source, normal) * normal - light_source;
    float specular = ks * dot(reflection_vec, normalize(camera_position));
    
    specular = max(0.0, specular);
    diffuse_red = max(0.0, diffuse_red);
    diffuse_green = max(0.0, diffuse_green);
    diffuse_blue = max(0.0, diffuse_blue);
    
    vertex_color = vec3(diffuse_red + specular, diffuse_green + specular, diffuse_blue + specular) + ambient_light.rgb;
}
'''

fragment_shader = '''
#version 330
uniform float albedo_r;
uniform float albedo_g;
uniform float albedo_b;
uniform float ks;
uniform int shading_type;

in vec3 view_position;
in vec3 pixel_position;
in vec3 vertex_color;
in vec3 camera_position;
in vec3 normal;
in vec3 light_source;
in vec4 ambient_light;

out vec4 frag_color;

void main()
{    
    if(shading_type == 0) //flat
    {
        vec3 x_tangent = dFdx(pixel_position);
        vec3 y_tangent = dFdy(pixel_position);
        vec3 face_normal = normalize(cross(x_tangent, y_tangent));
        
        float red_diffuse = albedo_r / 3.14 * dot(face_normal, light_source);
        float green_diffuse = albedo_g / 3.14 * dot(face_normal, light_source);
        float blue_diffuse = albedo_b / 3.14 * dot(face_normal, light_source);
        
        vec3 reflection_vec = 2.0 * dot(light_source, face_normal) * face_normal - light_source;
        float specular = ks * dot(reflection_vec, normalize(camera_position));
        
        specular = max(0.0, specular);
        red_diffuse = max(0.0, red_diffuse);
        green_diffuse = max(0.0, green_diffuse);
        blue_diffuse = max(0.0, blue_diffuse);
        
        frag_color = vec4(red_diffuse + specular, green_diffuse + specular, blue_diffuse + specular, 1.0) + ambient_light;
    }
    else if(shading_type == 1) // gouraud
    {
        frag_color = vec4(vertex_color, 1.0);
    }
    else // phong
    {
        vec3 interpolated_normal = normalize(normal);
        
        float red_diffuse = albedo_r / 3.14 * dot(interpolated_normal, light_source);
        float green_diffuse = albedo_g / 3.14 * dot(interpolated_normal, light_source);
        float blue_diffuse = albedo_b / 3.14 * dot(interpolated_normal, light_source);
        
        vec3 reflection_vec = 2.0 * dot(light_source, interpolated_normal) * interpolated_normal - light_source;
        float specular = ks * dot(reflection_vec, normalize(camera_position));
        
        specular = max(0.0, specular);
        red_diffuse = max(0.0, red_diffuse);
        green_diffuse = max(0.0, green_diffuse);
        blue_diffuse = max(0.0, blue_diffuse);
        
        frag_color = vec4(red_diffuse + specular, green_diffuse + specular, blue_diffuse + specular, 1.0) + ambient_light;
    }
}
'''

class Teapot():
    def __init__(self):
        vertices = []
        faces = []
        # obj loader super simples
        with open('teapot.obj', 'r') as obj: # le os vertices e faces do OBJ
            for line in obj.readlines():
                line = line.split(' ')
                if line[0] == 'v':
                    vertices.append([float(line[1])/3, float(line[2])/3, float(line[3])/3])
                elif line[0] == 'f':
                    faces.append([int(line[1])-1, int(line[2])-1, int(line[3])-1])
        
        # calculate vertex normals
        self.vertices = numpy.array(vertices, dtype=numpy.float32)
        self.faces = numpy.array(faces, dtype=numpy.uint32)
        vertex_normals = pyrr.vector3.generate_vertex_normals(self.vertices, self.faces)
        self.vertex_normals = numpy.array(vertex_normals, dtype=numpy.float32).flatten()
        self.vertices = numpy.array(vertices, dtype=numpy.float32).flatten()
        self.faces = numpy.array(faces, dtype=numpy.uint32).flatten()

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, len(self.vertices) * 4, self.vertices, GL_STATIC_DRAW)
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.faces) * 4, self.faces, GL_STATIC_DRAW)
        self.nbo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.nbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.vertex_normals) * 4, self.vertex_normals, GL_STATIC_DRAW)
        
    def render(self, shader):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        position = glGetAttribLocation(shader, 'position')
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.nbo)
        vertex_normal = glGetAttribLocation(shader, 'vertex_normal')
        glVertexAttribPointer(vertex_normal, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(vertex_normal)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glDrawElements(GL_TRIANGLES, len(self.faces), GL_UNSIGNED_INT, None)

# material
class PhongMaterial:
    def __init__(self, shader, shading_type, albedo_r, albedo_g, albedo_b, specular_constant):
        self.shader = shader
        self.shading_type = shading_type
        self.albedo = [albedo_r, albedo_g, albedo_b]
        self.specular_constant = specular_constant
    def set_up_rendering(self):
        glUseProgram(self.shader)
        
        #lighting uniforms
        r = glGetUniformLocation(self.shader, 'albedo_r')
        g = glGetUniformLocation(self.shader, 'albedo_g')
        b = glGetUniformLocation(self.shader, 'albedo_b')
        glUniform1f(r, self.albedo[0])
        glUniform1f(g, self.albedo[1])
        glUniform1f(b, self.albedo[2])
        
        ks = glGetUniformLocation(self.shader, 'ks')
        glUniform1f(ks, self.specular_constant)
        
        st = glGetUniformLocation(self.shader, 'shading_type')
        if self.shading_type == 'gouraud':
            glUniform1i(st, 1)
        elif self.shading_type == 'phong':
            glUniform1i(st, 2)
        else:
            glUniform1i(st, 0)

class Shape:
    def __init__(self, shape_type, material):
        self.material = material
        self.shape_type = shape_type
        self.teapot = Teapot()
    def render(self):
        self.material.set_up_rendering()
        
        #transformation uniforms
        model_transform = pyrr.Matrix44.identity()
        perspective_transform = pyrr.Matrix44.perspective_projection(45, 4/3, 0.01, 100)
        camera_transform = pyrr.Matrix44.look_at((2, 2, 2), (0, 0, 0), (0, 1, 0))
        
        mt_loc = glGetUniformLocation(self.material.shader, 'model_transform')
        glUniformMatrix4fv(mt_loc, 1, GL_FALSE, model_transform)
        pr_loc = glGetUniformLocation(self.material.shader, 'projection')
        glUniformMatrix4fv(pr_loc, 1, GL_FALSE, perspective_transform)
        cam_loc = glGetUniformLocation(self.material.shader, 'camera')
        glUniformMatrix4fv(cam_loc, 1, GL_FALSE, camera_transform)
        cam_pos_loc = glGetUniformLocation(self.material.shader, 'camera_pos_input')
        glUniform3f(cam_pos_loc, 2, 2, 2)
        
        qobj = gluNewQuadric()
        gluQuadricNormals(qobj, GLU_SMOOTH)
        gluQuadricOrientation(qobj, GLU_OUTSIDE)
        if self.shape_type == 'sphere':
            vn = glGetAttribLocation(self.material.shader, 'vertex_normal')
            glVertexAttrib3f(vn, 0.0, 0.0, 0.0)
            glDisableVertexAttribArray(vn)
            gluSphere(qobj, 1, 50, 50)
        elif self.shape_type == 'cylinder':
            vn = glGetAttribLocation(self.material.shader, 'vertex_normal')
            glVertexAttrib3f(vn, 0.0, 0.0, 0.0)
            glDisableVertexAttribArray(vn)
            gluCylinder(qobj, 1, 1, 1, 50, 50)
        else:
            self.teapot.render(self.material.shader)

def get_input(window, shape, flags):
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            shape.material.specular_constant -= 0.001
            if shape.material.specular_constant <= 0:
                shape.material.specular_constant = 0
                
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            shape.material.albedo[0] -= 0.01
            if shape.material.albedo[0] <= 0:
                shape.material.albedo[0] = 0
                
        if glfw.get_key(window, glfw.KEY_G) == glfw.PRESS:
            shape.material.albedo[1] -= 0.01
            if shape.material.albedo[1] <= 0:
                shape.material.albedo[1] = 0
                
        if glfw.get_key(window, glfw.KEY_B) == glfw.PRESS:
            shape.material.albedo[2] -= 0.01
            if shape.material.albedo[2] <= 0:
                shape.material.albedo[2] = 0
                
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            shape.material.specular_constant += 0.001
            if shape.material.specular_constant >= 1:
                shape.material.specular_constant = 1
                
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            shape.material.albedo[0] += 0.01
            if shape.material.albedo[0] >= 1:
                shape.material.albedo[0] = 1
                
        if glfw.get_key(window, glfw.KEY_G) == glfw.PRESS:
            shape.material.albedo[1] += 0.01
            if shape.material.albedo[1] >= 1:
                shape.material.albedo[1] = 1
                
        if glfw.get_key(window, glfw.KEY_B) == glfw.PRESS:
            shape.material.albedo[2] += 0.01
            if shape.material.albedo[2] >= 1:
                shape.material.albedo[2] = 1
                
    if glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS and not flags[0]:
        if shape.material.shading_type == 'flat':
            shape.material.shading_type = 'gouraud'
        elif shape.material.shading_type == 'gouraud':
            shape.material.shading_type = 'phong'
        elif shape.material.shading_type == 'phong':
            shape.material.shading_type = 'flat'
        flags[0] = True
    
    if glfw.get_key(window, glfw.KEY_ENTER) == glfw.RELEASE:
        flags[0] = False

    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS and not flags[1]:
        if shape.shape_type == 'sphere':
            shape.shape_type = 'cylinder'
        elif shape.shape_type == 'cylinder':
            shape.shape_type = 'teapot'
        elif shape.shape_type == 'teapot':
            shape.shape_type = 'sphere'
        flags[1] = True
        
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.RELEASE:
        flags[1] = False

def main():
    if not glfw.init():
        return
    window = glfw.create_window(1280, 760, 'Shadings', None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    
    glClearColor(0.2, 0.3, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    glCullFace(GL_BACK)
    
    shader = shaders.compileProgram(shaders.compileShader(vertex_shader, GL_VERTEX_SHADER), shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    shape = Shape('sphere', PhongMaterial(shader, 'flat', .0, .5, .5, 0.05)) # valores default iniciais
    key_flags = [False, False]
    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        get_input(window, shape, key_flags)
        shape.render()
        glfw.swap_buffers(window)
    glfw.terminate()
    
if __name__ == '__main__':
    main()