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

attribute vec3 position;
attribute vec3 normal;
uniform mat4 projection, modelview, normalMat;
varying vec3 normalInterp;
varying vec3 vertPos;

uniform float Ka;   // Ambient reflection coefficient
uniform float Kd;   // Diffuse reflection coefficient
uniform float Ks;   // Specular reflection coefficient
uniform float shininessVal; // Shininess

// Material color
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform vec3 lightPos; // Light position
varying vec4 color; //color

void main(){
  vec4 vertPos4 = modelview * vec4(position, 1.0);
  vertPos = vec3(vertPos4) / vertPos4.w;
  normalInterp = vec3(normalMat * vec4(normal, 0.0));
  gl_Position = projection * vertPos4;

  vec3 N = normalize(normalInterp);
  vec3 L = normalize(lightPos - vertPos);
  // Lambert's cosine law
  float lambertian = max(dot(N, L), 0.0);
  float specular = 0.0;
  if(lambertian > 0.0) {
    vec3 R = reflect(-L, N);      // Reflected light vector
    vec3 V = normalize(-vertPos); // Vector to viewer
    // Compute the specular term
    float specAngle = max(dot(R, V), 0.0);
    specular = pow(specAngle, shininessVal);
  }
  color = vec4(Ka * ambientColor +
               Kd * lambertian * diffuseColor +
               Ks * specular * specularColor, 1.0);

}

'''

fragment_shader = '''
#version 330

precision mediump float;

varying vec3 normalInterp;  // Surface normal
varying vec3 vertPos;       // Vertex position

uniform float Ka;   // Ambient reflection coefficient
uniform float Kd;   // Diffuse reflection coefficient
uniform float Ks;   // Specular reflection coefficient
uniform float shininessVal; // Shininess

// Material color
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform vec3 lightPos; // Light position


void main()
{    
    if(shading_type == 0) //flat
    {
        float diff = max(0.0, dot(normalize(normalInterp), normalize(lightPos)));
        vFragColor = diff * diffuseColor;
        vFragColor += ambientColor;
        vec3 vReflection = normalize(reflect(-normalize(lightPos),normalize(normalInterp)));
        float spec = max(0.0, dot(normalize(normalInterp), vReflection));

        if(diff != 0) {
        float fSpec = pow(spec, 32.0);
        gl_FragColor += vec3(fSpec, fSpec, fSpec);
    }
    else if(shading_type == 1) // gouraud
    {
        gl_FragColor = color;
    }
    else { // phong 
        vec3 N = normalize(normalInterp);
        vec3 L = normalize(lightPos - vertPos);

        // Lambert's cosine law
        float lambertian = max(dot(N, L), 0.0);
        float specular = 0.0;
        if(lambertian > 0.0) {
            vec3 R = reflect(-L, N);      // Reflected light vector
            vec3 V = normalize(-vertPos); // Vector to viewer
            // Compute the specular term
            float specAngle = max(dot(R, V), 0.0);
            specular = pow(specAngle, shininessVal);
        }
        gl_FragColor = vec4(Ka * ambientColor +
                            Kd * lambertian * diffuseColor +
                            Ks * specular * specularColor, 1.0);

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