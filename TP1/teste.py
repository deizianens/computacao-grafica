# faz o setup do ambiente
import sys
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pyrr
import glfw
import numpy
import math

PI = 3.14

# shaders
vertex_shader = '''
#version 330

in vec3 position;
in vec3 normal;

uniform mat4 projection, modelview, normalMat;
out vec3 normalInterp;
out vec3 vertPos;

uniform float Ka;   // Ambient reflection coefficient
uniform float Kd;   // Diffuse reflection coefficient
uniform float Ks;   // Specular reflection coefficient
uniform float shininessVal; // Shininess

// Material color
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform vec3 lightPos; // Light position

out vec3 color; //color

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

// precision mediump float;

varying vec3 normalInterp;  // Surface normal
varying vec3 vertPos;       // Vertex position

uniform float Ka;   // Ambient reflection coefficient
uniform float Kd;   // Diffuse reflection coefficient
uniform float Ks;   // Specular reflection coefficient
uniform float shininessVal; // Shininess

uniform int shading;

// Material color
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform vec3 lightPos; // Light position

varying vec4 color;

void main()
{    
    if(shading == 0) //flat
    {
        float diff = max(0.0, dot(normalize(normalInterp), normalize(lightPos)));
        vFragColor = diff * diffuseColor;
        vFragColor += ambientColor;
        vec3 vReflection = normalize(reflect(-normalize(lightPos),normalize(normalInterp)));
        float spec = max(0.0, dot(normalize(normalInterp), vReflection));

        if(diff != 0) {
        float fSpec = pow(spec, 32.0);
        color += vec3(fSpec, fSpec, fSpec);
    }
    else if(shading == 1) // gouraud
    {
        color = color;
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
        color = vec4(Ka * ambientColor +
                            Kd * lambertian * diffuseColor +
                            Ks * specular * specularColor, 1.0);

    }
}
'''

class Shape:
    def __init__(self, shape, shader):
        self.material = material
        self.shape = shape
        self.shader = shader

    def render(self):
        setupShaders()
        
        #transformation uniforms
        model_transform = pyrr.Matrix44.identity()
        perspective_transform = pyrr.Matrix44.perspective_projection(45, 4/3, 0.01, 100)
        camera_transform = pyrr.Matrix44.look_at((2, 2, 2), (0, 0, 0), (0, 1, 0))
        
        modelviewLoc = glGetUniformLocation(self.shader, 'modelView')
        glUniformMatrix4fv(modelviewLoc, 1, GL_FALSE, model_transform)
        projectionLoc = glGetUniformLocation(self.shader, 'projection')
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, perspective_transform)
        
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluQuadricOrientation(quad, GLU_OUTSIDE)
        if self.shape == 'sphere':
            gluSphere(quad, 2.0, 256, 256) # parameters: GLUquadric* quad, GLdouble radius, GLint slices, GLint stacks
        elif self.shape == 'cylinder':
            gluCylinder(quad, 2.0, 2.0, 5.0, 256, 256) # parameters: GLUquadric* quad, GLdouble base, GLdouble top, GLdouble height, GLint slices, GLint stacks
        else:
            glutSolidTeapot(50.0)


# def display():
#     glClearColor(0.0, 0.0, 1.0, 1.0)    # glClearColor — specify clear values for the color buffers
#     glClear(GL_COLOR_BUFFER_BIT)       # glClear — clear buffers to preset values
    
#     # camera orbits in the z=1.5 plane and looks at the origin
#     # mat4LookAt replaces gluLookAt
#     rad = PI / 180.0 * this.t

#     mat4LookAt(
#         modelview,
#         1.5 * math.cos(rad), 1.5 * math.sin(rad), 1.5, # eye
#         0.0, 0.0, 0.0, # look at
#         0.0, 0.0, 1.0
#     ); # up


def setupShaders(s, shader):
    glUseProgram(shader)
    
    # retrieve the location of the IN variables of the vertex shader
    vertexLoc = glGetAttribLocation(shader, "position")
    normalLoc = glGetAttribLocation(shader, "normal")

    # retrieve the location of the UNIFORM variables of the shader
    projectionLoc = glGetUniformLocation(shader, "projection")
    modelviewLoc = glGetUniformLocation(shader, "modelview")
    normalMatrixLoc = glGetUniformLocation(shader, "normalMat")
    lightPosLoc = glGetUniformLocation(shader, "lightPos")
    ambientColorLoc = glGetUniformLocation(shader, "ambientColor")
    diffuseColorLoc = glGetUniformLocation(shader, "diffuseColor")
    specularColorLoc = glGetUniformLocation(shader, "specularColor")
    shininessLoc = glGetUniformLocation(shader, "shininessVal")
    kaLoc = glGetUniformLocation(shader, "Ka")
    kdLoc = glGetUniformLocation(shader, "Kd")
    ksLoc = glGetUniformLocation(shader, "Ks")

    shading = glGetUniformLocation(shader, 'shading')
    if s == 'flat':
        glUniform1i(shading, 0)
    elif s == 'gouraud':
        glUniform1i(shading, 1)
    else:
        glUniform1i(shading, 2)


def main():
    if not glfw.init():
        return
    window = glfw.create_window(800, 800, 'Shadings', None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    
    glClearColor(0.0, 0.0, 1.0, 1.0)    # glClearColor — specify clear values for the color buffers
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    
    # create shader
    shader = shaders.compileProgram(shaders.compileShader(vertex_shader, GL_VERTEX_SHADER), 
                                    shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    
    setupShaders('gouraud', shader)
    shape = Shape('sphere', shader) 
    
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        shape.render()
        glfw.swap_buffers(window)
    glfw.terminate()
    
if __name__ == '__main__':
    main()