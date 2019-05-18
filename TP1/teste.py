# faz o setup do ambiente
import sys
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *
from enum import Enum 

import glfw
import math


PI = 3.14

# shaders
vertex_shader = '''
#version 330

in vec3 position;
in vec3 normal;
in vec2 TexCoord;

uniform mat4 projection, modelview, normalMat;

out vec3 normalInterp;
out vec3 vertPos;
out vec3 vertexColor; //only for gouraud


uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform vec3 lightPos; // Light position

// uniform float Ka;   // Ambient reflection coefficient
// uniform float Kd;   // Diffuse reflection coefficient
// uniform float Ks;   // Specular reflection coefficient
// uniform float shininessVal; // Shininess


def main(){
    gl_Position = projection * modelview * vec4(inputPosition, 1.0);

    vec4 vertPos4 = modelview * vec4(position, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
    normalInterp = vec3(normalMat * vec4(normal, 0.0));

    // ------------ only for gouraud -----------------
    vec3 normal = vec3(normalMat * vec4(normal, 0.0));
    vec3 lightDir = normalize(lightPos - vertPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    vec3 viewDir = normalize(-vertPos);

    float lambertian = max(dot(lightDir,normal), 0.0);
    float specular = 0.0;

    if(lambertian > 0.0) {
       float specAngle = max(dot(reflectDir, viewDir), 0.0);
       specular = pow(specAngle, 4.0);
    }

    vertexColor = vec4(ambientColor+lambertian*diffuseColor + specular*specColor, 1.0);

}

'''

fragment_shader = '''
#version 330

// precision mediump float;

in vec3 normalInterp;  // Surface normal
in vec3 vertPos;       // Vertex position
in vec4 vertexColor;  // Only for gouraud

uniform vec3 lightPos; // Light position
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform int shading;

out vec4 color;

// uniform float Ka;   // Ambient reflection coefficient
// uniform float Kd;   // Diffuse reflection coefficient
// uniform float Ks;   // Specular reflection coefficient
// uniform float shininessVal; // Shininess 


def main(){    
    //flat
    if(shading == 0) {
        vec3 normal = normalize(normalInterp);
        vec3 lightDir = normalize(lightPos - vertPos);
        vec3 reflectDir = reflect(-lightDir, normal);
        vec3 viewDir = normalize(-vertPos);

        float lambertian = max(dot(lightDir,normal), 0.0);
        float specular = 0.0;

        if(lambertian > 0.0) {
        float specAngle = max(dot(reflectDir, viewDir), 0.0);
        specular = pow(specAngle, 4.0);
        }

        color = vec4(ambientColor +
                      lambertian*diffuseColor +
                      specular*specColor, 1.0);

    }
    // gouraud
    else if(shading == 1) {
        color = vertexColor;
    }
    // phong
    else {  
        vec3 normal = normalize(normalInterp);
        vec3 lightDir = normalize(lightPos - vertPos);
        vec3 reflectDir = reflect(-lightDir, normal);
        vec3 viewDir = normalize(-vertPos);

        //lightDir*normal
        float lambertian = max(dot(lightDir,normal), 0.0);
        float specular = 0.0;

        if(lambertian > 0.0) {
        float specAngle = max(dot(reflectDir, viewDir), 0.0);
        specular = pow(specAngle, 4.0);
        }
        outputColor = vec4(ambientColor +
                        lambertian*diffuseColor +
                        specular*specColor, 1.0);


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

class Vertex:
    position = []
    texCoord = [] #coordinate
    normal = []
  

class Renderer:
    def __init__(self):
        self.vertex = Vertex()

        self.t = 0.0
        self.modeVal = 0

        Enum(Scene, numVAOs) #Vertex Array Object
        Enum(SceneAll, numVBOs)
        self.vaoID[numVAOs]
        self.bufID[numVBOs]
        self.sceneVertNo = 0
        self.progID = 0
        self.vertID = 0
        self.fragID = 0
        self.vertexLoc = -1
        self.texCoordLoc = -1
        self.normalLoc = -1
        self.projectionLoc = -1
        self.modelviewLoc = -1
        self.normalMatrixLoc = -1
        self.modeLoc = -1
        self.projection = [];  # projection matrix
        self.modelview = [];   # modelview matrix
        self.filename = "./sphere.vbo"
        self.file = 0

    def init(self):
        glEnable(GL_DEPTH_TEST)
        setupShaders()

        # create a Vertex Array Objects (VAO)
        glGenVertexArrays(numVAOs, vaoID)

        # generate a Vertex Buffer Object (VBO)
        glGenBuffers(numVBOs, bufID)

        # binding the pyramid VAO
        glBindVertexArray(vaoID[Scene])

        data = list()
        loadVertexData(filename, data)

        sceneVertexData = loadVertexData(currentFileName)
        sceneVertNo = int(data.lenght()) / (3+2+3)

        glBindBuffer(GL_ARRAY_BUFFER, bufID[SceneAll])
        glBufferData(GL_ARRAY_BUFFER, data[0], GL_STATIC_DRAW) # size can be omitted

        stride = sys.getsizeof(vertex)
        offset = None

        # position
        if(vertexLoc != -1):
            glVertexAttribPointer(vertexLoc, 3, GL_FLOAT, GL_FALSE, stride, offset)
            glEnableVertexAttribArray(vertexLoc)
        

        # texCoord
        if(texCoordLoc != -1):
            offset = None + 3*sys.getsizeof(float)
            glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, GL_FALSE, stride, offset)
            glEnableVertexAttribArray(texCoordLoc)
        

        # normal
        if(normalLoc != -1):
            offset = None + (3+2)*sys.getsizeof(float)
            glVertexAttribPointer(normalLoc, 3, GL_FLOAT, GL_FALSE, stride, offset)
            glEnableVertexAttribArray(normalLoc)
    
    def display(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rad = PI / 180.0 * t
    
        mat4LookAt(modelview,
               1.5*float(cos(rad)), 1.5*float(sin(rad)), 1.5, # eye
               0.0, 0.0, 0.0, # look at
               0.0, 0.0, 1.0) # up


        modelviewInv = [] 
        normalmatrix = []
        mat4Invert(modelview, modelviewInv)
        mat4Transpose(modelviewInv, normalmatrix)

        glUseProgram(progID)
        
        # load the current projection and modelview matrix into the
        # corresponding UNIFORM variables of the shader
        glUniformMatrix4fv(projectionLoc, 1, false, projection)
        glUniformMatrix4fv(modelviewLoc, 1, false, modelview)
        if(normalMatrixLoc != -1):
            glUniformMatrix4fv(normalMatrixLoc, 1, false, normalmatrix)
        if(modeLoc != -1):
            glUniform1i(modeLoc, modeVal)

        # bind Triangle VAO
        glBindVertexArray(vaoID[Scene])
        # render data
        glDrawArrays(GL_TRIANGLES, 0, sceneVertNo)
  

    def setupShaders(self, s):
        # compile the shader
        shader = shaders.compileProgram(shaders.compileShader(vertex_shader, GL_VERTEX_SHADER), shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

        glUseProgram(shader)
    
        # retrieve the location of the IN variables of the vertex shaders
        vertexLoc = glGetAttribLocation(progID,"Position")
        texCoordLoc = glGetAttribLocation(progID,"TexCoord")
        normalLoc = glGetAttribLocation(progID, "Normal")

        # retrieve the location of the UNIFORM variables of the shader
        projectionLoc = glGetUniformLocation(shader, "projection")
        modelviewLoc = glGetUniformLocation(shader, "modelview")
        normalMatrixLoc = glGetUniformLocation(shader, "normalMat")
        lightPosLoc = glGetUniformLocation(shader, "lightPos")
        ambientColorLoc = glGetUniformLocation(shader, "ambientColor")
        diffuseColorLoc = glGetUniformLocation(shader, "diffuseColor")
        specularColorLoc = glGetUniformLocation(shader, "specularColor")
        shading = glGetUniformLocation(shader, 'shading')

        if s == 'flat':
            glUniform1i(shading, 0)
        elif s == 'gouraud':
            glUniform1i(shading, 1)
        else:
            glUniform1i(shading, 2)
    
    # ----- the following functions are some matrix and vector helpers --------
    def vec3Dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def vec3Cross(a, b, res):
        res[0] = a[1] * b[2]  -  b[1] * a[2]
        res[1] = a[2] * b[0]  -  b[2] * a[0]
        res[2] = a[0] * b[1]  -  b[0] * a[1]


    def vec3Normalize(a):
        mag = math.sqrt(a[0] * a[0]  +  a[1] * a[1]  +  a[2] * a[2])
        a[0] /= mag; a[1] /= mag; a[2] /= mag
  

    def mat4Identity(a):
        for i in range(16):
            a[i] = 0.0
        for i in range(4):
            a[i + i * 4] = 1.0
    
    def mat4Multiply(a, b, res):
        for i in range(4):
            for j in range(4): 
                res[j*4 + i] = 0.0
                for k in range(4): 
                    res[j*4 + i] += a[k*4 + i] * b[j*4 + k]

    
    def mat4Perspective(a, fov, aspect, zNear, zFar):
        f = 1.0 / (tan (fov/2.0 * (PI / 180.0)))
        mat4Identity(a)
        a[0] = f / aspect
        a[1 * 4 + 1] = f
        a[2 * 4 + 2] = (zFar + zNear)  / (zNear - zFar)
        a[3 * 4 + 2] = (2.0 * zFar * zNear) / (zNear - zFar)
        a[2 * 4 + 3] = -1.0
        a[3 * 4 + 3] = 0.0
    

    def mat4LookAt(viewMatrix,
                    eyeX, eyeY, eyeZ,
                    centerX, centerY, centerZ,
                    upX, upY, upZ):

        dr = [] 
        right = [] 
        up = []
        eye = []

        up[0]=upX
        up[1]=upY 
        up[2]=upZ

        eye[0]=eyeX
        eye[1]=eyeY
        eye[2]=eyeZ

        dr[0]=centerX-eyeX
        dr[1]=centerY-eyeY
        dr[2]=centerZ-eyeZ

        vec3Normalize(dr)
        vec3Cross(dr,up,right)
        vec3Normalize(right)
        vec3Cross(right,dr,up)
        vec3Normalize(up)

        # first row
        viewMatrix[0]  = right[0]
        viewMatrix[4]  = right[1]
        viewMatrix[8]  = right[2]
        viewMatrix[12] = -vec3Dot(right, eye)

        # second row
        viewMatrix[1]  = up[0]
        viewMatrix[5]  = up[1]
        viewMatrix[9]  = up[2]
        viewMatrix[13] = -vec3Dot(up, eye)
        
        # third row
        viewMatrix[2]  = -dr[0]
        viewMatrix[6]  = -dr[1]
        viewMatrix[10] = -dr[2]
        viewMatrix[14] =  vec3Dot(dr, eye)

        # forth row
        viewMatrix[3]  = 0.0
        viewMatrix[7]  = 0.0
        viewMatrix[11] = 0.0
        viewMatrix[15] = 1.0
    

    def mat4Print(a):
        # opengl uses column major order
        for i in range(4):
            for i in range(4): 
                print(a[j * 4 + i] + " ")
            print("\n")
        

    def mat4Transpose(a, transposed):
        t = 0
        for i in range(4):
            for j in range(4): 
                transposed[t] = a[j * 4 + i]
                t +=1
        

    def mat4Invert(m, inverse):
        inv = []
        inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10]
        inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10]
        inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9]
        inv[12]= -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9]
        inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10]
        inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10]
        inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9]
        inv[13]= m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9]
        inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6]
        inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6]
        inv[10]= m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5]
        inv[14]= -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5]
        inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6]
        inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6]
        inv[11]= -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5]
        inv[15]= m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5]

        det = m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12]
        if (det == 0):
             return False
        det = 1.0 / det
        for j in range(16):
            inverse[i] = inv[i] * det
        return True
  


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA)
    glutInitWindowPosition(100,100)
    glutInitWindowSize(320, 320)

    window = glutCreateWindow("Shading")
    glutDisplayFunc(glutDisplay)
    glutFullScreen()

    glutIdleFunc(glutDisplay)
    glutReshapeFunc(glutResize)
    glutKeyboardFunc(glutKeyboard)

    renderer = Renderer()
    renderer.init()

    glutMainLoop()


if __name__ == '__main__':
    main()