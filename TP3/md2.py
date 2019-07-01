# encoding=utf-8
import sys
import glfw
import struct
import time
import numpy as np

import OpenGL.GL.shaders as gl_shaders
from OpenGL.GL import *
from OpenGL.GLU import *


vertex_shader = """
#version 330 compatibility

out vec3 normal;
out vec4 position;

void main() {
    normal = normalize(gl_NormalMatrix * gl_Normal);
    position = gl_ModelViewMatrix * gl_Vertex;
    
    gl_Position = ftransform();
}
"""

fragment_shader = """
#version 330 compatibility

in vec4 position;
in vec3 normal;

uniform vec3 lightPos;
uniform vec3 color;
const float specular = 0.4;
const float diffuse = 1.0 - specular;

out vec4 outColor;

void main() {
    vec3 p = vec3(gl_ModelViewMatrix * position);
    vec3 n = normalize(gl_NormalMatrix * normal);
    vec3 lightDir = normalize(lightPos - p);
    vec3 R = reflect(lightDir, n);
    vec3 viewVec = normalize(-p);
    
    float diff = max(0.0, dot(lightDir, n));
    float spec = 0.0;
    
    if (diff > 0.0) {
        spec = max(0.0, dot(R, viewVec));
        spec = pow(spec, 64.0);
    }
    float intensity = (diff * diffuse) + (spec * specular);
    
    vec3 ambientLight = vec3(0.15, 0.1, 0.1);
    
    outColor = vec4(ambientLight + color * intensity, 1.0);
}
"""

# window size
width, height = 800, 600

# magic number. must be equal to "IDP2"
id = 'IDP2' 

# md2 version. must be equal to 8
version = 8

# The MD2 normal look-up table
normals_table = [
    [ -0.525731,  0.000000,  0.850651], 
    [ -0.442863,  0.238856,  0.864188], 
    [ -0.295242,  0.000000,  0.955423], 
    [ -0.309017,  0.500000,  0.809017], 
    [ -0.162460,  0.262866,  0.951056], 
    [  0.000000,  0.000000,  1.000000], 
    [  0.000000,  0.850651,  0.525731], 
    [ -0.147621,  0.716567,  0.681718], 
    [  0.147621,  0.716567,  0.681718], 
    [  0.000000,  0.525731,  0.850651], 
    [  0.309017,  0.500000,  0.809017], 
    [  0.525731,  0.000000,  0.850651], 
    [  0.295242,  0.000000,  0.955423], 
    [  0.442863,  0.238856,  0.864188], 
    [  0.162460,  0.262866,  0.951056], 
    [ -0.681718,  0.147621,  0.716567], 
    [ -0.809017,  0.309017,  0.500000], 
    [ -0.587785,  0.425325,  0.688191], 
    [ -0.850651,  0.525731,  0.000000], 
    [ -0.864188,  0.442863,  0.238856], 
    [ -0.716567,  0.681718,  0.147621], 
    [ -0.688191,  0.587785,  0.425325], 
    [ -0.500000,  0.809017,  0.309017], 
    [ -0.238856,  0.864188,  0.442863], 
    [ -0.425325,  0.688191,  0.587785], 
    [ -0.716567,  0.681718, -0.147621], 
    [ -0.500000,  0.809017, -0.309017], 
    [ -0.525731,  0.850651,  0.000000], 
    [  0.000000,  0.850651, -0.525731], 
    [ -0.238856,  0.864188, -0.442863], 
    [  0.000000,  0.955423, -0.295242], 
    [ -0.262866,  0.951056, -0.162460], 
    [  0.000000,  1.000000,  0.000000], 
    [  0.000000,  0.955423,  0.295242], 
    [ -0.262866,  0.951056,  0.162460], 
    [  0.238856,  0.864188,  0.442863], 
    [  0.262866,  0.951056,  0.162460], 
    [  0.500000,  0.809017,  0.309017], 
    [  0.238856,  0.864188, -0.442863], 
    [  0.262866,  0.951056, -0.162460], 
    [  0.500000,  0.809017, -0.309017], 
    [  0.850651,  0.525731,  0.000000], 
    [  0.716567,  0.681718,  0.147621], 
    [  0.716567,  0.681718, -0.147621], 
    [  0.525731,  0.850651,  0.000000], 
    [  0.425325,  0.688191,  0.587785], 
    [  0.864188,  0.442863,  0.238856], 
    [  0.688191,  0.587785,  0.425325], 
    [  0.809017,  0.309017,  0.500000], 
    [  0.681718,  0.147621,  0.716567], 
    [  0.587785,  0.425325,  0.688191], 
    [  0.955423,  0.295242,  0.000000], 
    [  1.000000,  0.000000,  0.000000], 
    [  0.951056,  0.162460,  0.262866], 
    [  0.850651, -0.525731,  0.000000], 
    [  0.955423, -0.295242,  0.000000], 
    [  0.864188, -0.442863,  0.238856], 
    [  0.951056, -0.162460,  0.262866], 
    [  0.809017, -0.309017,  0.500000], 
    [  0.681718, -0.147621,  0.716567], 
    [  0.850651,  0.000000,  0.525731], 
    [  0.864188,  0.442863, -0.238856], 
    [  0.809017,  0.309017, -0.500000], 
    [  0.951056,  0.162460, -0.262866], 
    [  0.525731,  0.000000, -0.850651], 
    [  0.681718,  0.147621, -0.716567], 
    [  0.681718, -0.147621, -0.716567], 
    [  0.850651,  0.000000, -0.525731], 
    [  0.809017, -0.309017, -0.500000], 
    [  0.864188, -0.442863, -0.238856], 
    [  0.951056, -0.162460, -0.262866], 
    [  0.147621,  0.716567, -0.681718], 
    [  0.309017,  0.500000, -0.809017], 
    [  0.425325,  0.688191, -0.587785], 
    [  0.442863,  0.238856, -0.864188], 
    [  0.587785,  0.425325, -0.688191], 
    [  0.688191,  0.587785, -0.425325], 
    [ -0.147621,  0.716567, -0.681718], 
    [ -0.309017,  0.500000, -0.809017], 
    [  0.000000,  0.525731, -0.850651], 
    [ -0.525731,  0.000000, -0.850651], 
    [ -0.442863,  0.238856, -0.864188], 
    [ -0.295242,  0.000000, -0.955423], 
    [ -0.162460,  0.262866, -0.951056], 
    [  0.000000,  0.000000, -1.000000], 
    [  0.295242,  0.000000, -0.955423], 
    [  0.162460,  0.262866, -0.951056], 
    [ -0.442863, -0.238856, -0.864188], 
    [ -0.309017, -0.500000, -0.809017], 
    [ -0.162460, -0.262866, -0.951056], 
    [  0.000000, -0.850651, -0.525731], 
    [ -0.147621, -0.716567, -0.681718], 
    [  0.147621, -0.716567, -0.681718], 
    [  0.000000, -0.525731, -0.850651], 
    [  0.309017, -0.500000, -0.809017], 
    [  0.442863, -0.238856, -0.864188], 
    [  0.162460, -0.262866, -0.951056], 
    [  0.238856, -0.864188, -0.442863], 
    [  0.500000, -0.809017, -0.309017], 
    [  0.425325, -0.688191, -0.587785], 
    [  0.716567, -0.681718, -0.147621], 
    [  0.688191, -0.587785, -0.425325], 
    [  0.587785, -0.425325, -0.688191], 
    [  0.000000, -0.955423, -0.295242], 
    [  0.000000, -1.000000,  0.000000], 
    [  0.262866, -0.951056, -0.162460], 
    [  0.000000, -0.850651,  0.525731], 
    [  0.000000, -0.955423,  0.295242], 
    [  0.238856, -0.864188,  0.442863], 
    [  0.262866, -0.951056,  0.162460], 
    [  0.500000, -0.809017,  0.309017], 
    [  0.716567, -0.681718,  0.147621], 
    [  0.525731, -0.850651,  0.000000], 
    [ -0.238856, -0.864188, -0.442863], 
    [ -0.500000, -0.809017, -0.309017], 
    [ -0.262866, -0.951056, -0.162460], 
    [ -0.850651, -0.525731,  0.000000], 
    [ -0.716567, -0.681718, -0.147621], 
    [ -0.716567, -0.681718,  0.147621], 
    [ -0.525731, -0.850651,  0.000000], 
    [ -0.500000, -0.809017,  0.309017], 
    [ -0.238856, -0.864188,  0.442863], 
    [ -0.262866, -0.951056,  0.162460], 
    [ -0.864188, -0.442863,  0.238856], 
    [ -0.809017, -0.309017,  0.500000], 
    [ -0.688191, -0.587785,  0.425325], 
    [ -0.681718, -0.147621,  0.716567], 
    [ -0.442863, -0.238856,  0.864188], 
    [ -0.587785, -0.425325,  0.688191], 
    [ -0.309017, -0.500000,  0.809017], 
    [ -0.147621, -0.716567,  0.681718], 
    [ -0.425325, -0.688191,  0.587785], 
    [ -0.162460, -0.262866,  0.951056], 
    [  0.442863, -0.238856,  0.864188], 
    [  0.162460, -0.262866,  0.951056], 
    [  0.309017, -0.500000,  0.809017], 
    [  0.147621, -0.716567,  0.681718], 
    [  0.000000, -0.525731,  0.850651], 
    [  0.425325, -0.688191,  0.587785], 
    [  0.587785, -0.425325,  0.688191], 
    [  0.688191, -0.587785,  0.425325], 
    [ -0.955423,  0.295242,  0.000000], 
    [ -0.951056,  0.162460,  0.262866], 
    [ -1.000000,  0.000000,  0.000000], 
    [ -0.850651,  0.000000,  0.525731], 
    [ -0.955423, -0.295242,  0.000000], 
    [ -0.951056, -0.162460,  0.262866], 
    [ -0.864188,  0.442863, -0.238856], 
    [ -0.951056,  0.162460, -0.262866], 
    [ -0.809017,  0.309017, -0.500000], 
    [ -0.864188, -0.442863, -0.238856], 
    [ -0.951056, -0.162460, -0.262866], 
    [ -0.809017, -0.309017, -0.500000], 
    [ -0.681718,  0.147621, -0.716567], 
    [ -0.681718, -0.147621, -0.716567], 
    [ -0.850651,  0.000000, -0.525731], 
    [ -0.688191,  0.587785, -0.425325], 
    [ -0.587785,  0.425325, -0.688191], 
    [ -0.425325,  0.688191, -0.587785], 
    [ -0.425325, -0.688191, -0.587785], 
    [ -0.587785, -0.425325, -0.688191], 
    [ -0.688191, -0.587785, -0.425325]
]


class MD2Header:
    """
    Processes an MD2 file and returns mesh data in
    an easy to read and process format.
    """

    def __init__(self, header):
        self.ident = header[0].decode()  
        self.version = header[1]  
        self.skin_width = header[2]          # width of the texture
        self.skin_height = header[3]         # height of the texture
        self.frame_size = header[4]          # size of one frame in bytes
        self.num_skins = header[5]           # number of textures
        self.num_vertices = header[6]        # number of vertices
        self.num_st = header[7]              # number of texture coordinates
        self.num_triangles = header[8]       # number of triangles
        self.num_GLcmd = header[9]           # number of opengl commands
        self.num_frames = header[10]         # total number of frames
        self.offset_skins = header[11]       # offset to skin names (64 bytes each)
        self.offset_st = header[12]          # offset to s-t texture coordinates
        self.offset_triangles = header[13]   # offset to triangles
        self.offset_frames = header[14]      # offset to frame data
        self.offset_GLCmd = header[15]       # offset to opengl commands

class Animate:
    def __init__(self, firstFrame, lastFrame, fps):
        self.firstFrame = firstFrame
        self.lastFrame = lastFrame
        self.fps = fps


animations = {
    'STAND': Animate(0, 39, 9), 
    'RUN': Animate(40,  45, 10), 
    'ATTACK': Animate(46, 53, 10), 
    'PAIN_A': Animate(54, 57, 7),
    'PAIN_B': Animate(58, 61, 7), 
    'PAIN_C': Animate(62, 65, 7), 
    'JUMP': Animate(66, 71, 7), 
    'FLIP': Animate(72, 83, 7),
    'SALUTE': Animate(84, 94, 7), 
    'FALLBACK': Animate(95, 111, 10), 
    'WAVE': Animate(112, 122, 7), 
    'POINT': Animate(123, 134, 6),
    'CROUCH_STAND': Animate(135, 153, 10), 
    'CROUCH_WALK': Animate(154, 159, 7), 
    'CROUCH_ATTACK': Animate(160, 168, 10),
    'CROUCH_PAIN': Animate(169, 172, 7), 
    'CROUCH_DEATH': Animate(173, 177, 5), 
    'DEATH_FALLBACK': Animate(178, 183, 7),
    'DEATH_FALLFORWARD': Animate(184, 189, 7), 
    'DEATH_FALLBACKSLOW': Animate(190, 197, 7), 
    'BOOM': Animate(198, 198, 5)
}


class AnimState:
    def __init__(self, anim, time, p_time, interpol, typeName, currFrame, nextFrame):
        self.startFrame = anim.firstFrame
        self.endFrame = anim.lastFrame
        self.fps = anim.fps
        self.time = time
        self.p_time = p_time
        self.interpol = interpol
        self.type = typeName
        self.currFrame = currFrame
        self.nextFrame = nextFrame

class TexCoord:
    #  texture coords
    def __init__(self, s, t):
        self.s = s
        self.t = t

class Triangle:
    def __init__(self, v, texCoord):
        self.v = v
        self.texCoord = texCoord

class Vertex:
    def __init__(self, v, normal_index):
        self.v = v
        self.normal_index = normal_index

class Frame:
    def __init__(self, name, vertices):
        self.name = name
        self.vertices = vertices

class MD2:
    def __init__(self, anim):
        self.scl = 1.0
        self.texId = 0
        self.setAnimation(anim)

    """
    Loads an MD2 from the specified file.
    """
    def loadModel(self, filename):
        with open(filename, 'rb') as f:            
            data = f.read(68)
            fmt = '4s16i'
            temp = struct.unpack(fmt, data)
            self.header = MD2Header(temp)

            if self.header.ident == 'IPD2' and self.header.version != 8:
                print('MD2 identifier or version is incorrect!')
                f.close()
                return False

            f.seek(self.header.offset_skins)
            data = f.read(self.header.num_skins * 64)
            fmt = self.header.num_skins * '64s'
            self.skins = struct.unpack(fmt, data)

            f.seek(self.header.offset_st)
            data = f.read(self.header.num_st * 4)
            fmt = self.header.num_st * '2h'

            temp = struct.unpack(fmt, data)
            self.texCoords = [TexCoord(temp[i], temp[i+1]) for i in range(0, len(temp), 2)]

            f.seek(self.header.offset_triangles)
            data = f.read(self.header.num_triangles * 12)
            fmt = self.header.num_triangles * '6H'

            temp = struct.unpack(fmt, data)
            self.triangles = [Triangle(temp[i:i+3], temp[i+3:i+6]) for i in range(0, len(temp), 6)]

            f.seek(self.header.offset_GLCmd)
            data = f.read(self.header.num_GLcmd * 4)
            fmt = self.header.num_GLcmd * 'i'
            self.glcmds = struct.unpack(fmt, data)

            self.frames = []

            # Reading all frames
            f.seek(self.header.offset_frames)
            for _ in range(self.header.num_frames):
                data = f.read(40)
                fmt = '3f3f16s'
                temp = struct.unpack(fmt, data)
                
                scl = temp[0:3]
                translate = temp[3:6]
                name = temp[6:22][0]

                data = f.read(self.header.num_vertices * 4)
                fmt = self.header.num_vertices * '3B1B'
                temp = struct.unpack(fmt, data)

                vertices = [Vertex(temp[i:i+3], temp[i+3]) for i in range(0, len(temp), 4)]
 
                # apply the frame translation
                for i, vertex in enumerate(vertices):
                    x = vertex.v[0] * scl[0] + translate[0]
                    y = vertex.v[1] * scl[1] + translate[1]
                    z = vertex.v[2] * scl[2] + translate[2]

                    vertices[i] = Vertex([x, y, z], vertex.normal_index)

                frame = Frame(name, vertices)
                self.frames.append(frame)

            return True

    # define model animation
    def setAnimation(self, anim):
        if not anim in animations:
            print('Animação não encontrada!')
            return

        currFrame = animations[anim].firstFrame
        nextFrame = currFrame + 1
        self.animation = AnimState(animations[anim], 0, 0, 0, anim, currFrame, nextFrame)


    def render(self):
        # reverse the orientation of front-facing
        # polygons because gl command list's triangles
        # have clockwise winding
        glPushAttrib(GL_POLYGON_BIT)
        glFrontFace(GL_CW)

        # Habilitando backface culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # draw each triangle!
        i = 0
        while i < len(self.glcmds):
            cmd = self.glcmds[i]
            i += 1

            if cmd < 0:
                glBegin(GL_TRIANGLE_FAN)
                cmd *= -1
            else:
                glBegin(GL_TRIANGLE_STRIP)

            while cmd > 0:
                s = self.glcmds[i]
                t = self.glcmds[i+1]
                idx = self.glcmds[i+2]

                c_vertex = self.frames[self.animation.currFrame].vertices[idx]
                n_vertex = self.frames[self.animation.nextFrame].vertices[idx]

                c_normal = normals_table[c_vertex.normal_index]
                n_normal = normals_table[n_vertex.normal_index]
                
                x = c_normal[0] + self.animation.interpol * (n_normal[0] - c_normal[0])
                y = c_normal[1] + self.animation.interpol * (n_normal[1] - c_normal[1])
                z = c_normal[2] + self.animation.interpol * (n_normal[2] - c_normal[2])

                glNormal3fv([x, y, z])

                x = c_vertex.v[0] + self.animation.interpol * (n_vertex.v[0] - c_vertex.v[0])
                y = c_vertex.v[1] + self.animation.interpol * (n_vertex.v[1] - c_vertex.v[1])
                z = c_vertex.v[2] + self.animation.interpol * (n_vertex.v[2] - c_vertex.v[2])

                glVertex3fv([x, y, z])

                cmd -= 1
                i += 3

            glEnd()

        glDisable(GL_CULL_FACE)
        glPopAttrib()

    def animateModel(self, time):
        self.animation.time = time

        delta = self.animation.time - self.animation.p_time
        if delta > (1 / self.animation.fps):
            self.animation.currFrame = self.animation.nextFrame
            self.animation.nextFrame += 1

            if self.animation.nextFrame > self.animation.endFrame:
                self.animation.nextFrame = self.animation.startFrame

            self.animation.p_time = self.animation.time

        if self.animation.currFrame > self.header.num_frames - 1:
            self.animation.currFrame = 0

        if self.animation.nextFrame > self.header.num_frames - 1:
            self.animation.nextFrame = 0

        self.animation.interpol = self.animation.fps * (self.animation.time - self.animation.p_time )


    def draw(self, time):
        if time > 0:
            self.animateModel(time)

        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        glRotatef(-90, 0, 0, 1)
        self.render()
        glPopMatrix()


def main():
    if(len(sys.argv) < 3):
        print("Dois argumentos necessarios: nome do arquivo md2 e nome da animação.")
        exit(1)

    Model = MD2(sys.argv[2])

    if not Model.loadModel(sys.argv[1]):
        print('Não foi possível carregar o arquivo', sys.argv[1])
        exit(1)

    if not glfw.init():
        print('Não foi possível inicializar o glfw')
        exit(1)

    window = glfw.create_window(width, height, 'Model', None, None)
    if not window:
        print('Não foi possível criar a janela')
        glfw.terminate()
        exit(1)

    glfw.set_window_pos(window, 30, 60)
    glfw.make_context_current(window)

    shader = gl_shaders.compileProgram(gl_shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                       gl_shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    glUseProgram(shader)

    glClearColor(0.2, 0.2, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)

    lightPositionLoc = glGetUniformLocation(shader, 'lightPos')
    glUniform3f(lightPositionLoc, 100, -100, 0)

    objColorLoc = glGetUniformLocation(shader, 'color')
    glUniform3f(objColorLoc, 1.0, 1.0, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width/height, 0.1, 1000.0)
    glTranslatef(0.0, 0.0, -100.0)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        Model.draw(time.time())
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == '__main__':
    main()