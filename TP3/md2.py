# encoding=utf-8
import os
import sys
import math
import struct
import numpy
from collections import namedtuple, OrderedDict

import glfw
import OpenGL.GL.shaders as gl_shaders
from OpenGL.GL import *
from OpenGL.GLU import *

# from PIL import Image

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

header_layout = namedtuple(
    'MD2_Header',
    [
        'ident',         
        'version',
        'skin_width',       # width of the texture
        'skin_height',      # height of the texture
        'frame_size',       # size of one frame in bytes
        'num_skins',        # number of textures
        'num_vertices',     # number of vertices
        'num_st',           # number of texture coordinates
        'num_tris',         # number of triangles
        'num_glcmds',       # number of opengl commands
        'num_frames',       # total number of frames
        'offset_skins',     # offset to skin names (64 bytes each)
        'offset_st',        # offset to s-t texture coordinates
        'offset_tris',      # offset to triangles
        'offset_frames',    # offset to frame data
        'offset_glcmds',    # offset to opengl commands
        'offset_end'        # offset to end of file
        ]
    )

frame_layout = namedtuple(
    'MD2_Frame',
    [
        'name',
        'vertices',
        'normals'
        ]
    )

triangle_layout = namedtuple(
    'MD2_Triangles',
    [
        'vertex_indices',
        'tc_indices'
        ]
    )

# The MD2 normal look-up table
normals_table = numpy.array(
    [
        [-0.525731, 0.000000, 0.850651 ],
        [-0.442863, 0.238856, 0.864188 ],
        [-0.295242, 0.000000, 0.955423 ],
        [-0.309017, 0.500000, 0.809017 ],
        [-0.162460, 0.262866, 0.951056 ],
        [ 0.000000, 0.000000, 1.000000 ],
        [ 0.000000, 0.850651, 0.525731 ],
        [-0.147621, 0.716567, 0.681718 ],
        [ 0.147621, 0.716567, 0.681718 ],
        [ 0.000000, 0.525731, 0.850651 ],
        [ 0.309017, 0.500000, 0.809017 ],
        [ 0.525731, 0.000000, 0.850651 ],
        [ 0.295242, 0.000000, 0.955423 ],
        [ 0.442863, 0.238856, 0.864188 ],
        [ 0.162460, 0.262866, 0.951056 ],
        [-0.681718, 0.147621, 0.716567 ],
        [-0.809017, 0.309017, 0.500000 ],
        [-0.587785, 0.425325, 0.688191 ],
        [-0.850651, 0.525731, 0.000000 ],
        [-0.864188, 0.442863, 0.238856 ],
        [-0.716567, 0.681718, 0.147621 ],
        [-0.688191, 0.587785, 0.425325 ],
        [-0.500000, 0.809017, 0.309017 ],
        [-0.238856, 0.864188, 0.442863 ],
        [-0.425325, 0.688191, 0.587785 ],
        [-0.716567, 0.681718,-0.147621 ],
        [-0.500000, 0.809017,-0.309017 ],
        [-0.525731, 0.850651, 0.000000 ],
        [ 0.000000, 0.850651,-0.525731 ],
        [-0.238856, 0.864188,-0.442863 ],
        [ 0.000000, 0.955423,-0.295242 ],
        [-0.262866, 0.951056,-0.162460 ],
        [ 0.000000, 1.000000, 0.000000 ],
        [ 0.000000, 0.955423, 0.295242 ],
        [-0.262866, 0.951056, 0.162460 ],
        [ 0.238856, 0.864188, 0.442863 ],
        [ 0.262866, 0.951056, 0.162460 ],
        [ 0.500000, 0.809017, 0.309017 ],
        [ 0.238856, 0.864188,-0.442863 ],
        [ 0.262866, 0.951056,-0.162460 ],
        [ 0.500000, 0.809017,-0.309017 ],
        [ 0.850651, 0.525731, 0.000000 ],
        [ 0.716567, 0.681718, 0.147621 ],
        [ 0.716567, 0.681718,-0.147621 ],
        [ 0.525731, 0.850651, 0.000000 ],
        [ 0.425325, 0.688191, 0.587785 ],
        [ 0.864188, 0.442863, 0.238856 ],
        [ 0.688191, 0.587785, 0.425325 ],
        [ 0.809017, 0.309017, 0.500000 ],
        [ 0.681718, 0.147621, 0.716567 ],
        [ 0.587785, 0.425325, 0.688191 ],
        [ 0.955423, 0.295242, 0.000000 ],
        [ 1.000000, 0.000000, 0.000000 ],
        [ 0.951056, 0.162460, 0.262866 ],
        [ 0.850651,-0.525731, 0.000000 ],
        [ 0.955423,-0.295242, 0.000000 ],
        [ 0.864188,-0.442863, 0.238856 ],
        [ 0.951056,-0.162460, 0.262866 ],
        [ 0.809017,-0.309017, 0.500000 ],
        [ 0.681718,-0.147621, 0.716567 ],
        [ 0.850651, 0.000000, 0.525731 ],
        [ 0.864188, 0.442863,-0.238856 ],
        [ 0.809017, 0.309017,-0.500000 ],
        [ 0.951056, 0.162460,-0.262866 ],
        [ 0.525731, 0.000000,-0.850651 ],
        [ 0.681718, 0.147621,-0.716567 ],
        [ 0.681718,-0.147621,-0.716567 ],
        [ 0.850651, 0.000000,-0.525731 ],
        [ 0.809017,-0.309017,-0.500000 ],
        [ 0.864188,-0.442863,-0.238856 ],
        [ 0.951056,-0.162460,-0.262866 ],
        [ 0.147621, 0.716567,-0.681718 ],
        [ 0.309017, 0.500000,-0.809017 ],
        [ 0.425325, 0.688191,-0.587785 ],
        [ 0.442863, 0.238856,-0.864188 ],
        [ 0.587785, 0.425325,-0.688191 ],
        [ 0.688191, 0.587785,-0.425325 ],
        [-0.147621, 0.716567,-0.681718 ],
        [-0.309017, 0.500000,-0.809017 ],
        [ 0.000000, 0.525731,-0.850651 ],
        [-0.525731, 0.000000,-0.850651 ],
        [-0.442863, 0.238856,-0.864188 ],
        [-0.295242, 0.000000,-0.955423 ],
        [-0.162460, 0.262866,-0.951056 ],
        [ 0.000000, 0.000000,-1.000000 ],
        [ 0.295242, 0.000000,-0.955423 ],
        [ 0.162460, 0.262866,-0.951056 ],
        [-0.442863,-0.238856,-0.864188 ],
        [-0.309017,-0.500000,-0.809017 ],
        [-0.162460,-0.262866,-0.951056 ],
        [ 0.000000,-0.850651,-0.525731 ],
        [-0.147621,-0.716567,-0.681718 ],
        [ 0.147621,-0.716567,-0.681718 ],
        [ 0.000000,-0.525731,-0.850651 ],
        [ 0.309017,-0.500000,-0.809017 ],
        [ 0.442863,-0.238856,-0.864188 ],
        [ 0.162460,-0.262866,-0.951056 ],
        [ 0.238856,-0.864188,-0.442863 ],
        [ 0.500000,-0.809017,-0.309017 ],
        [ 0.425325,-0.688191,-0.587785 ],
        [ 0.716567,-0.681718,-0.147621 ],
        [ 0.688191,-0.587785,-0.425325 ],
        [ 0.587785,-0.425325,-0.688191 ],
        [ 0.000000,-0.955423,-0.295242 ],
        [ 0.000000,-1.000000, 0.000000 ],
        [ 0.262866,-0.951056,-0.162460 ],
        [ 0.000000,-0.850651, 0.525731 ],
        [ 0.000000,-0.955423, 0.295242 ],
        [ 0.238856,-0.864188, 0.442863 ],
        [ 0.262866,-0.951056, 0.162460 ],
        [ 0.500000,-0.809017, 0.309017 ],
        [ 0.716567,-0.681718, 0.147621 ],
        [ 0.525731,-0.850651, 0.000000 ],
        [-0.238856,-0.864188,-0.442863 ],
        [-0.500000,-0.809017,-0.309017 ],
        [-0.262866,-0.951056,-0.162460 ],
        [-0.850651,-0.525731, 0.000000 ],
        [-0.716567,-0.681718,-0.147621 ],
        [-0.716567,-0.681718, 0.147621 ],
        [-0.525731,-0.850651, 0.000000 ],
        [-0.500000,-0.809017, 0.309017 ],
        [-0.238856,-0.864188, 0.442863 ],
        [-0.262866,-0.951056, 0.162460 ],
        [-0.864188,-0.442863, 0.238856 ],
        [-0.809017,-0.309017, 0.500000 ],
        [-0.688191,-0.587785, 0.425325 ],
        [-0.681718,-0.147621, 0.716567 ],
        [-0.442863,-0.238856, 0.864188 ],
        [-0.587785,-0.425325, 0.688191 ],
        [-0.309017,-0.500000, 0.809017 ],
        [-0.147621,-0.716567, 0.681718 ],
        [-0.425325,-0.688191, 0.587785 ],
        [-0.162460,-0.262866, 0.951056 ],
        [ 0.442863,-0.238856, 0.864188 ],
        [ 0.162460,-0.262866, 0.951056 ],
        [ 0.309017,-0.500000, 0.809017 ],
        [ 0.147621,-0.716567, 0.681718 ],
        [ 0.000000,-0.525731, 0.850651 ],
        [ 0.425325,-0.688191, 0.587785 ],
        [ 0.587785,-0.425325, 0.688191 ],
        [ 0.688191,-0.587785, 0.425325 ],
        [-0.955423, 0.295242, 0.000000 ],
        [-0.951056, 0.162460, 0.262866 ],
        [-1.000000, 0.000000, 0.000000 ],
        [-0.850651, 0.000000, 0.525731 ],
        [-0.955423,-0.295242, 0.000000 ],
        [-0.951056,-0.162460, 0.262866 ],
        [-0.864188, 0.442863,-0.238856 ],
        [-0.951056, 0.162460,-0.262866 ],
        [-0.809017, 0.309017,-0.500000 ],
        [-0.864188,-0.442863,-0.238856 ],
        [-0.951056,-0.162460,-0.262866 ],
        [-0.809017,-0.309017,-0.500000 ],
        [-0.681718, 0.147621,-0.716567 ],
        [-0.681718,-0.147621,-0.716567 ],
        [-0.850651, 0.000000,-0.525731 ],
        [-0.688191, 0.587785,-0.425325 ],
        [-0.587785, 0.425325,-0.688191 ],
        [-0.425325, 0.688191,-0.587785 ],
        [-0.425325,-0.688191,-0.587785 ],
        [-0.587785,-0.425325,-0.688191 ],
        [-0.688191,-0.587785,-0.425325 ]
    ],
    dtype = numpy.float
    )


"""
Processes an MD2 file and returns mesh data in
an easy to read and process format.
"""
class MD2:
    def __init__(self):
        self.header = None
        self.skins = None
        self.triangles = None
        self.tcs = None
        self.frames = None
        self.text_id = None

    def load(self, filename):
        """
        Reads the MD2 data from the existing
        specified filename.
        @param filename: the filename of the md2 file
        to load.
        """
        with open(filename, 'rb') as f:
            self.load_from_buffer(f)
    
    def load_from_buffer(self, f):
        """
        Reads the MD2 data from a stream object.
        Can be called instead of load() if data
        is not present in a file.
        @param f: the stream object, usually a file.
        """
        # read all the data from the file
        self.header = self.read_header(f)
        self.skins = self.read_skins(f, self.header)
        self.tcs = self.read_texture_coordinates(f, self.header) 
        self.triangles = self.read_triangles(f, self.header)
        self.frames = self.read_frames(f, self.header)

    @staticmethod
    def _load_block( stream, format, count ):
        """
        Convenience method used to load blocks of
        data using the python 'struct' object format.
        Loads 'count' blocks from the file, each block
        will have the python struct format defined by 'format'.
        This is handy for loading large blocks without having
        to manually iterate over it.
        @param stream: the file object.
        @param format: the python 'struct' format of the block.
        @param count: the number of blocks to load.
        """
        def chunks( data, size ):
            """
            Return a generator that yields next 'size' bytes from data
            """
            offset = 0
            while offset < len( data ):
                yield data[ offset: offset + size ]
                offset += size

        struct_length = struct.calcsize( format )
        total_length = struct_length * count
        data = stream.read( total_length )

        if(len(data) < total_length):
            raise ValueError( "MD2: Failed to read '%d' bytes" % (total_length) )

        return [ struct.unpack(format, chunk) for chunk in chunks(data, struct_length) ]

    @staticmethod
    def read_header( f ):
        """
        Reads the MD2 header information from the MD2 file.
        @param f: the file object.
        @return Returns an header_layout named tuple.
        """
        # read the header
        # header is made up of 17 signed longs
        # this first is the ID which is also a 4 byte string
        header = header_layout._make(
            MD2._load_block( f, '< 4s16l', 1 )[ 0 ]
            )

        if header.ident != id:
            raise ValueError(
                "MD2 identifier is incorrect, expected '%i', found '%i'" % (
                    id,
                    header.ident
                    )
                )
        if header.version != version:
            raise ValueError(
                "MD2 version is incorrect, expected '%i', found '%i'" % (
                    version,
                    header.version
                    )
                )

        return header

    @staticmethod
    def read_skins(f, header):
        """
        Reads the skin filenames out of the MD2 header.
        @param f: the file object.
        @param header: the loaded MD2 header.
        @return: Returns a python list of skin filenames.
        The list is a 1D list of size header.num_skins.
        """
        # seek to the skins offset
        f.seek( header.offset_skins, os.SEEK_SET )

        # skins are stored as a list of 64 signed byte strings
        # each string is a path relative to /baseq2
        skin_struct = struct.Struct( '< %s' % ('64s' * header.num_skins) )

        # read the skins and convert to list
        # strip any \x00 characters while we're at it
        # because python gets confused by them
        return [ skin.rstrip('\x00') for skin in skin_struct.unpack( f.read( skin_struct.size ) ) ]

    @staticmethod
    def read_texture_coordinates(f, header):
        """
        Reads the texture coordinates from the MD2 file.
        @param f: the file object.
        @param header: the loaded MD2 header.
        @return: Returns a numpy array containing the texture
        coordinates. Values are converted from original
        absolute texels (0->width,0->height) to openGL
        coordinates (0.0->1.0).
        The array is an Nx2 dimension array where N
        is header.num_st.
        """
        # seek to the skins offset
        f.seek( header.offset_st, os.SEEK_SET )

        # st's are stored in a contiguous array of 2 short values
        # TCs do NOT map directly to vertices.
        # 1 vertex can have multiple TCs (one TC for each poly)
        # TCs are composed of 2 signed shorts
        tcs = numpy.array(
            MD2._load_block( f, '< 2h', header.num_st ),
            dtype = numpy.float
            )
        tcs.shape = (-1, 2)

        # convert from texel values to 0->1 float range
        tcs /= [ float(header.skin_width), float(header.skin_height) ]
        return tcs

    @staticmethod
    def read_triangles(f, header):
        """
        Reads the triangle information from the MD2 file.
        Triangle information includes the vertex and
        texture coordinate indices.
        @param f: the file object.
        @param header: the loaded MD2 header.
        @return: Returns an MD2 named tuple.
        The vertex and texture coordinate indices are
        arrays of Nx3 dimensions for vertices and Nx2 for
        texture coordinates where N is header.num_tris.
        """
        # seek to the triangles offset
        f.seek( header.offset_tris, os.SEEK_SET )

        # triangles are stored as 3 unsigned shorts for the vertex indices
        # and 3 unsigned shorts for the texture coordinates indices
        triangles = numpy.array(
            MD2._load_block( f, '< 6H', header.num_tris ),
            dtype = numpy.uint16
            )
        triangles.shape = (-1, 6)

        # extract the vertex indices and tcs
        vertex_indices = triangles[ : , :3 ]
        tc_indices = triangles[ : , 3: ]

        # md2 triangles are clock-wise, we need to change
        # them to counter-clock-wise
        vertex_indices[ :,[1,2] ] = vertex_indices[ :,[2,1] ]
        tc_indices[ :,[1,2] ] = tc_indices[ :,[2,1] ]

        vertex_indices = vertex_indices.flatten()
        tc_indices = tc_indices.flatten()

        return triangle_layout(
            vertex_indices,
            tc_indices
            )

    @staticmethod
    def read_frames(f, header):
        """
        Reads all frames from the MD2 file.
        This function simply calls read_frame in a loop.
        @param f: the file object.
        @param header: the loaded MD2 header.
        @return returns a python list of frame_layout
        named tuples. The list will be of length
        header.num_frames.
        """
        # seek to the frames offset
        f.seek( header.offset_frames, os.SEEK_SET )
        return [ MD2.read_frame( f, header ) for x in xrange( header.num_frames ) ]

    @staticmethod
    def read_frame(f, header):
        """
        Reads a frame from the MD2 file.
        The stream must already be at the start of the
        frame.
        @param f: the file object.
        @param header: the loaded MD2 header.
        @return: Returns an frame_layout named tuple.
        Returned vertices and normals are as read from
        the file and are not ready to render.
        To render these must be ordered according to
        the indices specified in the triangle information.
        @see convert_indices_for_all_frames
        @see convert_indices_for_frame
        """
        # frame scale and translation are 2x3 32 bit floats
        frame_translations = numpy.array(
            MD2._load_block( f, '< 3f', 2 ),
            dtype = numpy.float
            )
        # extract the scale and translation vector
        scale = frame_translations[0]
        translation = frame_translations[1]

        # read the frame name
        # frame name is a 16 unsigned byte string
        name, = MD2._load_block( f, '< 16s', 1 )[0]
        # remove any \x00 characters as they confuse python
        name = name.strip( '\x00' )

        # frame has 3 unsigned bytes for the vertex coordinates
        # and 1 unsigned byte for the normal index
        frame_vertex_data = numpy.array(
            MD2._load_block( f, '<4B', header.num_vertices ),
            dtype = numpy.uint8
            )
        frame_vertex_data.shape = (-1, 4)

        # extract the vertex values
        vertices_short = frame_vertex_data[ :, :3 ]
        vertices = vertices_short.astype( numpy.float )
        vertices.shape = (-1, 3)

        # apply the frame translation
        vertices *= scale
        vertices += translation

        # re-orient the mesh
        # md2's have +Z as up, +Y as left, +X as forward
        # up: +Z, left: +Y, forward: +X
        # we want
        # up: +Y, left: -X, forward: -Z
        vertices[:,0],vertices[:,1],vertices[:,2] = \
            -vertices[:,1],vertices[:,2],-vertices[:,0]

        # extract the normal values
        normal_indices = frame_vertex_data[ :, 3 ]
        # convert from normal indice to normal vector
        normals = normals_table[ normal_indices ]
        normals.shape = (-1, 3)

        return frame_layout(
            name,
            vertices,
            normals
            )

    def render(self, n):
        if n < 0 or n > self.header.num_frames - 1:
            return

        # Obtendo o frame desejado
        frame = self.frames[n]

        # glBindTexture(GL_TEXTURE_2D, self.text_id)

        # Desenhando o modelo
        glBegin(GL_TRIANGLES)
        for i in range(self.header.num_tris):
            # Iterando sobre cada vértice
            for j in range(3):
                # Compute texture coordinates
                s = self.tcs[self.triangles[i].tc_indices[j]] / self.header.skin_width
                t = self.tcs[self.triangles[i].tc_indices[j]] / self.header.skin_height

                # /* Pass texture coordinates to OpenGL */
                glTexCoord2f(s, t)

                glNormal3fv(frame.normals)

                # Calculando a posição real do vértice
                x = frame.vertices[0]
                y = frame.vertices[1]
                z = frame.vertices[2]

                # Desenhando aquele vértice
                glVertex3f(x, y, z)
        glEnd()

    def draw(self):
        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        glRotatef(-90, 0, 0, 1)
        self.render(0)
        glPopMatrix()

# main method 
def main():
    # create model
    model_ = MD2()

    # Carregando o modelo
    model_.load(sys.argv[1])
     
    # Inicializando o glfw
    if not glfw.init():
        print('Não foi possível inicializar o glfw')
        exit(1)

    # Criando uma janela
    window = glfw.create_window(width, height, 'Modelo', None, None)
    if not window:
        print('Não foi possível criar a janela')
        glfw.terminate()
        exit(1)

    # Definindo a posição da janela na tela
    glfw.set_window_pos(window, 30, 60)

    # Tornando a janela criada o contexto atual
    glfw.make_context_current(window)

    # Compilando e usando o shader
    shader = gl_shaders.compileProgram(gl_shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                       gl_shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    glUseProgram(shader)

    # Definindo a cor de limpeza da tela
    glClearColor(0.2, 0.2, 0.2, 1.0)

    # Habilitando o DEPTH_TEST
    glEnable(GL_DEPTH_TEST)

    # Definindo a posição da nossa luz
    lightPosLoc = glGetUniformLocation(shader, 'lightPos')
    glUniform3f(lightPosLoc, 100, -100, 0)

    # Definindo a cor do nosso objeto
    colorLoc = glGetUniformLocation(shader, 'color')
    glUniform3f(colorLoc, 1.0, 1.0, 1.0)

    # Definindo a cor de limpeza da tela
    glClearColor(0.2, 0.2, 0.2, 1.0)

    # Habilitando o DEPTH_TEST
    glEnable(GL_DEPTH_TEST)

    # Definindo o tipo de projeção
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width/height, 0.1, 500.0)

    # Transladando o modelo para aparecer na tela
    glTranslatef(0.0, 0.0, -100.0)

    # MAIN LOOP
    while not glfw.window_should_close(window):
        # Capturando eventos
        glfw.poll_events()

        # Limpando os buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        model_.draw()

        # Desenhando na tela
        glfw.swap_buffers(window)

    # Terminando o glfw
    glfw.terminate()

# Chamando a função principal
if __name__ == '__main__':
    main()