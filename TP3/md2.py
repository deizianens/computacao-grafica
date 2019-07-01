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
from OpenGL.error import GLError
from OpenGL import GL

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

# List of frame types used by Quake 2
# http://tfc.duke.free.fr/old/models/md2.htm
animations = OrderedDict([
    ('stand',                (0, 39, 9.0)),
    ('run',                  (40, 45, 10.0)),
    ('attack',               (46, 53, 10.0)),
    ('pain_a',               (54, 57, 7.0)),
    ('pain_b',               (58, 61, 7.0)),
    ('pain_c',               (62, 65, 7.0)),
    ('jump',                 (66, 71, 7.0)),
    ('flip',                 (72, 83, 7.0)),
    ('salute',               (84, 94, 7.0)),
    ('fallback',             (95, 111, 10.0)),
    ('wave',                 (112, 122, 7.0)),
    ('point',                (123, 134, 6.0)),
    ('crouch_stand',         (135, 153, 10.0)),
    ('crouch_walk',          (154, 159, 7.0)),
    ('crouch_attack',        (160, 168, 10.0)),
    ('crouch_pain',          (169, 172, 7.0)),
    ('crouch_death',         (173, 177, 5.0)),
    ('death_fallback',       (178, 183, 7.0)),
    ('death_fallforward',    (184, 189, 7.0)),
    ('death_fallbackslow',   (190, 197, 7.0)),
    ('boom',                 (198, 198, 5.0)),
    ])

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
        # self.text_id = None

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
        f.seek(header.offset_skins, os.SEEK_SET)

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
        f.seek(header.offset_st, os.SEEK_SET)

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
        f.seek(header.offset_tris, os.SEEK_SET)

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

        frame = self.frames[n]

        # Draw model
        glBegin(GL_TRIANGLES)
        for i in range(self.header.num_tris):
            for j in range(3):
                # Compute texture coordinates
                s = self.tcs[0]
                t = self.tcs[1]

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

class ShaderProgram(object):
    """Defines a complete Shader Program, consisting of at least
    a vertex and fragment shader.
    
    Shader objects are decoupled from ShaderPrograms to avoid recompilation
    when re-using shaders.
    Multiple shaders of the same type can be attached together.
    This lets you combine multiple smaller shaders into a single larger one.
    kwargs supported arguments:
        link_now: defaults to True. If set to True, the shader will be linked
            during the constructor.
        raise_invalid_variables:    defaults to False. If set to True,
            accessing invalid Uniforms and Attributes will trigger a
            ValueError exception to be raised.
    """
    
    def __init__( self, *args, **kwargs ):
        super(ShaderProgram, self).__init__()

        # create the program handle
        self._handle = GL.glCreateProgram()

        # store our attribute and uniform handler
        self.attributes = Attributes( self )
        self.uniforms = Uniforms( self )

        for shader in args:
            self.attach_shader(shader)

        # default raise exception to False
        self.raise_invalid_variables = kwargs.get(
            'raise_invalid_variables',
            False
            )

        # default link now to True
        link_now = kwargs.get( 'link_now', True )

        if link_now:
            self.link()

    @property
    def handle( self ):
        return self._handle

    def attach_shader( self, shader ):
        """Attaches a Shader object.
        This expects an instance of the Shader class (or equivalent).
        If you need to attach a normal GL shader handle, use the
        Shader.create_from_existing class method to instantiate a
        Shader object first.
        """
        try:
            # attach the shader
            GL.glAttachShader( self.handle, shader.handle )
        except Exception as e:
            print( "\tException: %s" % str(e) )

            # chain the exception
            raise

    def frag_location( self, name, buffers = 0 ):
        """Sets the fragment output name used within the program.
        Buffers is the number of buffers receiving the fragment output.
        By default, this is 0.
        Frag data locations MUST be bound BEFORE linking
        or the location will not take effect until
        the shader is linked again!
        http://www.opengl.org/sdk/docs/man3/xhtml/glBindFragDataLocation.xml
        """
        GL.glBindFragDataLocation(self.handle, buffers, name)

    def link( self ):
        """Links the specified shader into a complete program.
        It is important to set any attribute locations and
        the frag data location BEFORE calling link or these calls
        will not take effect.
        """
        # link the program
        try:
            GL.glLinkProgram( self.handle )
        except GLError as e:
            self._print_shader_errors( e.description )
            raise

        # retrieve the compile status
        if not GL.glGetProgramiv( self.handle, GL.GL_LINK_STATUS ):
            errors = GL.glGetProgramInfoLog( self.handle )
            self._print_shader_errors( errors )

            raise GLError( errors )

        self.uniforms._on_program_linked()
        self.attributes._on_program_linked()

    def _print_shader_errors( self, buffer ):
        """Parses the error buffer and prints it to the console.
        The buffer should be the exact contents of the GLSL
        error buffer converted to a Python String.
        """
        print( "Error linking shader:" )
        print( "\tDescription: %s" % ( buffer ) )

        # print the log to the console
        # errors = parse_shader_errors( buffer )

        # for error in errors:
            # line, desc = error

            # print( "Error linking shader" )
            # print( "\tDescription: %s" % ( desc ) )

    @property
    def linked( self ):
        """Returns the link status of the shader.
        """
        return GL.glGetProgramiv( self.handle, GL.GL_LINK_STATUS ) == GL.GL_TRUE

    def bind( self ):
        """Binds the shader program to be the active shader program.
        The shader MUST be linked for this to be valid.
        It is valid to bind one shader after another without calling
        unbind.
        """
        # bind the program
        GL.glUseProgram( self.handle )

    def unbind( self ):
        """Unbinds the shader program.
        This sets the current shader to null.
        It is valid to bind one shader after another without calling
        unbind.
        Be aware that this will NOT unwind the bind calls.
        Calling unbind will set the active shader to null.
        """
        # unbind the
        GL.glUseProgram( 0 )

    @property
    def bound( self ):
        """Returns True if the program is the currently bound program
        """
        return GL.glGetIntegerv( GL.GL_CURRENT_PROGRAM ) == self.handle

    def __getitem__(self, name):
        """Return the Uniform or Attribute with the specified name.
        """
        if name in self.uniforms.all():
            return self.uniforms[ name ]
        elif name in self.attributes.all():
            return self.attributes[ name ]
        else:
            raise KeyError( name )

    def __str__( self ):
        string = "%s(uniforms=[%s], attributes=[%s])" % (
            self.__class__.__name__,
            str( self.uniforms ),
            str( self.attributes )
            )
        return string

class Shader(object):
    """An individual Shader object.
    Used as part of a single ShaderProgram object.
    A vertex shader (GL_VERTEX_SHADER) and a fragment shader (GL_FRAGMENT_SHADER)
    must be used as part of a single Shader Program.
    Geometry shaders (GL_GEOMETRY_SHADER) are optional.
    Shaders can be used by multiple `py:class:pygly.shader.ShaderProgram`.
    Multiple shaders of the same type can be attached to a ShaderProgram.
    The GLSL linker will over-write any existing functions with the same signature
    with functions from the newly attached shader.
    """

    @classmethod
    def create_from_existing( cls, type, source, handle, compile_now = True ):
        """Creates a Shader object using an existing shader handle
        """
        obj = cls( type, source, False )
        obj._handle = handle
        if compile_now:
            obj.compile()
        return obj

    @classmethod
    def create_from_file( cls, type, filename, compile_now = True ):
        with open(filename) as f:
            return cls( type, f.readlines(), compile_now )

    def __init__( self, type, source, compile_now = True ):
        super( Shader, self ).__init__()

        self._handle = None
        self._type = type
        self._source = source

        if compile_now:
            self.compile()

    @property
    def handle( self ):
        return self._handle

    @property
    def type( self ):
        return self._type

    @property
    def source( self ):
        return self._source

    def compile( self ):
        """Compiles the shader using the current content
        value.
        Shaders are required to be compiled before a
        `py:class:pygly.shader.ShaderProgram` can be linked.
        This is not required to be performed in order to
        attach a Shader to a ShaderProgram. As long as the
        Shader is compiled prior to the ShaderProgram being
        linked.
        """
        self._handle = GL.glCreateShader( self.type )

        GL.glShaderSource( self.handle, self.source )

        # compile the shader
        try:
            GL.glCompileShader( self.handle )
        except GLError as e:
            self._print_shader_errors( e.description )
            raise

        # retrieve the compile status
        if not GL.glGetShaderiv( self.handle, GL.GL_COMPILE_STATUS ):
            errors = GL.glGetShaderInfoLog( self.handle )
            self._print_shader_errors( errors )

            raise GLError( errors )


    def _print_shader_errors( self, buffer ):
        """Parses the error buffer and prints it to the console.
        The buffer should be the exact contents of the GLSL
        error buffer converted to a Python String.
        """
        # print the log to the console
        # errors = parse_shader_errors( buffer )
        # lines = self.source.split('\n')

        # for error in errors:
        #     line, desc = error

        #     print( "Error compiling shader type: %s" % enum_to_string( self.type ) )
        #     print( "\tLine: %i" % line )
        #     print( "\tDescription: %s" % desc )
        #     print( "\tCode: %s" % lines[ line - 1 ] )

    def __str__( self ):
        string = "%s(type=%s)" % (
            self.__class__.__name__,
            self.type 
            )
        return string

class Attributes(object):
    """Provides access to `py:class:pygly.shader.ShaderProgram` attribute bindings.
    Because Attributes must be updated before the shader is linked,
    we cannot do the same validation as we can with Uniforms.
    Attributes are accessed using array semantics::
        shader.attributes[ 'in_position' ] = 0
        print( shader.attributes[ 'in_position' ] )
        >>> 0
    Attributes provides a mechanism to iterate over the active Attributes::
        for attribute in shader.attributes:
            print( attribute )
    """

    def __init__( self, program ):
        super( Attributes, self ).__init__()

        self._program = program
        self._attributes = {}

    @property
    def program( self ):
        return self._program

    def _on_program_linked( self ):
        self._attributes = dict(
            (name, Attribute( self.program, name ))
            for (name, size, type) in attributes(self._program.handle)
            )

    def __iter__(self):
        return self.next()

    def next(self):
        for attribute in self.all().values():
            yield attribute

    def all(self):
        """Returns a dictionary of all the available attributes.
        The key is the attribute name.
        The value is an Attribute object.
        """
        # get number of active attributes
        return self._attributes.copy()

    def __getitem__(self, name):
        """Returns the currently bound attribute value.
        The ShaderProgram MUST be linked or a ValueError is raised.
        """
        if name not in self._attributes:
            self._attributes[ name ] = Attribute( self.program, name )
        return self._attributes[ name ]

    def __setitem__(self, name, value):
        """Sets the location of the shader's attribute.
        Passes the value to the attribute's location.
        This lets us just call 'shader.attributes['variable'] = value'
        This value can be set at any time on the `py:class:pygly.shader.ShaderProgram`,
        but it will only take effect the next time the ShaderProgram is linked.
        """
        self[name].location = value

    def __str__( self ):
        string = "%s(" % (self.__class__.__name__)

        for attribute in self:
            string += str(attribute) + ", "

        return string[:-2] + ")"

class Attribute(object):
    """Wraps a GLSL Vertex Attribute.
    """

    def __init__( self, program, name ):
        super( Attribute, self ).__init__()

        self._program = program
        self._name = name

    @property
    def name( self ):
        """Returns the name of the uniform as specified in GLSL.
        Eg. 'in_position'
        """
        return self._name

    @property
    def program( self ):
        """Returns the ShaderProgram that owns the Uniform.
        """
        return self._program

    @property
    def type( self ):
        """Returns the GL enumeration type for the Attribute.
        Eg. GL_FLOAT_VEC4.
        :rtype: GL enumeration or None if invalid.
        """
        attribute = attribute_for_name( self.program.handle, self.name )
        if attribute:
            return attribute[ 2 ]
        return None

    @property
    def location( self ):
        """Returns the location of the Attribute.
        """
        return GL.glGetAttribLocation( self.program.handle, self.name )

    @location.setter
    def location( self, location ):
        """Sets the attributes location.
        """
        GL.glBindAttribLocation( self.program.handle, location, self.name )

    def __str__( self ):
        """Returns a human readable string representing the Attribute.
        """
        return "%s(name=%s, type=%s, location=%d)" % (
            self.__class__.__name__,
            self.name,
            self.type,
            self.location
            )

class Uniform(object):
    """Provides the base class for access to uniform variables.
    """

    def __init__( self, types, dtype ):
        """Creates a new Uniform object.
        This should only be called by inherited Uniform classes.
        Types is a dictionary with the following format:
            key: GL enumeration type as a string, Eg. 'GL_FLOAT_VEC4'.
            value: (uniform setter function, number of values per variable)
        The function is used when setting the uniform value.
        The number of values per variable is used to determine the number of
        variables passed to a uniform.
        Ie. Numver of variables = number of values / values per variable
        """
        super( Uniform, self ).__init__()

        self._types = _generate_enum_map( types )
        self._dtype = dtype

        # these values are set in _set_data which is called by
        # shader.uniforms when an assignment is made
        # this allows users to create uniforms and assign them to
        # a shader
        self._program = None
        self._name = None
        self._type = None
        self._func = None
        self._num_values = None
        self._location = None

    @property
    def name( self ):
        """Returns the name of the uniform as specified in GLSL.
        Eg. in_texture_diffuse
        """
        return self._name

    @property
    def program( self ):
        """Returns the ShaderProgram that owns the Uniform.
        """
        return self._program

    @property
    def location( self ):
        """Returns the location of the Uniform.
        """
        return self._location

    @property
    def type( self ):
        """Returns the GL enumeration type for the Uniform.
        Eg. GL_FLOAT_VEC4.
        """
        return self._type

    @property
    def dtype( self ):
        """Returns the numpy dtype string that represents this Uniform type.
        Eg. "float32"
        """
        return self._dtype

    @property
    def data_size( self ):
        """Returns the number of values that make up a single Uniform.
        Eg, for vec4, this would be 4.
        """
        return self._num_values

    def _set_data( self, program, name, type ):
        """Used by the `py:class:pygly.shader.Uniform` class to pass the data to the Uniform
        object once it is assigned to a ShaderProgram.
        """
        self._program = program
        self._name = name
        self._type = type

        if not self.program.linked:
            raise ValueError( "ShaderProgram must be linked before uniform can be set" )

        # ensure we have the right uniform type
        if self.type not in self._types:
            raise ValueError(
                "Uniform '%s' has type '%s' and is not supported by %s" % (
                    self.name,
                    self.type,
                    self.__class__.__name__
                    )
                )

        self._func, self._num_values = self._types[ self.type ]

        # set our location
        self._location = GL.glGetUniformLocation( self.program.handle, self.name )

    @property
    def value( self ):
        """Retrieves the current value of the Uniform.
        .. warning:: Not currently implemented
        """
        raise NotImplementedError

    @value.setter
    def value( self, *args ):
        """Assigns a value to the Uniform.
        """
        if not self.program.bound:
            raise ValueError( "ShaderProgram must be bound before uniform can be set" )

        values = numpy.array( args, dtype = self._dtype )

        # check we received the correct number of values
        if 0 != (values.size % self._num_values):
            raise ValueError(
                "Invalid number of values for Uniform, expected multiple of: %d, received: %d" % (
                    self._num_values,
                    values.size
                    )
                )

        count = values.size / self._num_values
        self._func( self.location, count, values )

    def __str__( self ):
        """Returns a human readable string representing the Uniform.
        """
        return "%s(name=%s, type=%s, location=%d)" % (
            self.__class__.__name__,
            self.name,
            self.type,
            self.location
            )
            
class Uniforms(object):
    """Provides access to `py:class:pygly.shader.ShaderProgram` uniform variables.
    Uniforms are accessed using array semantics::
        shader.uniforms[ 'model_view' ] = 0
        print( shader.uniforms[ 'model_view' ] )
        >>> 0
    Uniforms provides a mechanism to iterate over the active Uniforms::
        for uniform in shader.uniforms:
            print( uniform )
    """

    """This dictionary holds a list of GL shader enum types.
    Each type has a corresponding Uniform class.
    When processing uniforms, the appropriate class is instantiated
    for the specific time.
    The values are populated by calling
    `py:func:pygly.shader._register_uniform_class`.
    """
    types = {}

    @staticmethod
    def register_uniform_class( cls, types ):
        """Registers a Uniform class to be used for specific GLSL GL types.
        class_type is a class type, such as UniformFloat.
        types is a list of GL enumeration types as strings that the class
        is to be used for.
        For example::
            ['GL_FLOAT_VEC4', 'GL_SAMPLER_1D']
        There is no checking for duplicates, latter calls to this function can over-ride
        existing class registrations.
        """
        for type in types:
            # add to dictionary
            # check if the type is valid
            try:
                Uniforms.types[ getattr(GL, type) ] = cls
            except AttributeError:
                pass

    def __init__( self, program ):
        super( Uniforms, self ).__init__()

        self._program = program
        self._uniforms = {}

    @property
    def program( self ):
        return self._program

    def __iter__( self ):
        return self.next()

    def next( self ):
        for uniform in self.all().values():
            yield uniform

    def _on_program_linked( self ):
        """Called by a ShaderProgram when the program is linked
        successfully.
        """
        # get our active uniforms
        program = self.program
        self._uniforms = {}
        for name, size, type in uniforms( program.handle ):
            self._uniforms[ name ] = self.types[ type ]()
            self._uniforms[ name ]._set_data( program, name, type )

    def all( self ):
        """Returns a dictionary of all uniform objects.
        The key is the uniform name.
        The value is the uniform type as a string.
        Any uniform automatically detected or accessed programmatically
        in python will appear in this list.
        """
        # convert to a list
        return self._uniforms.copy()

    def __getitem__( self, name ):
        """Returns an appropriate uniform for the specified variable name.
        This variable name matches the uniform specified in the shader.
        The ShaderProgram MUST be linked or a ValueError is raised.
        """
        if not self.program.linked:
            raise ValueError( "ShaderProgram must be linked before attribute can be queried" )

        # check if a uniform already exists
        if name in self._uniforms:
            # return the existing uniform
            return self._uniforms[ name ]
        else:
            # the uniform doesn't exit
            # check if we should raise an exception
            # if not, create an InvalidUniform object and store it
            # this means it will only print a log message this one time
            if self.program.raise_invalid_variables:
                raise ValueError( "Uniform '%s' not specified in ShaderProgram" % name )
            else:
                # we shouldn't raise an exception
                # so create an invalid uniform object that will do nothing
                self._uniforms[ name ] = InvalidUniform()
                self._uniforms[ name ]._set_data( self.program, name, type = None )
                return self._uniforms[ name ]

    def __setitem__( self, name, value ):
        """Sets the value of the shader's uniform.
        This lets us just call 'shader.uniforms['variable'] = value'
        """
        self[ name ].value = value

    def __str__( self ):
        string = "%s(" % (self.__class__.__name__)

        for uniform in self:
            string += str(uniform) + ", "
        string = string[:-2] + ")"

        return string

class Data:
    _data = {}

    @classmethod 
    def load(cls, filename): 
        # check if the model has been loaded previously 
        if filename in Data._data: 
            # create a new mesh with the same data 
            return Data._data[ filename ]

        data = cls( filename ) 

        # store mesh for later 
        Data._data[ filename ] = data

        return data

    @classmethod
    def unload( cls, filename ):
        if filename in Data._data:
            del Data._data[ filename ]

    def __init__( self, filename = None, buffer = None ):
        """
        Loads an MD2 from the specified file.
        @param filename: the filename to load the mesh from.
        @param interpolation: the number of frames to generate
        between each loaded frame.
        0 is the default (no interpolation).
        It is suggested to keep the value low (0-2) to avoid
        long load times.
        """
        super(Data, self).__init__()
        
        self.frames = None
        self.vao = None
        self.tc_vbo = None
        self.indice_vbo = None

        self.shader = ShaderProgram(
            Shader( GL_VERTEX_SHADER, Data.shader_source['vert'] ),
            Shader( GL_FRAGMENT_SHADER, Data.shader_source['frag'] ),
            link_now = False
            )

        # set our shader data
        # we MUST do this before we link the shader
        self.shader.attributes.in_position_1 = 0
        self.shader.attributes.in_normal_1 = 1
        self.shader.attributes.in_position_2 = 2
        self.shader.attributes.in_normal_2 = 3
        self.shader.attributes.in_texture_coord = 4

        self.shader.frag_location( 'out_frag_colour' )

        # link the shader now
        self.shader.link()

        # bind our uniform indices
        self.shader.bind()
        self.shader.uniforms.in_diffuse = 0
        self.shader.unbind()

        self.md2 = pymesh.md2.MD2()
        if filename != None:
            self.md2.load( filename )
        else:
            self.md2.load_from_buffer( buffer )
        
        # load into OpenGL
        self._load()

    def __del__( self ):
        # free our vao
        vao = getattr( self, 'vao', None )
        if vao:
            glDeleteVertexArrays( 1, vao )

        # free our vbos
        # texture coords
        tcs = getattr( self, 'tc_vbo', None )
        if tcs:
            glDeleteBuffer( tcs )

        # indices
        indices = getattr( self, 'indice_vbo', None )
        if indices:
            glDeleteBuffer( indices )

        # frames
        frames = getattr( self, 'frames', None )
        if frames:
            for frame in frames:
                glDeleteBuffer( frame )

    def _load( self ):
        """
        Prepares the MD2 for rendering by OpenGL.
        """
        def process_vertices( md2 ):
            """Processes MD2 data to generate a single set
            of indices.
            MD2 is an older format that has 2 sets of indices.
            Vertex/Normal indices (md2.triangles.vertex_indices)
            and Texture Coordinate indices (md2.triangles.tc_indices).
            The problem is that modern 3D APIs don't like this.
            OpenGL only allows a single set of indices.
            We can either, extract the vertices, normals and
            texture coordinates using the indices.
            This will create a lot of data.
            This function provides an alternative.
            We iterate through the indices and determine if an index
            has a unique vertex/normal and texture coordinate value.
            If so, the index remains and the texture coordinate is moved
            into the vertex index location in the texture coordinate array.
            If not, a new vertex/normal/texture coordinate value is created
            and the index is updated.
            This function returns a tuple containing the following values.
            (
                [ new indices ],
                [ new texture coordinate array ],
                [ frame_layout( name, vertices, normals ) ]
                )
            """
            # convert our vertex / tc indices to a single indice
            # we iterate through our list and 
            indices = []
            frames = [
                (
                    frame.name,
                    list(frame.vertices),
                    list(frame.normals)
                    )
                for frame in md2.frames
                ]

            # set the size of our texture coordinate list to the
            # same size as one of our frame's vertex lists
            tcs = list( [[None, None]] * len(frames[ 0 ][ 1 ]) )

            for v_index, tc_index in zip(
                md2.triangles.vertex_indices,
                md2.triangles.tc_indices,
                ):

                indice = v_index

                if \
                    tcs[ v_index ][ 0 ] == None and \
                    tcs[ v_index ][ 1 ] == None:
                    # no tc set yet
                    # set ours
                    tcs[ v_index ][ 0 ] = md2.tcs[ tc_index ][ 0 ]
                    tcs[ v_index ][ 1 ] = md2.tcs[ tc_index ][ 1 ]

                elif \
                    tcs[ v_index ][ 0 ] != md2.tcs[ tc_index ][ 0 ] and \
                    tcs[ v_index ][ 1 ] != md2.tcs[ tc_index ][ 1 ]:

                    # a tc has been set and it's not ours
                    # create a new indice
                    indice = len( tcs )

                    # add a new unique vertice
                    for frame in frames:
                        # vertex data
                        frame[ 1 ].append( frame[ 1 ][ v_index ] )
                        # normal data
                        frame[ 2 ].append( frame[ 2 ][ v_index ] )
                    # texture coordinate
                    tcs.append(
                        [
                            md2.tcs[ tc_index ][ 0 ],
                            md2.tcs[ tc_index ][ 1 ]
                            ]
                        )

                # store the index
                indices.append( indice )

            # convert our frames to frame tuples
            frame_tuples = [
                pymesh.md2.MD2.frame_layout(
                    frame[ 0 ],
                    numpy.array( frame[ 1 ], dtype = numpy.float ),
                    numpy.array( frame[ 2 ], dtype = numpy.float )
                    )
                for frame in frames
                ]

            return (
                numpy.array( indices ),
                numpy.array( tcs ),
                frame_tuples
                )


        indices, tcs, frames = process_vertices( self.md2 )

        self.num_indices = len( indices )

        # create a vertex array object
        # and vertex buffer objects for our core data
        self.vao = (GLuint)()
        glGenVertexArrays( 1, self.vao )

        # load our buffers
        glBindVertexArray( self.vao )

        # create our vbo buffers
        # one for texture coordinates
        # one for indices
        vbos = (GLuint * 2)()
        glGenBuffers( len(vbos), vbos )
        self.tc_vbo = vbos[ 0 ]
        self.indice_vbo = vbos[ 1 ]

        # create our texture coordintes
        tcs = tcs.astype( 'float32' )
        glBindBuffer( GL_ARRAY_BUFFER, self.tc_vbo )
        glBufferData(
            GL_ARRAY_BUFFER,
            tcs.nbytes,
            (GLfloat * tcs.size)(*tcs.flat),
            GL_STATIC_DRAW
            )

        # create our index buffer
        indices = indices.astype( 'uint32' )
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.indice_vbo )
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            (GLuint * indices.size)(*indices.flat),
            GL_STATIC_DRAW
            )

        def create_frame_data( vertices, normals ):
            vbo = (GLuint)()
            glGenBuffers( 1, vbo )

            # interleave these arrays into a single array
            array = numpy.empty( (len(vertices) * 2, 3), dtype = 'float32' )
            array[::2] = vertices
            array[1::2] = normals

            glBindBuffer( GL_ARRAY_BUFFER, vbo )
            glBufferData(
                GL_ARRAY_BUFFER,
                array.nbytes,
                (GLfloat * array.size)(*array.flat),
                GL_STATIC_DRAW
                )

            return vbo

        # convert our frame data into VBOs
        self.frames = [
            create_frame_data( frame.vertices, frame.normals )
            for frame in frames
            ]

        # unbind our buffers
        glBindVertexArray( 0 )
        glBindBuffer( GL_ARRAY_BUFFER, 0 )
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 )

    @property
    def num_frames( self ):
        return len( self.md2.frames )

    def render( self, frame1, frame2, interpolation, projection, model_view ):
        # bind our shader and pass in our model view
        self.shader.bind()
        self.shader.uniforms.in_model_view = model_view
        self.shader.uniforms.in_projection = projection
        self.shader.uniforms.in_fraction = interpolation

        # we don't bind the diffuse texture
        # this is up to the caller to allow
        # multiple textures to be used per mesh instance
        frame1_data = self.frames[ frame1 ]
        frame2_data = self.frames[ frame2 ]

        # unbind the shader
        glBindVertexArray( self.vao )

        vertex_size = 6 * 4
        vertex_offset = 0 * 4
        normal_offset = 3 * 4

        # frame 1
        glBindBuffer( GL_ARRAY_BUFFER, frame1_data )
        glEnableVertexAttribArray( 0 )
        glEnableVertexAttribArray( 1 )
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, vertex_size, vertex_offset )
        glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, vertex_size, normal_offset )

        # frame 2
        glBindBuffer( GL_ARRAY_BUFFER, frame2_data )
        glEnableVertexAttribArray( 2 )
        glEnableVertexAttribArray( 3 )
        glVertexAttribPointer( 2, 3, GL_FLOAT, GL_FALSE, vertex_size, vertex_offset )
        glVertexAttribPointer( 3, 3, GL_FLOAT, GL_FALSE, vertex_size, normal_offset )

        # texture coords
        glBindBuffer( GL_ARRAY_BUFFER, self.tc_vbo )
        glEnableVertexAttribArray( 4 )
        glVertexAttribPointer( 4, 2, GL_FLOAT, GL_FALSE, 0, 0 )

        # indices
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, self.indice_vbo )

        glDrawElements(
            GL_TRIANGLES,
            self.num_indices,
            GL_UNSIGNED_INT,
            0
            )

        # reset our state
        glBindVertexArray( 0 )
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 )
        glBindBuffer( GL_ARRAY_BUFFER, 0 )
        self.shader.unbind()

class MD2_Mesh():
    """
    Provides the ability to load and render an MD2
    mesh.
    Uses MD2 to load MD2 mesh data.
    Loads mesh data onto the graphics card to speed
    up rendering. Allows for pre-baking the interpolation
    of frames.
    """
    
    def __init__( self, filename ):
        """
        Loads an MD2 from the specified file.
        """
        super( MD2_Mesh, self ).__init__()
        
        self.filename = filename
        self.data = None
        self.frame_1 = 0
        self.frame_2 = 0
        self.interpolation = 0.0

    @property
    def num_frames( self ):
        """Returns the number of keyframes.
        """
        return self.data.num_frames

    @property
    def animations( self ):
        """Returns the frame namesfor various animations.
        """
        return MD2.animations.keys()

    @property
    def animation( self ):
        """Returns the name of the current animation.
        This is determined by the current frame number.
        The animation name is taken from the standard MD2
        animation names and not from the MD2 file itself.
        """
        for name, value in MD2.animations.items():
            if \
                value[ 0 ] <= self.frame_1 and \
                value[ 1 ] >= self.frame_1:
                return name
        # unknown animation
        return None

    @property
    def frame_name( self ):
        return self.data.md2.frames[ self.frame_1 ].name

    @property
    def frame_rate( self ):
        """Returns the frames per second for the current animation.
        This uses the standard MD2 frame rate definition
        If the frame rate differs, over-ride this function.
        If the animation is outside the range of standard
        animations, a default value of 7.0 is returned.
        """
        anim = self.animation
        if anim:
            return self.animation_frame_rate( self.animation )
        else:
            return 7.0

    def animation_start_end_frame( self, animation ):
        return (
            MD2.animations[ animation ][ 0 ],
            MD2.animations[ animation ][ 1 ]
            )

    def animation_frame_rate( self, animation ):
        """Returns the frame rate for the specified animation
        """
        return MD2.animations[ animation ][ 2 ]

    def load( self ):
        """
        Reads the MD2 data from the existing
        specified filename.
        """
        if self.data == None:
            self.data = Data.load( self.filename )

    def unload( self ):
        if self.data != None:
            self.data = None
            Data.unload( self.filename )

    def render( self, projection, model_view ):
        # TODO: bind our diffuse texture to TEX0
        self.data.render(
            self.frame_1,
            self.frame_2,
            self.interpolation,
            projection,
            model_view
            )

def attribute_for_name( handle, name ):
    """Returns the attribute for the specified attribute index.
    This iterates over the attributes returned by 
    `py:func:pygly.shader.attribute_for_index`
    until it finds a matching name.
    If no name is found, None is returned.
    :rtype: tuple(name, size, type)
    :return: The attribute tuple or None.
    """
    # we can't get attributes directly
    # we have to iterate over the active attributes and find our
    # attribute match by the name given
    for attribute in attributes( handle ):
        name_, size_, type_ = attribute

        if name_ == name:
            return name_, size_, type_

    # no match found
    return None

def _generate_enum_map( enum_names ):
    """Convert dicts of format {'GL_ENUM_NAME': value, ...}
    to { GL_ENUM_NAME : value, ...}
    Used to ignore NameErrors that would otherwise result from incomplete
    OpenGL implementations.
    """
    map = {}
    for (key, value) in enum_names.items():
        try:
            map[ getattr(GL, key) ] = value
        except AttributeError:
            pass
    return map

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

    glClearColor(0.2, 0.2, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)

    lightPosLoc = glGetUniformLocation(shader, 'lightPos')
    glUniform3f(lightPosLoc, 100, -100, 0)

    colorLoc = glGetUniformLocation(shader, 'color')
    glUniform3f(colorLoc, 1.0, 1.0, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width/height, 0.1, 500.0)

    glTranslatef(0.0, 0.0, -100.0)

    # MAIN LOOP
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        model_.draw()
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == '__main__':
    main()