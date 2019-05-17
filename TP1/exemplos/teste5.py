import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders
import numpy
#import pyrr
import glm


def main():

    # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
  

    vertex_shader = """
    #version 330
    in vec3 position;    
    uniform mat4 transform;
    out vec3 newColor;
    void main()
    {
        gl_Position = transform * vec4(position, 1.0f);
        //newColor = vec3(1.0,0.0,0.0);
        newColor = vec3(position.x,0.0,0.0);
    }
    """

    fragment_shader = """
    #version 330
    in vec3 newColor;

    out vec4 outColor;
    void main()
    {
        outColor = vec4(newColor, 1.0f);
    }
    """
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    glUseProgram(shader)

    glClearColor(0.2, 0.3, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    pers = glm.perspective(0.5,1.0,0.1,10.0)

    #print pers
    trans = glm.translate(glm.mat4(1.0), glm.vec3(0.0,0.0,-8.5))

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #rot_x = glm.rotate(glm.mat4(1.0),0.5 * glfw.get_time(),glm.vec3(1.0,0.0,0.0))
        #rot_y = glm.rotate(glm.mat4(1.0),0.5 * glfw.get_time(),glm.vec3(0.0,1.0,0.0))

        transformLoc = glGetUniformLocation(shader, "transform")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm.value_ptr(pers*trans))

        qobj = gluNewQuadric()
        gluSphere(qobj, 1, 50, 50)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
