import contextlib, ctypes, logging, math, sys
from OpenGL import GL as gl
import glfw
from ctypes import c_void_p
import numpy as np


scr_width = 500
scr_height = 400

global window



class Vec3:
    x : float
    y : float
    z : float

    def __init__(self, x : float, y : float, z : float):
        self.x = x
        self.y = y
        self.z = z

    def sqrMagnitude(self):
        return self.x ** 2 + self.y ** 2 + self.y ** 2

    def magnitude(self):
        return np.sqrt(self.magnitude)

    def normalized(self):
        return self / self.magnitude

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
    def __mul__(self, other : float):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    def __mod__(self, other):
        return Vec3(self.x % other.x, self.y % other.y, self.z % other.z)
    def __mod__(self, other : float):
        return Vec3(self.x % other, self.y % other, self.z % other)

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def Tup(self):
        return (self.x, self.y, self.z)


class Mesh:
    vao : np.uint8
    vbo : np.uint8
    ebo : np.uint8

    vertCount : np.uint16

 
    def __init__(self, verts, ind):
        self.vertCount = len(ind)

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        array_type = (gl.GLfloat * len(verts))
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(verts) * ctypes.sizeof(ctypes.c_float), array_type(*verts), gl.GL_STATIC_DRAW)

        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        array_type = (ctypes.c_uint16 * len(ind))
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(ind) * ctypes.sizeof(ctypes.c_uint16), array_type(*indices), gl.GL_STATIC_DRAW)
        
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), None)

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float),  ctypes.c_void_p(ctypes.sizeof(ctypes.c_float) * 3))
    

    def Draw(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, 0)


class Mesh2:
    vao : np.uint8
    vbo : np.uint8

    vertCount : np.uint16

 
    def __init__(self, verts):
        self.vertCount = len(verts)

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        array_type = (gl.GLfloat * len(verts))
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(verts) * ctypes.sizeof(ctypes.c_float), array_type(*verts), gl.GL_STATIC_DRAW)
        
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), None)

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float),  ctypes.c_void_p(ctypes.sizeof(ctypes.c_float) * 3))
 

class Shader:
    program : np.uint8
    vert : np.uint8
    frag : np.uint8

    def __init__(self, vert, frag):
        self.vert = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(self.vert, vert)
        gl.glCompileShader(self.vert)

        success = gl.glGetShaderiv(self.vert, gl.GL_COMPILE_STATUS)
        if not success:
            print('ERROR::SHADER::VERTEX::COMPILATION_FAILED' + str(gl.glGetShaderInfoLog(self.vert)))

 
        self.frag = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(self.frag, frag)
        gl.glCompileShader(self.frag)


        success = gl.glGetShaderiv(self.frag, gl.GL_COMPILE_STATUS)
        if not success:
            print('ERROR::SHADER::FRAGMENT::COMPILATION_FAILED' + str(gl.glGetShaderInfoLog(self.frag)))


        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, self.vert)
        gl.glAttachShader(self.program, self.frag)
        gl.glLinkProgram(self.program)

 
        success = gl.glGetProgramiv(self.program, gl.GL_VALIDATE_STATUS)

        if not success:
            print('ERROR::PROGRAM::COMPILATION_FAILED' + str(gl.glGetProgramInfoLog(self.program)))


        self.Use()

 
    def Use(self):
        gl.glUseProgram(self.program)

 
 
 

class Object: 

    mesh : Mesh
    shader : Shader

    def __init__(self, verts, ind, vertex, fragment):
        self.shader = Shader(vertex, fragment)
        self.mesh = Mesh(verts, ind)


    def Draw(self):
        self.shader.Use()
        self.mesh.Draw()


 
vertices = (
    -0.5,-0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, 0.5, 0.0, 0.0, 1.0, 0.5,
     0.5, 0.5, 0.0, 1.0, 1.0, 0.5,
     0.5,-0.5, 0.0, 1.0, 0.0, 0.5
)

indices : ctypes.c_uint16 = (
    0, 1, 2,
    0, 2, 3
)


vertShader = '''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aCol;

out vec3 vCol;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    vCol = aCol;
}''' 

 
 

fragShader = '''
#version 330 core
out vec4 FragColor;

in vec3 vCol;

void main()
{
    FragColor = vec4(vCol, 1.0f);
}'''




def perspective_fov(fov, aspect_ratio, near_plane, far_plane):
	num = 1.0 / np.tan(fov / 2.0)
	num9 = num / aspect_ratio
	return np.array([
		[num9, 0.0, 0.0, 0.0],
		[0.0, num, 0.0, 0.0],
		[0.0, 0.0, far_plane / (near_plane - far_plane), -1.0],
		[0.0, 0.0, (near_plane * far_plane) / (near_plane - far_plane), 0.0]
	])

 

def CreateMainWindow():
    global window
    global scr_width 
    global scr_height

 
    if not glfw.init():
        print('failed to initialize GLFW')
        sys.exit(1)

 
    print('requiring modern OpenGL without any legacy features')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)


    print('opening window') 
    title = 'Hi! =]' 
    window = glfw.create_window(scr_width, scr_height, title, None, None)

    if not window:
        print('failed to open GLFW window.')
        sys.exit(2) 
    glfw.make_context_current(window) 

 
    print('set background to dark blue') 
    gl.glClearColor(0, 0, 0.4, 0)


if (__name__ == '__main__'):
    CreateMainWindow() 
    testObj = Object(vertices, indices, vertShader, fragShader) 

    while(
        glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and 
        not glfw.window_should_close(window) 
    ): 
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) 
        # Draw the triangle 
        testObj.Draw()

        glfw.swap_buffers(window) 

        glfw.poll_events() 

 
 

    glfw.terminate() 

 