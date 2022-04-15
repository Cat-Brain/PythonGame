from cmath import log
import contextlib, ctypes, logging, math, sys
from turtle import pos
from OpenGL import GL as gl
import glfw
from ctypes import c_void_p
import numpy as np
import glm as glm

scr_width = 500
scr_height = 400

global window




# class Vec3:
#     x : float
#     y : float
#     z : float

#     def __init__(self, x : float, y : float, z : float):
#         self.x = x
#         self.y = y
#         self.z = z

#     def sqrMagnitude(self):
#         return self.x ** 2 + self.y ** 2 + self.y ** 2

#     def magnitude(self):
#         return np.sqrt(self.magnitude)

#     def normalized(self):
#         return self / self.magnitude

#     def __add__(self, other):
#         return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

#     def __sub__(self, other):
#         return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

#     def __mul__(self, other):
#         return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
#     def __mul__(self, other : float):
#         return Vec3(self.x * other, self.y * other, self.z * other)

#     def __truediv__(self, other):
#         return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)

#     def __mod__(self, other):
#         return Vec3(self.x % other.x, self.y % other.y, self.z % other.z)
#     def __mod__(self, other : float):
#         return Vec3(self.x % other, self.y % other, self.z % other)

#     def __str__(self):
#         return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

#     def Tup(self):
#         return (self.x, self.y, self.z)


class Mesh:
    vao : np.uint8
    vbo : np.uint8

    vertCount : np.uint16

 
    def __init__(self, verts):
        self.vertCount = len(verts)

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        array_type = (gl.GLfloat * self.vertCount)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertCount * ctypes.sizeof(ctypes.c_float), array_type(*verts), gl.GL_STATIC_DRAW)
        
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), None)

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float),  ctypes.c_void_p(ctypes.sizeof(ctypes.c_float) * 3))

    def Draw(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertCount)

    def Destroy(self):
        gl.glDeleteVertexArrays(1, [self.vao])
        gl.glDeleteBuffers(1, [self.vbo])

 

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
        log = str(gl.glGetProgramInfoLog(self.program))
        if not success and len(log) > 3:
            print('ERROR::PROGRAM::COMPILATION_FAILED' + log)


        self.Use()

 
    def Use(self):
        gl.glUseProgram(self.program)


    def Destroy(self):
        gl.glDeleteProgram(self.program)


class Object: 

    mesh : Mesh
    shader : Shader

    def __init__(self, verts, shader):
        self.shader = shader
        self.mesh = Mesh(verts)


    def Draw(self):
        self.shader.Use()
        self.mesh.Draw()

    
    def Destroy(self):
        self.mesh.Destroy()
        self.shader.Destroy()


class Camera:
    pos : glm.vec3

    yaw : float
    pitch : float

    forward : glm.vec3
    right : glm.vec3
    up : glm.vec3

    perspective : glm.mat4

    def __init__(self, pos : glm.vec3):
        self.pos = pos
        self.perspective = glm.perspective(glm.radians(45.0), scr_width / scr_height, 0.1, 100.0)


class PlayerStats:
    speed : float

    def __init__(self):
        self.speed = 100


class Player:
    cam : Camera
    stats : PlayerStats

    def __init__(self, pos : glm.vec3):
        self.cam = Camera(pos)
        self.stats = PlayerStats()


class ShaderManager:
    default_diffuse : Shader

    def __init__(self):
        self.default_diffuse = Shader(vertShader, fragShader)

    def Update(self, player : Player):
        self.default_diffuse.Use()
        gl.glUniform3f(gl.glGetUniformLocation(self.default_diffuse.program, "camPos"), player.cam.pos.x, player.cam.pos.y, player.cam.pos.z)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.default_diffuse.program, "perspective"), 1, False, glm.value_ptr(player.cam.perspective))



squareVertsNoInd = (
    -0.5,-0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, 0.5, 0.0, 0.0, 1.0, 0.5,
     0.5, 0.5, 0.0, 1.0, 1.0, 0.5,

    -0.5,-0.5, 0.0, 0.0, 0.0, 0.5,
     0.5, 0.5, 0.0, 1.0, 1.0, 0.5,
     0.5,-0.5, 0.0, 1.0, 0.0, 0.5
)


vertShader = '''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aCol;

out vec3 vCol;

uniform vec3 camPos;
uniform mat4 perspective;

void main()
{
    gl_Position = perspective * vec4(aPos - camPos, 1.0);
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

    glfw.set_input_mode(window, glfw.STICKY_KEYS, glfw.TRUE)

 
    print('set background to dark blue') 
    gl.glClearColor(0, 0, 0.4, 0)



global player
global shaderManager
global testObj

 
def ProcessInputs(deltaTime):
    global window
    global player

    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        player.cam.pos.x -= deltaTime * player.stats.speed

    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        player.cam.pos.x += deltaTime * player.stats.speed

    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        player.cam.pos.z += deltaTime * player.stats.speed

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        player.cam.pos.z -= deltaTime * player.stats.speed

    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        player.cam.pos.y -= deltaTime * player.stats.speed

    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        player.cam.pos.y += deltaTime * player.stats.speed


if (__name__ == '__main__'):
    CreateMainWindow() 

    player = Player(glm.vec3(0, 0, -0.5))

    shaderManager = ShaderManager()
    testObj = Object(squareVertsNoInd, shaderManager.default_diffuse)

    lastTime = glfw.get_time()

    while(
        glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and 
        not glfw.window_should_close(window) 
    ): 
        deltaTime = glfw.get_time() - lastTime

        ProcessInputs(deltaTime)
        
        print(str(player.cam.pos))
        shaderManager.Update(player)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Draw the triangle 
        testObj.Draw()

        glfw.swap_buffers(window) 

        glfw.poll_events()

        lastTime = glfw.get_time()

    
    testObj.Destroy()
    
    

    glfw.terminate()