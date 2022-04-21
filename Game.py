from cmath import log
import contextlib, ctypes, logging, math, sys
from tkinter import Scale
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


class Transform:
    position : glm.vec3
    rotation : glm.vec3
    scale : glm.vec3

    currentMatrix : glm.mat4

    def __init__(self):
        self.position = glm.vec3(0.0)
        self.rotation = self.position
        self.scale = glm.vec3(1.0)

        self.currentMatrix = glm.mat4(1.0)

    def __init__(self, position : glm.vec3, rotation : glm.vec3, scale : glm.vec3):
        self.position = position
        self.rotation = rotation
        self.scale = scale

        self.FindCurrentMatrix()

    def FindCurrentMatrix(self):
        self.currentMatrix = glm.mat4(1.0)

        self.currentMatrix = glm.translate(self.currentMatrix, self.position)

        self.currentMatrix = glm.rotate(self.currentMatrix, self.rotation.z, glm.vec3(0.0, 0.0, 1.0))
        self.currentMatrix = glm.rotate(self.currentMatrix, self.rotation.y, glm.vec3(0.0, 1.0, 0.0))
        self.currentMatrix = glm.rotate(self.currentMatrix, self.rotation.x, glm.vec3(1.0, 0.0, 0.0))

        self.currentMatrix = glm.scale(self.currentMatrix, self.scale)


class Object: 

    mesh : Mesh
    transform : Transform
    shader : Shader

    def __init__(self, verts, shader):
        self.transform = Transform(glm.vec3(0.0), glm.vec3(0.0), glm.vec3(1.0))
        self.shader = shader
        self.mesh = Mesh(verts)


    def Draw(self):
        self.shader.Use()
        self.transform.FindCurrentMatrix()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shader.program, "model"), 1, False, glm.value_ptr(self.transform.currentMatrix))
        self.mesh.Draw()

    
    def Destroy(self):
        self.mesh.Destroy()
        self.shader.Destroy()


class Camera:
    pos : glm.vec3

    yaw : float = 90.0
    pitch : float = 0.0
    sensitivity : float = 0.1

    forward : glm.vec3 = glm.vec3(0)
    right : glm.vec3
    up : glm.vec3

    view : glm.mat4
    perspective : glm.mat4

    def __init__(self, pos : glm.vec3):
        self.pos = pos
        self.FindPerspective()
        self.FindAllDirections()

    def FindForward(self):
        cosOfPitch = np.cos(glm.radians(self.pitch))
        self.forward.x = np.cos(glm.radians(self.yaw)) * cosOfPitch
        self.forward.y = np.sin(glm.radians(self.pitch))
        self.forward.z = np.sin(glm.radians(self.yaw)) * cosOfPitch

    def FindAllDirections(self):
        self.FindForward()
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.cross(self.forward, self.right)

    def FindDirections(self):
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.cross(self.forward, self.right)

    def FindView(self):
        #self.view = glm.lookAt(glm.vec3(self.pos.x, self.pos.y, -self.pos.z), glm.vec3(0), glm.vec3(0, 1, 0))
        self.view = glm.mat4(self.right.x, self.up.x, -self.forward.x, 0, self.right.y, self.up.y, -self.forward.y, 0, self.right.z, self.up.z, -self.forward.z, 0, 0, 0, 0, 1) * glm.mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -self.pos.x, -self.pos.y, -self.pos.z, 1)

    def FindPerspective(self):
        self.perspective = glm.perspective(glm.radians(45.0), scr_width / scr_height, 0.1, 100.0)



class PlayerStats:
    speed : float

    def __init__(self):
        self.speed = 500


class Player:
    cam : Camera
    stats : PlayerStats

    def __init__(self, pos : glm.vec3):
        self.cam = Camera(pos)
        self.stats = PlayerStats()


class ShaderManager:
    default_diffuse : Shader
    voxel_diffuse : Shader

    def __init__(self):
        self.default_diffuse = Shader(vertShader, fragShader)
        self.default_diffuse = Shader(voxelVertShader, fragShader)

    def Update(self, player : Player):
        self.default_diffuse.Use()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.default_diffuse.program, "view"), 1, False, glm.value_ptr(player.cam.view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.default_diffuse.program, "perspective"), 1, False, glm.value_ptr(player.cam.perspective))



class Voxel:
    value : np.uint16

    def __init__(self, value : np.uint16):
        self.value = value


class Chunk:
    voxels : Voxel
    pos : glm.vec3

    mesh : Mesh

    def __init__(self, pos : glm.vec3):
        self.voxels = [[[None] * 16], [[None] * 16], [[None] * 16]]
        self.pos = pos
        self.mesh.CreateChunkMesh()

    def CreateChunkMesh(self):
        self.mesh.Create()


class VoxelWorld:
    chunks : Chunk

    def __init__(self):
        self.chunks = []
        self.chunks.append(Chunk(glm.vec3(0, 0, 0)))
        
    def Draw(self, shader : Shader):
        shader.Use()

        for chunk in self.chunks:
            chunk.mesh.Draw()




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

uniform mat4 perspective;
uniform mat4 view;
uniform mat4 model;

void main()
{
    gl_Position = perspective * view * model * vec4(aPos, 1.0);
    vCol = aCol;
}''' 

voxelVertShader = '''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aCol;

out vec3 vCol;

uniform mat4 perspective;
uniform mat4 view;

void main()
{
    gl_Position = perspective * view * vec4(aPos, 1.0);
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



global player
player : Player
global shaderManager
shaderManager : ShaderManager
global testObj
testObj : Object
global world
world : VoxelWorld



def FramebufferSizeCallback(window, width, height):
    gl.glViewport(0, 0, width, height)
    global scr_width
    global scr_height

    scr_width = width
    scr_height = height

    global player
    player.cam.FindPerspective()


lastMouseX = scr_width / 2
lastMouseY = scr_height / 2
def MouseCallback(window, xPos, yPos):
    global lastMouseX
    global lastMouseY
    global player

    xOffset = (xPos - lastMouseX) * player.cam.sensitivity
    yOffset = (yPos - lastMouseY) * player.cam.sensitivity

    player.cam.yaw += xOffset
    player.cam.pitch += yOffset

    player.cam.pitch = np.minimum(89.9, np.maximum(-89.9, player.cam.pitch))

    lastMouseX = xPos
    lastMouseY = yPos




def Start():
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

    glfw.set_window_size_callback(window, FramebufferSizeCallback)
    glfw.set_cursor_pos_callback(window, MouseCallback)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_input_mode(window, glfw.STICKY_KEYS, glfw.TRUE)

 
    print('set background to dark blue') 
    gl.glClearColor(0, 0, 0.4, 0)


    global world
    world = VoxelWorld()


global lastTime
def Update():
    global lastTime
    global player
    global testObj
    global shaderManager

    time = glfw.get_time()
    deltaTime = time - lastTime

    player.cam.FindAllDirections()

    ProcessInputs(deltaTime)

    player.cam.FindView()


    testObj.transform.rotation = glm.vec3(0.1 * time, 0.2 * time, 0.3 * time)
    #testObj.transform.scale = glm.vec3(1.0 + 0.1 * time, 1.0 + 0.2 * time, 1.0 + 0.3 * time)
        
    print(str(player.cam.pos))
    shaderManager.Update(player)

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    # Draw the triangle 
    testObj.Draw()

    glfw.swap_buffers(window) 

    glfw.poll_events()

    lastTime = glfw.get_time()




 
def ProcessInputs(deltaTime):
    global window
    global player

    acceleration = glm.vec3(0)

    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        acceleration.x -= 1

    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        acceleration.x += 1

    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        acceleration.z -= 1

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        acceleration.z += 1

    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        acceleration.y -= 1

    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        acceleration.y += 1

    acceleration *= deltaTime * player.stats.speed

    player.cam.pos += player.cam.right * acceleration.x + player.cam.up * acceleration.y + player.cam.forward * acceleration.z


if (__name__ == '__main__'):
    Start() 

    player = Player(glm.vec3(0, 0, -5.0))

    shaderManager = ShaderManager()
    testObj = Object(squareVertsNoInd, shaderManager.default_diffuse)

    lastTime = glfw.get_time()

    while(
        glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and 
        not glfw.window_should_close(window) 
    ):
        Update()

    
    testObj.Destroy()
    
    

    glfw.terminate()