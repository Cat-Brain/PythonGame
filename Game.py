from cmath import log
import contextlib, ctypes, logging, math, sys
from lib2to3.pygram import python_grammar_no_print_statement
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
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 9 * ctypes.sizeof(ctypes.c_float), None)

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 9 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(ctypes.sizeof(ctypes.c_float) * 3))

        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, False, 9 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(ctypes.sizeof(ctypes.c_float) * 6))
        
    def Draw(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.vertCount / 9))

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

    def FindDirections(self):
        self.right = glm.normalize(glm.cross(glm.vec3(0, 1, 0), self.forward))
        self.up = glm.cross(self.forward, self.right)

    def FindAllDirections(self):
        self.FindForward()
        self.FindDirections()

    def FindView(self):
        self.view = glm.mat4(self.right.x, self.up.x, -self.forward.x, 0, self.right.y, self.up.y, -self.forward.y, 0, self.right.z, self.up.z, -self.forward.z, 0, 0, 0, 0, 1) * glm.mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -self.pos.x, -self.pos.y, -self.pos.z, 1)

    def FindPerspective(self):
        self.perspective = glm.perspective(glm.radians(45.0), scr_width / scr_height, 0.1, 100.0)



class PlayerStats:
    speed : float
    chunkRenderDist : float

    def __init__(self):
        self.speed = 500
        self.chunkRenderDist = 2.0


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
        self.voxel_diffuse = Shader(voxelVertShader, fragShader)

    def Update(self, player : Player):
        self.default_diffuse.Use()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.default_diffuse.program, "view"), 1, False, glm.value_ptr(player.cam.view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.default_diffuse.program, "perspective"), 1, False, glm.value_ptr(player.cam.perspective))
        gl.glUniform3f(gl.glGetUniformLocation(self.default_diffuse.program, "lightPos"), player.cam.pos.x, player.cam.pos.y, player.cam.pos.z)
        gl.glUniform3f(gl.glGetUniformLocation(self.default_diffuse.program, "lightCol"), 1.0, 1.0, 1.0)
        
        self.voxel_diffuse.Use()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.voxel_diffuse.program, "view"), 1, False, glm.value_ptr(player.cam.view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.voxel_diffuse.program, "perspective"), 1, False, glm.value_ptr(player.cam.perspective))
        gl.glUniform3f(gl.glGetUniformLocation(self.voxel_diffuse.program, "lightPos"), player.cam.pos.x, player.cam.pos.y, player.cam.pos.z)
        gl.glUniform3f(gl.glGetUniformLocation(self.voxel_diffuse.program, "lightCol"), 1.0, 1.0, 1.0)
        


class Voxel:
    value : np.uint16

    def __init__(self, value : np.uint16):
        self.value = value


class Chunk:
    voxels : Voxel
    pos : glm.vec3
    nextQuad : float

    verts : float
    mesh : Mesh

    def __init__(self, pos : glm.vec3, noise):
        self.pos = pos
        self.voxels =  [[[0 for i in range(17)] for j in range(17)]for k in range(17)]
        self.verts = []
        self.nextQuad = [0.0 for i in range(54)]
        self.GenerateVoxels(noise)
        self.GenerateChunkMesh()


    def GenerateVoxels(self, noise):
        for x in range(17):
            for y in range(17):
                for z in range(17):
                    print("(" + str(x) + ", " + str(y) + ", " + str(z))
                    
                    self.voxels[x][y][z] = np.floor(((((x + self.pos.x) / 2 + (y + self.pos.y) / 2 + (z + self.pos.z) / 2) % 6) * 6) / 24 + 0.1)
                        
        print(self.voxels)
                    
                    
    def GenerateXQuad(self, x : int, y : int, z : int):
        dir : float
        dir = self.voxels[x][y][z] - self.voxels[x + 1][y][z]
        
        
        self.nextQuad[0] = x + 1
        self.nextQuad[1] = y
        self.nextQuad[2] = z
        self.nextQuad[3] = 1.0
        self.nextQuad[4] = 1.0
        self.nextQuad[5] = 1.0
        self.nextQuad[6] = dir
        self.nextQuad[7] = 0.0
        self.nextQuad[8] = 0.0
        
        self.nextQuad[9] = x + 1
        self.nextQuad[10] = y + 1
        self.nextQuad[11] = z
        self.nextQuad[12] = 1.0
        self.nextQuad[13] = 1.0
        self.nextQuad[14] = 1.0
        self.nextQuad[15] = dir
        self.nextQuad[16] = 0.0
        self.nextQuad[17] = 0.0
        
        self.nextQuad[18] = x + 1
        self.nextQuad[19] = y + 1
        self.nextQuad[20] = z + 1
        self.nextQuad[21] = 1.0
        self.nextQuad[22] = 1.0
        self.nextQuad[23] = 1.0
        self.nextQuad[24] = dir
        self.nextQuad[25] = 0.0
        self.nextQuad[26] = 0.0
        
        self.nextQuad[27] = x + 1
        self.nextQuad[28] = y
        self.nextQuad[29] = z
        self.nextQuad[30] = 1.0
        self.nextQuad[31] = 1.0
        self.nextQuad[32] = 1.0
        self.nextQuad[33] = dir
        self.nextQuad[34] = 0.0
        self.nextQuad[35] = 0.0
        
        self.nextQuad[36] = x + 1
        self.nextQuad[37] = y + 1
        self.nextQuad[38] = z + 1
        self.nextQuad[39] = 1.0
        self.nextQuad[40] = 1.0
        self.nextQuad[41] = 1.0
        self.nextQuad[42] = dir
        self.nextQuad[43] = 0.0
        self.nextQuad[44] = 0.0
        
        self.nextQuad[45] = x + 1
        self.nextQuad[46] = y
        self.nextQuad[47] = z + 1
        self.nextQuad[48] = 1.0
        self.nextQuad[49] = 1.0
        self.nextQuad[50] = 1.0
        self.nextQuad[51] = dir
        self.nextQuad[52] = 0.0
        self.nextQuad[53] = 0.0
        
        
        self.verts.extend(self.nextQuad)
                    
    def GenerateYQuad(self, x : int, y : int, z : int):
        dir : float
        dir = self.voxels[x][y][z] - self.voxels[x][y + 1][z]
        
        
        self.nextQuad[0] = x
        self.nextQuad[1] = y + 1
        self.nextQuad[2] = z
        self.nextQuad[3] = 1.0
        self.nextQuad[4] = 1.0
        self.nextQuad[5] = 1.0
        self.nextQuad[6] = 0.0
        self.nextQuad[7] = dir
        self.nextQuad[8] = 0.0
        
        self.nextQuad[9] = x
        self.nextQuad[10] = y + 1
        self.nextQuad[11] = z + 1
        self.nextQuad[12] = 1.0
        self.nextQuad[13] = 1.0
        self.nextQuad[14] = 1.0
        self.nextQuad[15] = 0.0
        self.nextQuad[16] = dir
        self.nextQuad[17] = 0.0
        
        self.nextQuad[18] = x + 1
        self.nextQuad[19] = y + 1
        self.nextQuad[20] = z + 1
        self.nextQuad[21] = 1.0
        self.nextQuad[22] = 1.0
        self.nextQuad[23] = 1.0
        self.nextQuad[24] = 0.0
        self.nextQuad[25] = dir
        self.nextQuad[26] = 0.0
        
        self.nextQuad[27] = x
        self.nextQuad[28] = y + 1
        self.nextQuad[29] = z
        self.nextQuad[30] = 1.0
        self.nextQuad[31] = 1.0
        self.nextQuad[32] = 1.0
        self.nextQuad[33] = 0.0
        self.nextQuad[34] = dir
        self.nextQuad[35] = 0.0
        
        self.nextQuad[36] = x + 1
        self.nextQuad[37] = y + 1
        self.nextQuad[38] = z + 1
        self.nextQuad[39] = 1.0
        self.nextQuad[40] = 1.0
        self.nextQuad[41] = 1.0
        self.nextQuad[42] = 0.0
        self.nextQuad[43] = dir
        self.nextQuad[44] = 0.0
        
        self.nextQuad[45] = x + 1
        self.nextQuad[46] = y + 1
        self.nextQuad[47] = z
        self.nextQuad[48] = 1.0
        self.nextQuad[49] = 1.0
        self.nextQuad[50] = 1.0
        self.nextQuad[51] = 0.0
        self.nextQuad[52] = dir
        self.nextQuad[53] = 0.0
        
        
        self.verts.extend(self.nextQuad)
                    
    def GenerateZQuad(self, x : int, y : int, z : int):
        dir : float
        dir = self.voxels[x][y][z] - self.voxels[x][y][z + 1]
        
        
        self.nextQuad[0] = x
        self.nextQuad[1] = y
        self.nextQuad[2] = z + 1
        self.nextQuad[3] = 1.0
        self.nextQuad[4] = 1.0
        self.nextQuad[5] = 1.0
        self.nextQuad[6] = 0.0
        self.nextQuad[7] = 0.0
        self.nextQuad[8] = dir
        
        self.nextQuad[9] = x
        self.nextQuad[10] = y + 1
        self.nextQuad[11] = z + 1
        self.nextQuad[12] = 1.0
        self.nextQuad[13] = 1.0
        self.nextQuad[14] = 1.0
        self.nextQuad[15] = 0.0
        self.nextQuad[16] = 0.0
        self.nextQuad[17] = dir
        
        self.nextQuad[18] = x + 1
        self.nextQuad[19] = y + 1
        self.nextQuad[20] = z + 1
        self.nextQuad[21] = 1.0
        self.nextQuad[22] = 1.0
        self.nextQuad[23] = 1.0
        self.nextQuad[24] = 0.0
        self.nextQuad[25] = 0.0
        self.nextQuad[26] = dir
        
        self.nextQuad[27] = x
        self.nextQuad[28] = y
        self.nextQuad[29] = z + 1
        self.nextQuad[30] = 1.0
        self.nextQuad[31] = 1.0
        self.nextQuad[32] = 1.0
        self.nextQuad[33] = 0.0
        self.nextQuad[34] = 0.0
        self.nextQuad[35] = dir
        
        self.nextQuad[36] = x + 1
        self.nextQuad[37] = y + 1
        self.nextQuad[38] = z + 1
        self.nextQuad[39] = 1.0
        self.nextQuad[40] = 1.0
        self.nextQuad[41] = 1.0
        self.nextQuad[42] = 0.0
        self.nextQuad[43] = 0.0
        self.nextQuad[44] = dir
        
        self.nextQuad[45] = x + 1
        self.nextQuad[46] = y
        self.nextQuad[47] = z + 1
        self.nextQuad[48] = 1.0
        self.nextQuad[49] = 1.0
        self.nextQuad[50] = 1.0
        self.nextQuad[51] = 0.0
        self.nextQuad[52] = 0.0
        self.nextQuad[53] = dir
        
        
        self.verts.extend(self.nextQuad)

    def GenerateChunkMesh(self):

        
        for x in range(16):
            for y in range(16):
                for z in range(16):
                    if self.voxels[x][y][z] != self.voxels[x + 1][y][z]:
                        #print("(" + str(self.voxels[x][y][z]) + ", " + str(self.voxels[x + 1][y][z]) + ")")
                        self.GenerateXQuad(x, y, z)
                    if self.voxels[x][y][z] != self.voxels[x][y + 1][z]:
                        #print("(" + str(self.voxels[x][y][z]) + ", " + str(self.voxels[x][y + 1][z]) + ")")
                        self.GenerateYQuad(x, y, z)
                    if self.voxels[x][y][z] != self.voxels[x][y][z + 1]:
                        #print("(" + str(self.voxels[x][y][z]) + ", " + str(self.voxels[x][y][z + 1]) + ")")
                        self.GenerateZQuad(x, y, z)

        self.mesh = Mesh(self.verts)


class VoxelWorld:
    chunks : Chunk
    noise : int

    def __init__(self, player : Player, noise : int):
        self.noise = noise
        self.Update(player)
        
    
    def CreateNearbyChunks(self, player : Player):
        minPos : glm.ivec3
        minPos = glm.ivec3(0, 0, 0)
        minPos.x = int(np.floor(player.cam.pos.x - player.stats.chunkRenderDist))
        minPos.y = int(np.floor(player.cam.pos.y - player.stats.chunkRenderDist))
        minPos.z = int(np.floor(player.cam.pos.z - player.stats.chunkRenderDist))
        
        maxPos : glm.ivec3
        maxPos = glm.ivec3(0, 0, 0)
        maxPos.x = int(np.ceil(player.cam.pos.x + player.stats.chunkRenderDist))
        maxPos.y = int(np.ceil(player.cam.pos.y + player.stats.chunkRenderDist))
        maxPos.z = int(np.ceil(player.cam.pos.z + player.stats.chunkRenderDist))
        
        for x in range(maxPos.x - minPos.x):
            for y in range(maxPos.y - minPos.y):
                for z in range(maxPos.z - minPos.z):
                    self.chunks.append(Chunk(glm.vec3(x + minPos.x, y + minPos.y, z + minPos.z), self.noise))
    
        
    def Update(self, player : Player):
        self.chunks = []
        self.CreateNearbyChunks(player)
        
        
    def Draw(self, shader : Shader):
        shader.Use()

        for chunk in self.chunks:
            chunk : Chunk
            chunk.mesh.Draw()




squareVertsNoInd = (
    -0.5,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0,
    -0.5, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0,
     0.5, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0,

    -0.5,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0,
     0.5, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0,
     0.5,-0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0
)


vertShader = '''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aCol;
layout (location = 2) in vec3 aNorm;

out vec3 vPos;
out vec3 vCol;
out vec3 vNorm;

uniform mat4 perspective;
uniform mat4 view;
uniform mat4 model;

void main()
{
    vPos = mat3(model) * aPos;
    gl_Position = perspective * view * model * vec4(aPos.x, aPos.y, aPos.z, 1.0);
    vCol = aCol;
    vNorm = aNorm;
}''' 

voxelVertShader = '''
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aCol;
layout (location = 2) in vec3 aNorm;

out vec3 vPos;
out vec3 vCol;
out vec3 vNorm;

uniform mat4 perspective;
uniform mat4 view;

void main()
{
    vec4 v4VPos = vec4(aPos.x, aPos.y, aPos.z, 1.0);
    vPos = aPos;
    gl_Position = perspective * view * v4VPos;
    vCol = aCol;
    vNorm = aNorm;
}''' 

 
 

fragShader = '''
#version 330 core
out vec4 FragColor;

in vec3 vPos;
in vec3 vCol;
in vec3 vNorm;

uniform vec3 lightPos;
uniform vec3 lightCol;

void main()
{
    vec3 lightDir = normalize(lightPos - vPos);
    vec3 diff = max(dot(vNorm, lightDir), 0.0) * lightCol;
    FragColor = vec4((diff) * vCol, 1.0f);
}'''



global player
player : Player
global shaderManager
shaderManager : ShaderManager
global testObj
testObj : Object
global world
world : VoxelWorld
global perlinNoise
perlinNoise : int



def FramebufferSizeCallback(window, width, height):
    gl.glViewport(0, 0, width, height)
    global scr_width, scr_height, player

    scr_width = width
    scr_height = height

    player.cam.FindPerspective()


lastMouseX = scr_width / 2
lastMouseY = scr_height / 2
firstMouse = True
def MouseCallback(window, xPos, yPos):
    global lastMouseX, lastMouseY, firstMouse, player

    if firstMouse: # initially set to true
        lastMouseX = xPos
        lastMouseY = yPos
        firstMouse = False


    xOffset = (lastMouseX - xPos) * player.cam.sensitivity
    yOffset = (lastMouseY - yPos) * player.cam.sensitivity

    player.cam.yaw += xOffset
    player.cam.pitch += yOffset
    
    player.cam.yaw += 180.0
    player.cam.yaw %= 360.0
    player.cam.yaw -= 180.0

    player.cam.pitch = np.minimum(89.9, np.maximum(-89.9, player.cam.pitch))

    lastMouseX = xPos
    lastMouseY = yPos




def Start():
    global window, scr_width, scr_height, perlinNoise, world

 
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
    
    gl.glEnable(gl.GL_DEPTH_TEST);

    glfw.set_window_size_callback(window, FramebufferSizeCallback)
    glfw.set_cursor_pos_callback(window, MouseCallback)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_input_mode(window, glfw.STICKY_KEYS, glfw.TRUE)

 
    print('set background to dark blue') 
    gl.glClearColor(0, 0, 0.4, 0)


    perlinNoise = 1
    

    world = VoxelWorld(player, perlinNoise)


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
    global world
    world.Draw(shaderManager.voxel_diffuse)

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

    print(player.cam.forward)
    player.cam.pos += player.cam.right * acceleration.x + player.cam.up * acceleration.y + player.cam.forward * acceleration.z


if (__name__ == '__main__'):
    player = Player(glm.vec3(0, 0, -5.0))

    Start() 

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
