#ifndef __MC_H__
#define __MC_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <helper_cuda.h>  // includes cuda.h and cuda_runtime_api.h
#include <helper_functions.h>

#include <GL/freeglut.h>
#include "marchingCubes_kernel.cuh"

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY 10  // ms
#define EPSILON 5.0f
#define THRESHOLD 0.30f
#define DEBUG_BUFFERS 0

namespace MC
{

    const char *volumeFilename = "Bucky.raw";
    // constants
    const unsigned int window_width = 512;
    const unsigned int window_height = 512;

    // grid size parameters
    uint3 gridSizeLog2 = make_uint3(5, 5, 5);
    uint3 gridSizeShift;
    uint3 gridSize;
    uint3 gridSizeMask;

    float3 voxelSize;
    uint numVoxels = 0;
    uint maxVerts = 0;
    uint activeVoxels = 0;
    uint totalVerts = 0;

    float isoValue = 0.2f;
    float dIsoValue = 0.005f;

    StopWatchInterface *timer = 0;

    // Auto-Verification Code
    const int frameCheckNumber = 4;
    int fpsCount = 0;  // FPS count for averaging
    int fpsLimit = 1;  // FPS limit for sampling
    int g_Index = 0;
    unsigned int frameCount = 0;
    bool g_bValidate = false;

    int argc {NULL};
    char **argv {NULL};
    // device data
    GLuint posVbo, normalVbo;
    GLint gl_Shader;
    struct cudaGraphicsResource *cuda_posvbo_resource,
    *cuda_normalvbo_resource;  // handles OpenGL-CUDA exchange

    float4 *d_pos = 0, *d_normal = 0;

    uchar *d_volume = 0;
    uint *d_voxelVerts = 0;
    uint *d_voxelVertsScan = 0;
    uint *d_voxelOccupied = 0;
    uint *d_voxelOccupiedScan = 0;
    uint *d_compVoxelArray;

    // tables
    uint *d_numVertsTable = 0;
    uint *d_edgeTable = 0;
    uint *d_triTable = 0;

    // mouse controls
    int mouse_old_x, mouse_old_y;
    int mouse_buttons = 0;
    float3 rotate = make_float3(0.0, 0.0, 0.0);
    float3 translate = make_float3(0.0, 0.0, -3.0);

    // toggles
    bool wireframe = false;
    bool animate = true;
    bool lighting = true;
    bool render = true;
    bool compute = true;

    void runGraphicsTest(int argc, char **argv);
    void runAutoTest(int argc, char **argv);
    void initMC(int argc, char **argv);

    void computeIsosurface();
    void dumpFile(void *dData, int data_bytes, const char *file_name);
    template <class T>
    void dumpBuffer(T *d_buffer, int nelements, int size_element);
    void cleanup();
    bool initGL(int *argc, char **argv);
    void createVBO(GLuint *vbo, unsigned int size);
    void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_resource);
    void idle();

    void display();
    void keyboard(unsigned char key, int x, int y);
    void mouse(int button, int state, int x, int y);
    void motion(int x, int y);
    void reshape(int w, int h);

    void mainMenu(int i);
    void animation();
    void timerEvent(int value);
    void computeFPS();
    void initMenus();
    uchar *loadRawFile(char *filename, int size);
    GLuint compileASMShader(GLenum program_type, const char *code);
    void renderIsosurface();
}
#endif  // #ifndef _MARCHING_CUBES_KERNEL_CU_