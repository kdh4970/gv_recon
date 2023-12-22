#include "marchingCubes.h"
#include <helper_gl.h>
////////////////////////////////////
// Marching Cubes Variables Setup //
////////////////////////////////////

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY 10  // ms
#define EPSILON 5.0f
#define THRESHOLD 0.30f

// Shared Variable for DeviceKernel and marchingcubes
std::mutex mtx;

uchar* d_voxelRaw;
uchar* d_mc_input;


std::map<std::string, unsigned char> mask_decode_map;
std::string target_segment_classes[5] = {"person", "chair", "table", "wall", "tv"};
std::vector<float3> Segmented_Vertices[5];
std::vector<int3> Segmented_Triangles[5];
unsigned char offset = 6;

bool isMultiThread = true;
bool isMask;

unsigned int window_width = 1280;
unsigned int window_height = 720;

// grid size parameters
uint3 gridSizeLog2 = make_uint3(9, 9, 9);
uint3 gridSizeShift = make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);
uint3 gridSize = make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
uint3 gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);

float3 voxelSize = make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
uint numVoxels = gridSize.x * gridSize.y * gridSize.z;
uint maxVerts = gridSize.x * gridSize.y * 100;
uint activeVoxels = 0;
uint totalVerts = 0;

float isoValue = 0.1f;
float dIsoValue = 0.005f;

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 1;  // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
bool g_bValidate = false;

// device data
GLuint posVbo, normalVbo;
GLint gl_Shader;
struct cudaGraphicsResource *cuda_posvbo_resource, *cuda_normalvbo_resource, *cuda_facevbo_resource;  // handles OpenGL-CUDA exchange

float4 *d_pos = 0, *d_normal = 0;

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
float _fovy = 45.0;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float3 glrotate = make_float3(0.0, 0.0, 0.0);
float3 gltranslate = make_float3(0.0, 0.0, -3.0);

// toggles
bool wireframe = false;
bool animate = false;
bool lighting = true;
bool render = true;
bool compute = true;

SharedMemoryManager shm_manager[5];

namespace MarchingCubes 
{

void getVoxelData(boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap, const int numVoxels)
{
  mtx.lock();
  
  cudaMemset(d_voxelRaw, 0, sizeof(uchar) * numVoxels);
  // launch kernel to get voxel data
  ptrbitVoxmap->getVoxelRaw(d_voxelRaw,isMask);
	cudaDeviceSynchronize();
  mtx.unlock();
}

void run_glut_with_marchingCubes(int argc, char** argv)
{
  mtx.lock();
  printf("=============================== MC INFO ================================\n");
  printf("grid: %d x %d x %d = %d voxels\n", gridSize.x, gridSize.y, gridSize.z, numVoxels);
  printf("max verts = %d\n", maxVerts);
  cudaMalloc(&d_mc_input, sizeof(uchar) * numVoxels);
  cudaMemset(d_mc_input, 0, sizeof(uchar) * numVoxels);
  createVolumeTexture(d_mc_input, gridSize.x * gridSize.y * gridSize.z * sizeof(uchar));
  mtx.unlock();

  mask_decode_map.insert(std::pair<std::string,unsigned char>(target_segment_classes[0],offset + 0));
  mask_decode_map.insert(std::pair<std::string,unsigned char>(target_segment_classes[1],offset + 14));
  mask_decode_map.insert(std::pair<std::string,unsigned char>(target_segment_classes[2],offset + 45));
  mask_decode_map.insert(std::pair<std::string,unsigned char>(target_segment_classes[3],offset + 48));
  mask_decode_map.insert(std::pair<std::string,unsigned char>(target_segment_classes[4],offset + 19));

  // set shared memory manager
  for(int i{0};i<5;i++)
  {
    shm_manager[i].init(1000+i,target_segment_classes[i]);
  }
  

////////////////
// GLUT Setup //
////////////////

  if (false == initGL(&argc, argv)) {
    return;
  }
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutReshapeFunc(reshape);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  initMenus();
  sdkCreateTimer(&timer);

//////////////////////////
// Marching Cubes Setup //
//////////////////////////
  
  if (g_bValidate) {
    cudaMalloc((void **)&(d_pos), maxVerts * sizeof(float) * 4);
    cudaMalloc((void **)&(d_normal), maxVerts * sizeof(float) * 4);
  } else {
    // create VBOs
    createVBO(&posVbo, maxVerts * sizeof(float) * 4);
    // DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(posVbo) );
    cudaGraphicsGLRegisterBuffer(
        &cuda_posvbo_resource, posVbo, cudaGraphicsMapFlagsWriteDiscard);

    createVBO(&normalVbo, maxVerts * sizeof(float) * 4);
    // DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(normalVbo));
    cudaGraphicsGLRegisterBuffer(
        &cuda_normalvbo_resource, normalVbo, cudaGraphicsMapFlagsWriteDiscard);
    
  }
  // allocate textures
  allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

  // allocate device memory
  unsigned int memSize = sizeof(uint) * numVoxels;
  cudaMalloc((void **)&d_voxelVerts, memSize);
  cudaMalloc((void **)&d_voxelVertsScan, memSize);
  cudaMalloc((void **)&d_voxelOccupied, memSize);
  cudaMalloc((void **)&d_voxelOccupiedScan, memSize);
  cudaMalloc((void **)&d_compVoxelArray, memSize);

  // start rendering mainloop
  glutMainLoop();
}




void cleanup() {
  if (g_bValidate) {
    cudaFree(d_pos);
    cudaFree(d_normal);
  } else {
    sdkDeleteTimer(&timer);

    deleteVBO(&posVbo, &cuda_posvbo_resource);
    deleteVBO(&normalVbo, &cuda_normalvbo_resource);
  }
  destroyAllTextureObjects();
  cudaFree(d_edgeTable);
  cudaFree(d_triTable);
  cudaFree(d_numVertsTable);
  cudaFree(d_voxelVerts);
  cudaFree(d_voxelVertsScan);
  cudaFree(d_voxelOccupied);
  cudaFree(d_voxelOccupiedScan);
  cudaFree(d_compVoxelArray);

  if (d_mc_input) {
    cudaFree(d_mc_input);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void computeIsosurface() {
  int threads = 128;
  dim3 grid(numVoxels / threads, 1, 1);

  // get around maximum grid size of 65535 in each dimension
  if (grid.x > 65535) {
    grid.y = grid.x / 32768;
    grid.x = 32768;
  }

  // calculate number of vertices need per voxel
  launch_classifyVoxel(grid, threads, d_voxelVerts, d_voxelOccupied, d_mc_input,
                      gridSize, gridSizeShift, gridSizeMask, numVoxels,
                      voxelSize, isoValue);

#if SKIP_EMPTY_VOXELS
  // scan voxel occupied array
  ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

  // read back values to calculate total number of non-empty voxels
  // since we are using an exclusive scan, the total is the last value of
  // the scan result plus the last value in the input array
  {
    uint lastElement, lastScanElement;
    cudaMemcpy((void *)&lastElement,
                              (void *)(d_voxelOccupied + numVoxels - 1),
                              sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&lastScanElement,
                              (void *)(d_voxelOccupiedScan + numVoxels - 1),
                              sizeof(uint), cudaMemcpyDeviceToHost);
    activeVoxels = lastElement + lastScanElement;
  }

  if (activeVoxels == 0) {
    // return if there are no full voxels
    totalVerts = 0;
    return;
  }

  // compact voxel index array
  launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied,
                      d_voxelOccupiedScan, numVoxels);
  // getLastCudaError("compactVoxels failed");

#endif  // SKIP_EMPTY_VOXELS

  // scan voxel vertex count array
  ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);

  // readback total number of vertices
  {
    uint lastElement, lastScanElement;
    cudaMemcpy((void *)&lastElement,
                              (void *)(d_voxelVerts + numVoxels - 1),
                              sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&lastScanElement,
                              (void *)(d_voxelVertsScan + numVoxels - 1),
                              sizeof(uint), cudaMemcpyDeviceToHost);
    totalVerts = lastElement + lastScanElement;
  }

  // generate triangles, writing to vertex buffers
  if (!g_bValidate) {
    size_t num_bytes;
    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_pos,
    // posVbo));
    cudaGraphicsMapResources(1, &cuda_posvbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer(
        (void **)&d_pos, &num_bytes, cuda_posvbo_resource);

    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_normal,
    // normalVbo));
    cudaGraphicsMapResources(1, &cuda_normalvbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer(
        (void **)&d_normal, &num_bytes, cuda_normalvbo_resource);

  }

#if SKIP_EMPTY_VOXELS
  dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);
#else
  dim3 grid2((int)ceil(numVoxels / (float)NTHREADS), 1, 1);
#endif

  while (grid2.x > 65535) {
    grid2.x /= 2;
    grid2.y *= 2;
  }

#if SAMPLE_VOLUME
  launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
                            d_voxelVertsScan, d_mc_input, gridSize, gridSizeShift,
                            gridSizeMask, voxelSize, isoValue, activeVoxels,
                            maxVerts);
#else
  launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
                          d_voxelVertsScan, gridSize, gridSizeShift,
                          gridSizeMask, voxelSize, isoValue, activeVoxels,
                          maxVerts);
#endif
  if (!g_bValidate) {
    // DEPRECATED:      checkCudaErrors(cudaGLUnmapBufferObject(normalVbo));
    cudaGraphicsUnmapResources(1, &cuda_normalvbo_resource, 0);
    // DEPRECATED:      checkCudaErrors(cudaGLUnmapBufferObject(posVbo));
    cudaGraphicsUnmapResources(1, &cuda_posvbo_resource, 0);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, unsigned int size) {
  // create buffer object
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);

  // initialize buffer object
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glutReportErrors();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_resource) {
  glBindBuffer(1, *vbo);
  glDeleteBuffers(1, vbo);
  // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(*vbo));
  cudaGraphicsUnregisterResource(*cuda_resource);

  *vbo = 0;
}




void SegmentMCinput(std::string target_class, int target_index){
  Segmented_Vertices[target_index].clear();
  Segmented_Triangles[target_index].clear();
  auto start = std::chrono::system_clock::now();
  
  // first, choose the class. and call the kernel to convert voxel raw data to marching cubes input data.
  launch_generateMCInput(gridSize, d_mc_input, mask_decode_map[target_class]);
  cudaDeviceSynchronize();

  // second, compute isosurface. get d_pos
  computeIsosurface(); cudaDeviceSynchronize();

  // third, copy d_pos to host.
  float4* h_pos = new float4[totalVerts];
  cudaMemcpy(h_pos, d_pos, sizeof(float) * totalVerts * 4, cudaMemcpyDeviceToHost);

  for(int i{0};i<totalVerts;i++){
    float3 temp {h_pos[i].x, h_pos[i].y, h_pos[i].z};
    Segmented_Vertices[target_index].push_back(temp);

    if(i%3==2){
      int3 temp2 {i-2,i-1,i};
      Segmented_Triangles[target_index].push_back(temp2);
    }
  }
  auto end = std::chrono::system_clock::now();
  printf("| [%s] Data Copy Time (Device to Host)              : %f sec\n", target_class.c_str(), std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);

  // fourth, free the memory.
  delete[] h_pos;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
  bool zoom = false;
  switch (key) {
    case (27):
      cleanup();
      exit(EXIT_SUCCESS);

    case '=':
      isoValue += 0.01f;
      break;

    case '-':
      isoValue -= 0.01f;
      break;

    case '+':
      isoValue += 0.1f;
      break;

    case '_':
      isoValue -= 0.1f;
      break;

    case 'w':
      wireframe = !wireframe;
      break;

    case ' ':
      animate = !animate;
      break;

    case 'l':
      lighting = !lighting;
      break;

    case 'r':
      render = !render;
      break;

    case 'c':
      compute = !compute;
      break;

    case 'a':
      _fovy -= 2.0;
      zoom=true;
      if (_fovy < 15.0) {_fovy = 15.0; zoom = false;}
      if (_fovy > 90.0) {_fovy = 90.0; zoom = false;}
      if(zoom){
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(_fovy, (float)window_width / (float)window_height, 0.1, 10.0);
        glMatrixMode(GL_MODELVIEW);
        glViewport(0, 0, window_width, window_height);
      }
      break;

    case 'd':
      _fovy += 2.0;
      zoom=true;
      if (_fovy < 15.0) {_fovy = 15.0; zoom = false;}
      if (_fovy > 90.0) {_fovy = 90.0; zoom = false;}
      if(zoom){
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(_fovy, (float)window_width / (float)window_height, 0.1, 10.0);
        glMatrixMode(GL_MODELVIEW);
        glViewport(0, 0, window_width, window_height);
      }
      break;
    
    case 's':
      mtx.lock();
      testSave();

      mtx.unlock();
  }

  printf("isoValue = %f\n", isoValue);
  printf("voxels = %d\n", activeVoxels);
  printf("verts = %d\n", totalVerts);
  printf("occupancy: %d / %d = %.2f%%\n", activeVoxels, numVoxels,
        activeVoxels * 100.0f / (float)numVoxels);

  if (compute) {
    mtx.lock();
    computeIsosurface();
    mtx.unlock();
  }
}


////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y) {
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1 << button;
  } else if (state == GLUT_UP) {
    mouse_buttons = 0;
  }
  
  mouse_old_x = x;
  mouse_old_y = y;
}


void motion(int x, int y) {
  float dx = (float)(x - mouse_old_x);
  float dy = (float)(y - mouse_old_y);

  if (mouse_buttons == 1) {
    glrotate.x += dy * 0.2f;
    glrotate.y += dx * 0.2f;
  } else if (mouse_buttons == 2) {
    gltranslate.x += dx * 0.01f;
    gltranslate.y -= dy * 0.01f;
  } else if (mouse_buttons == 3) {
    gltranslate.z += dy * 0.01f;
  }

  mouse_old_x = x;
  mouse_old_y = y;
  glutPostRedisplay();
}

void reshape(int w, int h) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(_fovy, (float)w / (float)h, 0.1, 10.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);
  window_height=h;
  window_width=w;
}


void mainMenu(int i) { keyboard((unsigned char)i, 0, 0); }

void animation() {
  if (animate) {
    isoValue += dIsoValue;

    if (isoValue < 0.1f) {
      isoValue = 0.1f;
      dIsoValue *= -1.0f;
    } else if (isoValue > 0.9f) {
      isoValue = 0.9f;
      dIsoValue *= -1.0f;
    }
  }
}

void timerEvent(int value) {
  animation();
  glutPostRedisplay();
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display() {
  sdkStartTimer(&timer);
  
  // run CUDA kernel to generate geometry
  if (compute) {
    mtx.lock();
    if(isMask)
    {
      cudaMemcpy(d_mc_input, d_voxelRaw, sizeof(uchar) * numVoxels, cudaMemcpyDeviceToDevice);
      computeIsosurface();
      testSave();
      // do nothing now...
    }
    else{
      cudaMemcpy(d_mc_input, d_voxelRaw, sizeof(uchar) * numVoxels, cudaMemcpyDeviceToDevice);
      computeIsosurface();
    }
    mtx.unlock();
  }

  // Common display code path
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(gltranslate.x, gltranslate.y, gltranslate.z);
    glRotatef(glrotate.x, 1.0, 0.0, 0.0);
    glRotatef(glrotate.y, 0.0, 1.0, 0.0);

    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    if (lighting) {
      glEnable(GL_LIGHTING);
    }

    // render
    if (render) {
      glPushMatrix();
      glRotatef(180.0, 0.0, 0.0, 1.0);
      glRotatef(180.0, 0.0, 1.0, 0.0);
      glRotatef(90.0, 1.0, 0.0, 0.0);
      renderIsosurface();
      glPopMatrix();
    }

    glDisable(GL_LIGHTING);
  }
  glutSwapBuffers();
  glutReportErrors();

  sdkStopTimer(&timer);
  destroyAllTextureObjects();
  computeFPS();
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize OpenGL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv) {
  // Create GL context
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(window_width, window_height);
  glutCreateWindow("CUDA Marching Cubes");

  if (!isGLVersionSupported(2, 0)) {
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    return false;
  }

  // default initialization
  glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
  glEnable(GL_DEPTH_TEST);

  // good old-fashioned fixed function lighting
  float black[] = {0.0f, 0.0f, 0.0f, 1.0f};
  float white[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float ambient[] = {0.1f, 0.1f, 0.1f, 1.0f};
  float diffuse[] = {0.9f, 0.9f, 0.9f, 1.0f};
  float lightPos[] = {0.0f, 0.0f, 1.0f, 0.0f};

  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

  glLightfv(GL_LIGHT0, GL_AMBIENT, white);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
  glLightfv(GL_LIGHT0, GL_SPECULAR, white);
  glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

  glEnable(GL_LIGHT0);
  glEnable(GL_NORMALIZE);

  // load shader program
  // shader for displaying floating-point texture
  static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";
  gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

  glutReportErrors();
  return true;
}

void initMenus() {
  glutCreateMenu(mainMenu);
  glutAddMenuEntry("Toggle animation [ ]", ' ');
  glutAddMenuEntry("Increment isovalue [+]", '+');
  glutAddMenuEntry("Decrement isovalue [-]", '-');
  glutAddMenuEntry("Toggle computation [c]", 'c');
  glutAddMenuEntry("Toggle rendering [r]", 'r');
  glutAddMenuEntry("Toggle lighting [l]", 'l');
  glutAddMenuEntry("Toggle wireframe [w]", 'w');
  glutAddMenuEntry("Quit (esc)", '\033');
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void computeFPS() {
  frameCount++;
  fpsCount++;

  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    sprintf(fps, "Vertices : %d, Active Voxels : %d", totalVerts, activeVoxels);

    glutSetWindowTitle(fps);
    fpsCount = 0;

    fpsLimit = ftoi(MAX(1.f, ifps));
    sdkResetTimer(&timer);
  }
}

GLuint compileASMShader(GLenum program_type, const char *code) {
  GLuint program_id;
  glGenProgramsARB(1, &program_id);
  glBindProgramARB(program_type, program_id);
  glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
                    (GLsizei)strlen(code), (GLubyte *)code);

  GLint error_pos;
  glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

  if (error_pos != -1) {
    const GLubyte *error_string;
    error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
    fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos,
            error_string);
    return 0;
  }

  return program_id;
}


////////////////////////////////////////////////////////////////////////////////
// Render isosurface geometry from the vertex buffers
////////////////////////////////////////////////////////////////////////////////
void renderIsosurface() {
  glBindBuffer(GL_ARRAY_BUFFER, posVbo);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, normalVbo);
  glNormalPointer(GL_FLOAT, sizeof(float) * 4, 0);
  glEnableClientState(GL_NORMAL_ARRAY);

  glColor3f(1.0, 0.0, 0.0);
  glDrawArrays(GL_TRIANGLES, 0, totalVerts);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void RemoveDuplicatedVertices(std::vector<float3> &vertices, std::vector<int3> &triangles)
{
  typedef std::tuple<double, double, double> Coordinate3;
  std::unordered_map<Coordinate3, size_t, hash_tuple<Coordinate3>>point_to_old_index;
  std::vector<int> index_old_to_new(vertices.size());
  size_t old_vertex_num = vertices.size();
  size_t k = 0;                                  // new index
  for (size_t i = 0; i < old_vertex_num; i++) {  // old index
      Coordinate3 coord = std::make_tuple(vertices[i].x, vertices[i].y,
                                          vertices[i].z);
      if (point_to_old_index.find(coord) == point_to_old_index.end()) {
        point_to_old_index[coord] = i;
        vertices[k] = vertices[i];
        index_old_to_new[i] = (int)k;
        k++;
      } else {
        index_old_to_new[i] = index_old_to_new[point_to_old_index[coord]];
      }
  }
  vertices.resize(k);
  if (k < old_vertex_num) {
      for (auto &triangle : triangles) {
        triangle.x = index_old_to_new[triangle.x];
        triangle.y = index_old_to_new[triangle.y];
        triangle.z = index_old_to_new[triangle.z];
      }
  }
}

void RemoveUnreferencedVertices(std::vector<float3> &vertices, std::vector<int3> &triangles)
{
  std::vector<bool> vertex_has_reference(vertices.size(), false);
  for (const auto &triangle : triangles) {
      vertex_has_reference[triangle.x] = true;
      vertex_has_reference[triangle.y] = true;
      vertex_has_reference[triangle.z] = true;
  }
  std::vector<int> index_old_to_new(vertices.size());
  size_t old_vertex_num = vertices.size();
  size_t k = 0;                                  // new index
  for (size_t i = 0; i < old_vertex_num; i++) {  // old index
      if (vertex_has_reference[i]) {
          vertices[k] = vertices[i];
          index_old_to_new[i] = (int)k;
          k++;
      } else {
          index_old_to_new[i] = -1;
      }
  }
  vertices.resize(k);
  if (k < old_vertex_num) {
      for (auto &triangle : triangles) {
          triangle.x = index_old_to_new[triangle.x];
          triangle.y = index_old_to_new[triangle.y];
          triangle.z = index_old_to_new[triangle.z];
      }
  }
}


/**
 * @brief Function for simplifying mesh by removing duplicated vertices and removing unreferenced vertices
 * 
 * @param vertices 
 * @param triangles 
 */
void SimplifyMesh(std::vector<float3> &vertices, std::vector<int3> &triangles)
{
  size_t original_vertex_size = vertices.size();
  size_t original_triangle_size = triangles.size();

  auto t_start = std::chrono::steady_clock::now();
  RemoveDuplicatedVertices(vertices, triangles);

  auto t_chk1 = std::chrono::steady_clock::now();
  RemoveUnreferencedVertices(vertices, triangles);

  auto t_end = std::chrono::steady_clock::now();

  std::cout << "| >>> Removing Duplicated Vertices...\n";
  std::cout << "| Number of Vertex changed from " << original_vertex_size <<" to " << vertices.size() << "\n";
  std::cout << "| Number of Triangle changed from " << original_triangle_size <<" to "<< triangles.size() << "\n";
  std::cout << "| Time for removing duplicated vertices        : " << std::chrono::duration_cast<std::chrono::milliseconds>(t_chk1 - t_start).count() <<" ms\n";
  std::cout << "| Time for removing unreferenced vertices      : " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_chk1).count() <<" ms\n";
}

void SimplifyMeshSeg(std::string target_class, std::vector<float3> &vertices, std::vector<int3> &triangles)
{
  size_t original_vertex_size = vertices.size();
  size_t original_triangle_size = triangles.size();

  auto t_start = std::chrono::steady_clock::now();
  RemoveDuplicatedVertices(vertices, triangles);

  auto t_chk1 = std::chrono::steady_clock::now();
  RemoveUnreferencedVertices(vertices, triangles);

  auto t_end = std::chrono::steady_clock::now();

  printf("| [%s] Input Vertices : %d, Input Triangles : %d\n",target_class.c_str(), original_vertex_size, original_triangle_size);
  printf("| [%s] Deduplicated Vertices : %d, Deduplicated Triangles : %d\n", target_class.c_str(), vertices.size(), triangles.size());
  printf("| [%s] Time for removing duplicated vertices      : %lld ms\n", target_class.c_str(), std::chrono::duration_cast<std::chrono::milliseconds>(t_chk1 - t_start).count());
}

/**
 * @brief Function for writing mesh data to txt file for all classes data
 * 
 * @param target_class 
 * @param decimation_ratio 
 */
void WriteTxtFile(std::string target_class, double decimation_ratio)
{
  std::cout << "+------------------- Saving Mesh Data of [" << target_class << "] -------------------+\n";
  std::string filename = std::string("/home/do/ros2_ws/src/gv_recon/mesh_data_") + target_class + ".txt";

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  // Copy device data to host data
  std::cout<<"| >>> Getting Mesh Data of [" << target_class << "] \n";
  float4* h_pos = new float4[totalVerts];
  cudaMemcpy(h_pos, d_pos, sizeof(float) * totalVerts * 4, cudaMemcpyDeviceToHost);

  // For Mesh simplification by using Open3D
  std::vector<float3> vertices;
  std::vector<int3> triangles;
  for (auto i{0}; i<totalVerts; ++i)
  {
    auto const& v = h_pos[i];
    float3 temp {v.x, v.y, v.z};
    vertices.push_back(temp);
    if(i%3==2) 
    {
      int3 temp2 {i-2,i-1,i};
      triangles.push_back(temp2);
    }
  }
  end = std::chrono::system_clock::now();
  std::cout << "| Data Copy and Convert Time                   : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
  // First, Simplify mesh by removing duplicated vertices and removing unreferenced vertices
  SimplifyMesh(vertices,triangles);

  // Second, Simplify mesh by using Fast Quadric Decimation 
  start = std::chrono::system_clock::now();
  FastQuadricDecimator decimator(vertices,triangles);
  size_t target_face_num = (size_t)(decimator.getTriangleCount() * decimation_ratio);
  decimator.simplify_mesh(target_face_num,7.0,true);
  end = std::chrono::system_clock::now();

  std::cout << "| >>> Simplifying Mesh... (Decimation Ratio : "<< decimation_ratio << ")\n";
  std::cout << "| Output Vertices : " << decimator.getVertexCount() << " , Output Triangles : "<< decimator.getTriangleCount() << "\n";
  std::cout << "| Time for Fast Quadric Decimation             : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
  decimator.save_txt(filename);
	// for (auto i: Simplify::vertices)
  // {
  //   fprintf(fp, "v %f %f %f\n", i.p.x, i.p.y, i.p.z);
  // }
  // for(auto i: Simplify::triangles)
  // {
  //   fprintf(fp, "f %d %d %d\n", i.v[0], i.v[1], i.v[2]);
  // }
  
  delete[] h_pos;
  std::cout << "+-------------------------------------------------------------------------------------+\n";
}


/**
 * @brief Function for writing mesh data to txt file for specific class data
 * 
 * @param target_class 
 * @param target_index 
 * @param decimation_ratio 
 */
void WriteTxtFileSeg(std::string target_class, size_t target_idx, std::vector<float3> &vertices, std::vector<int3> &triangles, double decimation_ratio)
{
  std::vector<float3> simple_vertices;
  std::vector<int3> simple_triangles;

  std::string filename = std::string("/home/do/ros2_ws/src/gv_recon/mesh_data_") + target_class + ".txt";
  auto start = std::chrono::system_clock::now();
  // First, Simplify mesh by removing duplicated vertices and removing unreferenced vertices
  SimplifyMeshSeg(target_class, vertices, triangles);

  // Second, Simplify mesh by using Fast Quadric Decimation 
  FastQuadricDecimator decimator(vertices, triangles);
  size_t target_face_num = (size_t)(decimator.getTriangleCount() * decimation_ratio);
  decimator.simplify_mesh(target_face_num,7.0,true);
  auto end = std::chrono::system_clock::now();

  printf("| [%s] Output Vertices : %ld, Output Triangles : %ld\n", target_class.c_str(), decimator.getVertexCount(), decimator.getTriangleCount());
  printf("| [%s] Time for Removing Duplicates and Fast Quadric Decimation : %ld ms\n", target_class.c_str(), std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
  // decimator.save_txt(filename);
	simple_vertices = decimator.getVertices();
  simple_triangles = decimator.getTriangles();

  // for(int i{0};i<5;i++)
  // {
  //   shm_manager[i].init(1000+i,target_segment_classes[i]);
  // }

  shm_manager[target_idx].SendMesh(simple_vertices, simple_triangles);
  
  vertices.clear();
  triangles.clear();
  std::cout << "+-------------------------------------------------------------------------------------+\n";
}

void testSave(){
  if(isMask)
  {
    std::cout << "+---------------------------------- Mesh Generation ----------------------------------+\n";
    
    if(isMultiThread){
      auto start = std::chrono::system_clock::now();
      // Get the segmented Marching Cubes input data from voxel data
      for(int i {0}; i<5; i++){
        cudaMemcpy(d_mc_input, d_voxelRaw, sizeof(uchar) * numVoxels, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        SegmentMCinput(target_segment_classes[i],i);
      }

      auto chk = std::chrono::system_clock::now();

      // Generate thread for each class, and save the mesh data to txt file.
      std::vector<std::thread> threads;
      for (int i {0}; i<5; i++){
        threads.push_back(std::thread {WriteTxtFileSeg, target_segment_classes[i], i, std::ref(Segmented_Vertices[i]), std::ref(Segmented_Triangles[i]), 0.5});
      }
      for (auto& th : threads) th.join();

      auto end = std::chrono::system_clock::now();
      std::cout << "Total Sequential Calculation Time for 5 classes: "<< std::chrono::duration_cast<std::chrono::milliseconds>(chk - start).count() / 1000.0 << "sec\n";
      std::cout << "Total Parallel Processing for 5 classes: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - chk).count() / 1000.0 << "sec\n";
    }
    else{
      auto start = std::chrono::system_clock::now();
      for(int i{0};i<5;i++){
        cudaMemcpy(d_mc_input, d_voxelRaw, sizeof(uchar) * numVoxels, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        SegmentMCinput(target_segment_classes[i],i);
        WriteTxtFile(target_segment_classes[i],i);
      }
      auto end = std::chrono::system_clock::now();
      printf("Total saving Time for 5 classes: %f sec\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);
    }
  }
  else{
    computeIsosurface(); cudaDeviceSynchronize(); 
    WriteTxtFile("all");
  }
}

}