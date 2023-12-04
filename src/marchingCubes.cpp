
#include "marchingCubes.h"
#include <helper_gl.h>
#include "Simplify.h"


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

bool isValid = false;
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

std::map<std::string, unsigned char> mask_decode_map;
unsigned char offset = 6;


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


void SegmentMCinput(std::string target_class){
  // first, data copy.
  cudaMemcpy(d_mc_input, d_voxelRaw, sizeof(uchar) * numVoxels, cudaMemcpyDeviceToDevice);
  // second, choose the class. and call the kernel change the voxel raw data.
  launch_generateMCInput(gridSize, d_mc_input, mask_decode_map[target_class]);
  // third, compute isosurface.
  computeIsosurface(); cudaDeviceSynchronize();
  // fourth, write the file.
  WriteTxtFile(target_class);
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
      if(isMask)
      {
        SegmentMCinput("person");
        SegmentMCinput("chair");
        SegmentMCinput("table");
        SegmentMCinput("wall");
        SegmentMCinput("floor");
      }
      else{
        computeIsosurface(); cudaDeviceSynchronize(); 
        WriteTxtFile("all");
      }
      

      

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

void removeDuplicatedVertexOpen3d(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3d> &normals, std::vector<Eigen::Vector3i> &triangles)
{
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  open3d::geometry::TriangleMesh mesh;
  std::shared_ptr<open3d::geometry::TriangleMesh> output;
  mesh.vertices_ = vertices;
  mesh.vertex_normals_ = normals;
  mesh.triangles_ = triangles;

  mesh.RemoveDuplicatedVertices();
  mesh.RemoveUnreferencedVertices();

  vertices = output->vertices_;
  normals = output->vertex_normals_;
  triangles = output->triangles_;

  end = std::chrono::system_clock::now();
  printf("Reducing Duplicated Vertex Time: %f sec\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);
  printf("Total Vertices : %ld\n", vertices.size());
  printf("Total Normals : %ld\n", normals.size());
  printf("Total Triangles : %ld\n", triangles.size());
}

void RemoveDuplicatedVertex(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3d> &normals, std::vector<Eigen::Vector3i> &triangles)
{
  typedef std::tuple<double, double, double> Coordinate3;
  std::unordered_map<Coordinate3, size_t, hash_tuple<Coordinate3>>point_to_old_index;
  std::vector<int> index_old_to_new(vertices.size());
  size_t old_vertex_num = vertices.size();
  size_t k = 0;                                  // new index
  for (size_t i = 0; i < old_vertex_num; i++) {  // old index
      Coordinate3 coord = std::make_tuple(vertices[i](0), vertices[i](1),
                                          vertices[i](2));
      if (point_to_old_index.find(coord) == point_to_old_index.end()) {
          point_to_old_index[coord] = i;
          vertices[k] = vertices[i];
          normals[k] = normals[i];
          index_old_to_new[i] = (int)k;
          k++;
      } else {
          index_old_to_new[i] = index_old_to_new[point_to_old_index[coord]];
      }
  }
  vertices.resize(k);
  normals.resize(k);
  if (k < old_vertex_num) {
      for (auto &triangle : triangles) {
          triangle(0) = index_old_to_new[triangle(0)];
          triangle(1) = index_old_to_new[triangle(1)];
          triangle(2) = index_old_to_new[triangle(2)];
      }
      // if (HasAdjacencyList()) {
      //     ComputeAdjacencyList();
      // }
  }
}

void RemoveUnreferencedVertices(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3d> &normals, std::vector<Eigen::Vector3i> &triangles)
{
  std::vector<bool> vertex_has_reference(vertices.size(), false);
  for (const auto &triangle : triangles) {
      vertex_has_reference[triangle(0)] = true;
      vertex_has_reference[triangle(1)] = true;
      vertex_has_reference[triangle(2)] = true;
  }
  std::vector<int> index_old_to_new(vertices.size());
  size_t old_vertex_num = vertices.size();
  size_t k = 0;                                  // new index
  for (size_t i = 0; i < old_vertex_num; i++) {  // old index
      if (vertex_has_reference[i]) {
          vertices[k] = vertices[i];
          normals[k] = normals[i];
          index_old_to_new[i] = (int)k;
          k++;
      } else {
          index_old_to_new[i] = -1;
      }
  }
  vertices.resize(k);
  normals.resize(k);
  if (k < old_vertex_num) {
      for (auto &triangle : triangles) {
          triangle(0) = index_old_to_new[triangle(0)];
          triangle(1) = index_old_to_new[triangle(1)];
          triangle(2) = index_old_to_new[triangle(2)];
      }
  //     if (HasAdjacencyList()) {
  //         ComputeAdjacencyList();
  //     }
  // }
  }
}

void SimplifyMesh(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3d> &normals, std::vector<Eigen::Vector3i> &triangles)
{
  size_t original_vertex_size = vertices.size();
  size_t original_triangle_size = triangles.size();

  auto t_start = std::chrono::steady_clock::now();
  RemoveDuplicatedVertex(vertices, normals, triangles);

  auto t_chk = std::chrono::steady_clock::now();
  RemoveUnreferencedVertices(vertices, normals, triangles);

  auto t_end = std::chrono::steady_clock::now();

  printf("| Number of Vertex and Normal changed from %ld to %ld\n", original_vertex_size, vertices.size());
  printf("| Number of Triangle changed from %ld to %ld\n", original_triangle_size, triangles.size());
  printf("| Time for removing duplicated vertices: %f sec\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(t_chk - t_start).count() / 1000.0);
  printf("| Time for removing unreferenced vertices: %f sec\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_chk).count() / 1000.0);
  printf("+-------------------------------------------------------------+\n");
}

void WriteTxtFile(std::string target_class)
{
  std::cout<<"+------------------- Saving " << target_class << " -------------------+\n";
  std::string filename = std::string("/home/do/ros2_ws/src/gv_recon/mesh_data_") + target_class + ".txt";

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  // Copy device data to host data
  std::cout<<"| Start Saving Mesh Data...\n";
  float4* h_pos = new float4[totalVerts];
  float4* h_normal = new float4[totalVerts];
  cudaMemcpy(h_pos, d_pos, sizeof(float) * totalVerts * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_normal, d_normal, sizeof(float) * totalVerts * 4, cudaMemcpyDeviceToHost);

  // Open target obj file to write
	FILE *fp=fopen(filename.c_str(), "w");
  std::cout<<"| Converting...\n";

  // For Mesh simplification by using Open3D
  std::vector<Eigen::Vector3d> vertices, normals;
  std::vector<Eigen::Vector3i> triangles;
  for (auto i{0}; i<totalVerts; ++i)
  {
    auto const& v = h_pos[i];
    auto const& vn = h_normal[i];
    vertices.push_back(Eigen::Vector3d(v.x, v.y, v.z));
    normals.push_back(Eigen::Vector3d(vn.x, vn.y, vn.z));
    if(i%3==2) triangles.push_back(Eigen::Vector3i(i-2, i-1, i));
  }
  end = std::chrono::system_clock::now();
  printf("| Data Copy and Convert Time: %f sec\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);
  // removeDuplicatedVertexOpen3d(vertices,normals,triangles);
  SimplifyMesh(vertices,normals,triangles);

	// vertices and normals
	for (auto i: vertices)
  {
    fprintf(fp, "v %f %f %f\n", i.x(), i.y(), i.z());
  }
	for (auto i: normals)
  {
    fprintf(fp, "vn %f %f %f\n", i.x(), i.y(), i.z());
  }
  for (auto i: triangles)
  {
    fprintf(fp, "f %d %d %d\n", i.x(), i.y(), i.z());
  }


  fclose(fp);
  delete[] h_pos;
  delete[] h_normal;
  std::cout<<"[DEBUG] Saved.\n========================================\n";
}