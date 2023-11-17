#include <cstdlib>
#include <signal.h>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <string>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <GL/freeglut.h>

#include "kernelCall.h"
#include "SyncedSubNode.h"
#include "marchingCubes_kernel.h"
//using std::placeholders::_1;
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
bool isValid = false;

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


///////////////////////////////////////
// forward declarations of GL and MC //
///////////////////////////////////////
void renderIsosurface();
void cleanup();
void initMC(int argc, char **argv);
void runGraphicsTest(int argc, char **argv); 
void computeIsosurface();
void createVBO(GLuint *vbo, unsigned int size);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_resource);
void display();
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void reshape(int w, int h);
void mainMenu(int i);
void animation();
void timerEvent(int value);
bool initGL(int *argc, char **argv);
void initMenus();
void computeFPS();
void WriteTxtFile();
GLuint compileASMShader(GLenum program_type, const char *code);


/////////////////////////////////////
// ROS2 gpu_voxels Variables Setup //
/////////////////////////////////////

using boost::dynamic_pointer_cast;
using boost::shared_ptr;

enum CamType
{
  AzureKinect,
  IsaacSimCam,
};

enum AzureDepthMode
{
  NFOV_2x2Binned,
  NFOV_UNBinned,
  WFOV_2x2Binned,
  WFOV_UNBinned,
  RGB_R1536p,
  RGB_720,
};

int AzureMode = AzureDepthMode(RGB_720);
const int DepthWidth[6] = {320, 640, 512, 1024, 2048, 1280};
const int DepthHeight[6] = {288, 576, 512, 1024, 1536, 720};

const int depth_width = DepthWidth[AzureMode];
const int depth_height = DepthHeight[AzureMode];

// replace vector to c array
float* inputDepths[3] {nullptr};
uint8_t* inputMasks[3] {nullptr};
uint16_t mask_width=640;
uint16_t mask_height=360;
shared_ptr<GpuVoxels> gvl;




void ctrlchandler(int)
{
  printf("User Interrupt 'Ctrl + C' : shutting down...\n");
  for (int i = 0; i < sizeof(inputDepths)/sizeof(float*); i++) {
    delete[] inputDepths[i];
    delete[] inputMasks[i];
    inputDepths[i] = nullptr;
    inputMasks[i] = nullptr;
  }
  rclcpp::shutdown();
  exit(EXIT_SUCCESS);
}

void killhandler(int)
{
  rclcpp::shutdown();
  exit(EXIT_SUCCESS);
}

/**
 * @brief Get the Device Voxel Data Object
 * 
 * @param ptrbitVoxmap : shared pointer of voxel map what type is gpu_voxels::voxelmap::BitVectorVoxelMap
 * @param numVoxels : number of voxels
 */
void getVoxelData(boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap, const int numVoxels)
{
  mtx.lock();
	cudaMemset(d_voxelRaw, 0, sizeof(uchar) * numVoxels);
	// launch kernel to get voxel data
	ptrbitVoxmap->getVoxelRaw(d_voxelRaw);
	cudaDeviceSynchronize();
  isValid = true;
  mtx.unlock();
}


void run_ros2_with_gpuVoxels()
{
  mtx.lock();
  // Initialize ROS
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<SyncedSubNode>(inputDepths, inputMasks, depth_width, depth_height ,mask_width, mask_height);
  executor.add_node(node);

  // Get Parameter
  const int numofcamera=node->_num_cameras;
  const int map_size_x=node->_map_size_x;
  const int map_size_y=node->_map_size_y;
  const int map_size_z=node->_map_size_z;
  const bool debug_chrono=node->_debug_chrono;
  float voxel_side_length;
  int usleep_time=node->_usleep_time;
  if(!node->_isMask) {
    for(int i{0};i<3;i++) {delete inputMasks[i];} 
  }
  // Set Intrinsic
  std::vector<float4> ks;
  if(AzureMode == AzureDepthMode(RGB_R1536p)){
    ks = {make_float4(974.333f, 973.879f, 1019.83f, 782.927f),
          make_float4(974.243f, 974.095f, 1021.18f, 771.734f),
          make_float4(971.426f, 971.43f, 1023.2f, 776.102f)
          };
  }
  else if(AzureMode == AzureDepthMode(RGB_720)){
    ks = {make_float4(608.958f, 608.674f, 637.205f, 369.142f),
          make_float4(608.902f, 608.809f, 638.053f, 362.146f),
          make_float4(607.141f, 607.143f, 639.314f, 364.876f)
          };
  }

  // Read Extrinsic File
  std::ifstream readExtrinsics;
  std::string line2, stringBuffer2;
  std::vector<Matrix4> extrinsicsVector;
  std::vector<Eigen::Matrix4f> extrinsicInv;
  std::vector<Eigen::Matrix4f> extrinsicsVectorEigenInv;
  std::vector<gpu_voxels::Matrix4f> gpuvox_extrinsic;
  gpu_voxels::Matrix4f gpuvox_ExtrArr[3];
  std::vector<float> ExtrinsicsList;
  
  
  gpu_voxels::Matrix4f tempG;

  readExtrinsics.open("/home/do/ros2_ws/src/gv_recon/ExtrinsicFile.txt");
  if (readExtrinsics.is_open()) ////tx ty tz r p y
  {
    printf("Reading Extrinsic.\n");
      while ( getline (readExtrinsics,line2) )
      {
        std::istringstream ss2(line2);
        while (getline(ss2, stringBuffer2, ' '))
        {
        double d2;
        std::stringstream(stringBuffer2) >> d2;
        ExtrinsicsList.push_back(d2);
        // std::cout<<d2<<" ";
        }
      }
      readExtrinsics.close();
  }

  for (int i=0; i<3; i++)
	{

    Eigen::Matrix4f temp ;
    temp = setExtrinsicInvEigen((ExtrinsicsList[0 + i*6]),(ExtrinsicsList[1 + i*6]),
                                (ExtrinsicsList[2 + i*6]),(ExtrinsicsList[3 + i*6]),
                                (ExtrinsicsList[4 + i*6]),(ExtrinsicsList[5 + i*6]));
    tempG = setExtrinsicGVox((ExtrinsicsList[0 + i*6]),(ExtrinsicsList[1 + i*6]), 
                              (ExtrinsicsList[2 + i*6]),(ExtrinsicsList[3 + i*6]), 
                              (ExtrinsicsList[4 + i*6]),(ExtrinsicsList[5 + i*6])); 
    // std::cout<< ExtrinsicsList[0 + i*6] << " " <<ExtrinsicsList[2 + i*6] <<" " <<  ExtrinsicsList[5 + i*6] << std::endl;
    extrinsicsVectorEigenInv.push_back(temp);
    extrinsicInv.push_back(temp);
    gpuvox_extrinsic.push_back(tempG);
    gpuvox_ExtrArr[i] = tempG;
  }


  printf("============================== Extrinsic ===============================\n");
  for(int i{0};i<3;i++){std::cout<< gpuvox_ExtrArr[i] << std::endl;}
  printf("========================================================================\n");
  icl_core::logging::initialize();
  printf("============================= Device Info ==============================\n");
  gpu_voxels::getDeviceInfos();
  printf("========================================================================\n");
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  // recommand map_size_x and map_size_y are same. 
  if(map_size_x==256 && map_size_y == 256) {
    voxel_side_length = icl_core::config::paramOptDefault<float>("voxel_side_length", 0.02f);
    gridSizeLog2 = make_uint3(8, 8, 8);
    gridSizeShift = make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);
    gridSize = make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
    gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
    voxelSize = make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
    numVoxels = gridSize.x * gridSize.y * gridSize.z;
    maxVerts = gridSize.x * gridSize.y * 100;
  }
  else if(map_size_x==512 && map_size_x == 512){
    voxel_side_length = icl_core::config::paramOptDefault<float>("voxel_side_length", 0.01f);
  }
  else{
    printf("Map size error!\n map size should be 256 or 512 and x and y should be same.\n");
    exit(EXIT_SUCCESS);
  }

  // Generate a GPU-Voxels instance:
  gvl = gpu_voxels::GpuVoxels::getInstance();
  gvl->initialize(map_size_x, map_size_y, map_size_z, voxel_side_length);
  gvl->addMap(MT_BITVECTOR_VOXELMAP, "voxelmap_1");
  
  boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap(gvl->getMap("voxelmap_1")->as<voxelmap::BitVectorVoxelMap>());
  std::cout << "[DEBUG] ptrbitVoxmap's m_dev_data (address) : " << ptrbitVoxmap->getVoidDeviceDataPtr() << std::endl;
  
   // update kernerl parameter and set constant memory
  ptrbitVoxmap->updateReconThresh(node->_recon_threshold);
  ptrbitVoxmap->updateVisUnknown(node->_vis_unknown);
  node->_ptrBitVoxMap = ptrbitVoxmap;
  ptrbitVoxmap->setConstMemory(depth_width, depth_height,mask_width,static_cast<float>(mask_width) / depth_width);
  const int numVoxels = ptrbitVoxmap->getDimensions().x * ptrbitVoxmap->getDimensions().y * ptrbitVoxmap->getDimensions().z;
	
  
  cudaMalloc(&d_voxelRaw, sizeof(uchar) * numVoxels);
  mtx.unlock();


  // Create device kernel for call gpu_voxels kernel function 
  DeviceKernel dk(node->_isMask,depth_width, depth_height, numofcamera, ks, extrinsicInv, gpuvox_ExtrArr);
  size_t iteration = 0;
  
  LOGGING_INFO(Gpu_voxels, "waiting to start visualizing maps" << endl);
  // Reconstrction main loop
  while (rclcpp::ok())
  {
    // read parameter reconfigure
    usleep_time=node->_usleep_time;
    executor.spin_once();
    if(node->_data_received)
    {
      node->_data_received = false;
      std::cout << "[DEBUG] Iteration =========> " << iteration << std::endl;
      dk.toDeviceMemoryYosoArray(inputDepths,inputMasks,mask_width,mask_height);
      float reconTimeOnce{0};
      if(node->_isClearMap) 
      {
        ptrbitVoxmap->clearMap();
        reconTimeOnce = dk.ReconVoxelToDepthtest(ptrbitVoxmap);
      }
      else reconTimeOnce = dk.ReconVoxelWithPreprocess(ptrbitVoxmap);

      // gvl->visualizeMap("voxelmap_1");

      getVoxelData(ptrbitVoxmap, numVoxels);
      usleep(usleep_time);
      iteration++;
    }
  }

  for (int i = 0; i < numofcamera; i++) {
    delete[] inputDepths[i];
    delete[] inputMasks[i];
    inputDepths[i] = nullptr;
    inputMasks[i] = nullptr;
  }
}

void run_glut_with_marchingCubes(int argc, char** argv)
{
  mtx.lock();
  printf("=============================== MC INFO ================================\n");
  printf("grid: %d x %d x %d = %d voxels\n", gridSize.x, gridSize.y, gridSize.z, numVoxels);
  printf("max verts = %d\n", maxVerts);
  createVolumeTexture(d_voxelRaw, gridSize.x * gridSize.y * gridSize.z * sizeof(uchar));
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

  if (d_voxelRaw) {
    cudaFree(d_voxelRaw);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void computeIsosurface() {
  mtx.lock();
  if(!isValid){mtx.unlock(); return;}
  
  isValid = false;
  int threads = 128;
  dim3 grid(numVoxels / threads, 1, 1);

  // get around maximum grid size of 65535 in each dimension
  if (grid.x > 65535) {
    grid.y = grid.x / 32768;
    grid.x = 32768;
  }

  // calculate number of vertices need per voxel
  launch_classifyVoxel(grid, threads, d_voxelVerts, d_voxelOccupied, d_voxelRaw,
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
                            d_voxelVertsScan, d_voxelRaw, gridSize, gridSizeShift,
                            gridSizeMask, voxelSize, isoValue, activeVoxels,
                            maxVerts);
#else
  launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
                          d_voxelVertsScan, gridSize, gridSizeShift,
                          gridSizeMask, voxelSize, isoValue, activeVoxels,
                          maxVerts);
#endif
  mtx.unlock();
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
      WriteTxtFile();
      
  }

  printf("isoValue = %f\n", isoValue);
  printf("voxels = %d\n", activeVoxels);
  printf("verts = %d\n", totalVerts);
  printf("occupancy: %d / %d = %.2f%%\n", activeVoxels, numVoxels,
        activeVoxels * 100.0f / (float)numVoxels);

  if (!compute) {
    computeIsosurface();
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
    computeIsosurface();
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

void WriteTxtFile()
{
  char* filename = "/home/do/ros2_ws/src/gv_recon/mesh_data.txt";

  // Copy device data to host data
  std::cout<<"[DEBUG] Saving Vertices and Normals...\n";
  float4* h_pos = new float4[totalVerts];
  float4* h_normal = new float4[totalVerts];
  cudaMemcpy(h_pos, d_pos, sizeof(float) * totalVerts * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_normal, d_normal, sizeof(float) * totalVerts * 4, cudaMemcpyDeviceToHost);
  std::cout<<"[DEBUG] Copy Done.\n";

  // Open target obj file to write
	FILE *fp=fopen(filename, "w");
  std::cout<<"[DEBUG] Writing...\n";

	// vertices and normals
	for (auto i = 0; i < totalVerts; ++i)
	{
    auto const& v = h_pos[i];
    auto const& vn = h_normal[i];
		fprintf(fp, "v %f %f %f\n", v.x, v.y, v.z);
    fprintf(fp, "vn %f %f %f\n", vn.x, vn.y, vn.z);
  }

  fclose(fp);
  delete[] h_pos;
  delete[] h_normal;
  std::cout<<"[DEBUG] Saved.\n";
}

// Main Function
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  for(int i{0};i<3;i++)
  {
    inputDepths[i] = new float[depth_width * depth_height];
    std::cout << "In Main : " << &inputDepths[i] << std::endl;
    inputMasks[i] = new uint8_t[mask_width * mask_height];
  }
  // Start ROS2 and gpu_voxels thread
  std::thread ros2_thread(run_ros2_with_gpuVoxels);
  sleep(1);
  // Start MarchingCubes and glut thread
  std::thread glut_thread(run_glut_with_marchingCubes, argc, argv);

  glut_thread.join();
  ros2_thread.join();
  
  
  cleanup();
  rclcpp::shutdown();
  LOGGING_INFO(Gpu_voxels, "shutting down" << endl);
  exit(EXIT_SUCCESS);
}