#include <cstdlib>
#include <signal.h>
#include <stdlib.h>


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <GL/freeglut.h>

#include "kernelCall.h"
#include "SyncedSubNode.h"
#include "marchingCubes_kernel.h"
//using std::placeholders::_1;
#include <helper_gl.h>


// /////////////////////////////////////////////////////////////////////////
// // for marching cubes //
// /////////////////////////////////////////////////////////////////////////
#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY 10  // ms
#define EPSILON 5.0f
#define THRESHOLD 0.30f

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
float3 glrotate = make_float3(0.0, 0.0, 0.0);
float3 gltranslate = make_float3(0.0, 0.0, -3.0);

// toggles
bool wireframe = false;
bool animate = true;
bool lighting = true;
bool render = true;
bool compute = true;
/////////////////////////////////////////////////////////////////////////



///////////////////////////////////////
// forward declarations of GL and MC //
///////////////////////////////////////
template <class T>
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
void idle();
void reshape(int w, int h);
void mainMenu(int i);
void animation();
void timerEvent(int value);
bool initGL(int *argc, char **argv);
void initMenus();
void computeFPS();

uchar *loadRawFile(char *filename, int size);
GLuint compileASMShader(GLenum program_type, const char *code);


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

bool cam0_show, cam1_show, cam2_show;
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

int main(int argc, char** argv)
{
  
  for(int i{0};i<3;i++)
  {
    inputDepths[i] = new float[depth_width * depth_height];
    std::cout << "In Main : " << &inputDepths[i] << std::endl;
    inputMasks[i] = new uint8_t[mask_width * mask_height];
  }

  // Initialize ROS
  printf("Program Start.\n");
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<SyncedSubNode>(inputDepths, inputMasks, depth_width, depth_height ,mask_width, mask_height);
  executor.add_node(node);

  ////////////////////
  // cuda test code //
  ////////////////////

  int h_data = 1;
  int* d_data;    
  cudaMalloc((void**)&d_data, sizeof(int));
  cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

  printf("[CUDA test] Host-Device data transfer : %s\n", h_data==1 ? "PASS" : "FAIL"); 
  cudaFree(d_data);



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

  // Read Extrinsic File
  std::ifstream readExtrinsics;
  std::string line2, stringBuffer2;
  std::vector<Matrix4> extrinsicsVector;
  std::vector<Eigen::Matrix4f> extrinsicInv;
  std::vector<Eigen::Matrix4f> extrinsicsVectorEigenInv;
  std::vector<gpu_voxels::Matrix4f> gpuvox_extrinsic;
  gpu_voxels::Matrix4f gpuvox_ExtrArr[3];
  std::vector<float> ExtrinsicsList;
  std::ofstream writeFile;
  std::vector<std::vector<double>> timeMeasureTable;
  
  if(debug_chrono){
    timeMeasureTable.resize(6);
    for (int t = 0; t < 6; t++)
    {
      timeMeasureTable[t].resize(1000);
    }
  }
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
  if(map_size_x==256 && map_size_y == 256) voxel_side_length = icl_core::config::paramOptDefault<float>("voxel_side_length", 0.02f);
  else if(map_size_x==512 && map_size_x == 512) voxel_side_length = icl_core::config::paramOptDefault<float>("voxel_side_length", 0.01f);
  else{
    printf("Map size error!\n map size should be 256 or 512 and x and y should be same.\n");
    exit(EXIT_SUCCESS);
  }

  // Generate a GPU-Voxels instance:
  gvl = gpu_voxels::GpuVoxels::getInstance();

  gvl->initialize(map_size_x, map_size_y, map_size_z, voxel_side_length);
  
  // ks  fx fy cx cy

  //std::vector<float4> ks = {make_float4(1108.512f, 1108.512f, 640.0f, 360.0f),
  //                          make_float4(1108.512f, 1108.512f, 640.0f, 360.0f),
  //                          make_float4(1108.512f, 1108.512f, 640.0f, 360.0f)};
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

  /*
  //////////////////////////
  // Marching Cubes setup //
  //////////////////////////
  setenv("DISPLAY", ":0", 0);
  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA
  // interop.
  if (false == initGL(&argc, argv)) {
    std::cout<<"initGL failed\n";
    exit(EXIT_FAILURE);
  }
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutReshapeFunc(reshape);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  initMenus();
  initMC(argc, argv);
  sdkCreateTimer(&timer);
  // start rendering mainloop
  //glutMainLoop();
  std::cout<< "[DEBUG] : MC init finished!" << std::endl;
  */

  // Create device kernel for call gpu_voxels kernel function 
  DeviceKernel dk(node->_isMask,depth_width, depth_height, numofcamera, ks, extrinsicInv, gpuvox_ExtrArr);

  gvl->addMap(MT_BITVECTOR_VOXELMAP, "voxelmap_1");
  boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap(gvl->getMap("voxelmap_1")->as<voxelmap::BitVectorVoxelMap>());
  std::cout << "[DEBUG] ptrbitVoxmap's m_dev_data (address) : " << ptrbitVoxmap->getVoidDeviceDataPtr() << std::endl;
  
  // update kernerl parameter and set constant memory
  ptrbitVoxmap->updateReconThresh(node->_recon_threshold);
  ptrbitVoxmap->updateVisUnknown(node->_vis_unknown);
  node->_ptrBitVoxMap = ptrbitVoxmap;
  ptrbitVoxmap->setConstMemory(depth_width, depth_height,mask_width,static_cast<float>(mask_width) / depth_width);

  
  size_t iteration = 0;
  // double toDeviceMem, genPointCloud, reconTime, visualTime, transformTime, copyTime, generatemsgTime = 0.0;
  double copyTime, genPointCloud, reconTime,callbackTime,visualTime,syncTime = 0.0;

  LOGGING_INFO(Gpu_voxels, "waiting to start visualizing maps" << endl);
  bool is_loop = true;
  std::chrono::system_clock::time_point loop_start, callback_end, copy_end, reconend, visualend, loop_end;
  std::chrono::duration<double> callback_time, synchronize_time, copy_time, visualtimecount, elapsed_seconds;

  while (rclcpp::ok())
  {
    if(is_loop ) {
      if(debug_chrono) {loop_start = std::chrono::system_clock::now();}
      is_loop = false;}

    // read parameter reconfigure
    cam0_show=node->_master_show;
    cam1_show=node->_sub1_show; 
    cam2_show=node->_sub2_show;
    usleep_time=node->_usleep_time;
    executor.spin_once();
    if(node->_data_received)
    {
      node->_data_received = false;
      is_loop = true;
      std::cout << "[DEBUG] Iteration =========> " << iteration << std::endl;
      if(debug_chrono){
        std::chrono::system_clock::time_point callback_end = std::chrono::system_clock::now();
        std::chrono::duration<double> callback_time = node->callback_duration;
        std::chrono::duration<double> synchronize_time = callback_end - loop_start;
        std::cout << "[DEBUG] : (chrono::system_clock) Synchronization Time         : " << synchronize_time.count()*1000 - callback_time.count()*1000 << "ms" << std::endl;
        timeMeasureTable[0][iteration] = synchronize_time.count()*1000 - callback_time.count()*1000;
        syncTime += synchronize_time.count()*1000 - callback_time.count()*1000;
        timeMeasureTable[1][iteration] = callback_time.count()*1000;
        callbackTime += callback_time.count() * 1000;
        std::cout << "[DEBUG] : (chrono::system_clock) Callback Processing Time     : " << callback_time.count()*1000 << "ms" << std::endl;
      }
    
      
      // toDeviceMem += dk.toDeviceMemory(inputDepths);
      // dk.toDeviceMemoryYoso(inputDepths,inputMasks,mask_width,mask_height);
      dk.toDeviceMemoryYosoArray(inputDepths,inputMasks,mask_width,mask_height);
      if(debug_chrono){
        std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
        std::chrono::duration<double> copy_time = copy_end - callback_end;
        std::cout << "[DEBUG] : (chrono::system_clock) Copy to Device Memory Time   : " << copy_time.count()*1000 << "ms" << std::endl;
        timeMeasureTable[2][iteration] = copy_time.count()*1000;
        copyTime += copy_time.count() * 1000;
      }

      if(node->_isClearMap) ptrbitVoxmap->clearMap();
      float reconTimeOnce=0;
      // reconTimeOnce = dk.generatePointCloudFromDevice3DSpace();
      // this for yoso
      // reconTimeOnce+=dk.RuninsertPclWithYoso(ptrbitVoxmap);
      // reconTime+=reconTimeOnce;
      if(node->_isClearMap) reconTimeOnce = dk.ReconVoxelToDepthtest(ptrbitVoxmap);
      else reconTimeOnce = dk.ReconVoxelWithPreprocess(ptrbitVoxmap);

      // reconTimeOnce= node->_isClearMap ? dk.ReconVoxelToDepthtest(ptrbitVoxmap): dk.ReconVoxelWithPreprocess(ptrbitVoxmap);
      if(debug_chrono){
        reconTime+=reconTimeOnce;
        std::cout << "[DEBUG] : (chrono::system_clock) Reconstruction Time          : " << reconTimeOnce << " ms" << std::endl;
        timeMeasureTable[3][iteration] = reconTimeOnce;
        reconTimeOnce=0;
        std::chrono::system_clock::time_point reconend = std::chrono::system_clock::now();
        gvl->visualizeMap("voxelmap_1");
        std::chrono::system_clock::time_point visualend = std::chrono::system_clock::now();
        std::chrono::duration<double> visualtimecount = visualend - reconend;
        std::cout << "[DEBUG] : (chrono::system_clock) Visualize Map Time           : " << visualtimecount.count() * 1000 << " ms" << std::endl;
        timeMeasureTable[4][iteration] = visualtimecount.count() * 1000;
        visualTime += visualtimecount.count() * 1000;
        timeMeasureTable[5][iteration] = usleep_time;
        std::cout << "[DEBUG] : (chrono::system_clock) Usleep Time                  : " << usleep_time/1000 << " ms" << std::endl;
        usleep(usleep_time);
        gvl->visualizeMap("voxelmap_1");
      }
      else {gvl->visualizeMap("voxelmap_1");usleep(usleep_time); }
      
      if(iteration==50) dk.saveVoxelRaw(ptrbitVoxmap);

      iteration++;

      if(debug_chrono && iteration == 500)
      {
        std::cout << "*******************************************************" << std::endl;
        std::cout << "[During 500 iteration] Check the mean time for task" << std::endl;
        std::cout << "[DEBUG] Synchronization Time         : " << syncTime/iteration << "ms" << std::endl;
        std::cout << "[DEBUG] Callback Processing Time     : " << callbackTime/iteration << "ms" << std::endl;
        std::cout << "[DEBUG] Copy to Device Memory Time   : " << copyTime/iteration << "ms" << std::endl;
        std::cout << "[DEBUG] Reconstruction Time          : " << reconTime/iteration << "ms" << std::endl;
        std::cout << "[DEBUG] Visualize Time               : " << visualTime/iteration << "ms" << std::endl;
        std::cout << "[DEBUG] Usleep Time                  : " << usleep_time/1000 << "ms" << std::endl;
        std::cout << "*******************************************************" << std::endl;

        writeFile.open("/home/do/ros2_ws/src/gv_recon/TimeMeasures.txt");
        // writeFile << "frame" << " " << "SyncTime" << " " << "CallbackTime" << " " << " CopyTime " <<"\n";
        writeFile << "frame" << " " << "SyncTime" << " " << "CallbackTime" << " " << "CopyTime" 
                  << " " << "ReconTime" << " " << "VisTime" << " " << "UsleepTime" << "\n";

        for(int i = 0; i < iteration; i++) 
        {
          writeFile << i << " " << timeMeasureTable[0][i] << " " << timeMeasureTable[1][i] << " " << timeMeasureTable[2][i] << " " 
          << timeMeasureTable[3][i] << " " << timeMeasureTable[4][i] << " " << timeMeasureTable[5][i] <<"\n";
        }
        writeFile.close();

        timeMeasureTable[0].clear();
        timeMeasureTable[1].clear();
        timeMeasureTable[2].clear();
        timeMeasureTable[3].clear();
        timeMeasureTable[4].clear();
        timeMeasureTable[5].clear();

        copyTime, genPointCloud, reconTime,callbackTime,visualTime,syncTime = 0.0;
        // toDeviceMem, genPointCloud, reconTime, visualTime = 0.0;
        iteration = 0;

        std::chrono::system_clock::time_point loop_end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = loop_end - loop_start;
        std::cout << "[DEBUG] : (chrono::system_clock) Total Reconstruction Time    : " << elapsed_seconds.count() * 1000 << " ms" << std::endl;
      }

    }
  }
  for (int i = 0; i < numofcamera; i++) {
    delete[] inputDepths[i];
    delete[] inputMasks[i];
    inputDepths[i] = nullptr;
    inputMasks[i] = nullptr;
  }
  rclcpp::shutdown();
  // delete[] occupied_bit_map;
  LOGGING_INFO(Gpu_voxels, "shutting down" << endl);
  exit(EXIT_SUCCESS);
}



////////////////////////////////////////////////////////////////////////////////
// initialize marching cubes
////////////////////////////////////////////////////////////////////////////////
void initMC(int argc, char **argv) {
  // parse command line arguments
  int n;
  if (checkCmdLineFlag(argc, (const char **)argv, "grid")) {
    n = getCmdLineArgumentInt(argc, (const char **)argv, "grid");
    gridSizeLog2.x = gridSizeLog2.y = gridSizeLog2.z = n;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "gridx")) {
    n = getCmdLineArgumentInt(argc, (const char **)argv, "gridx");
    gridSizeLog2.x = n;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "gridy")) {
    n = getCmdLineArgumentInt(argc, (const char **)argv, "gridy");
    gridSizeLog2.y = n;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "gridz")) {
    n = getCmdLineArgumentInt(argc, (const char **)argv, "gridz");
    gridSizeLog2.z = n;
  }

  char *filename;

  if (getCmdLineArgumentString(argc, (const char **)argv, "file", &filename)) {
    volumeFilename = filename;
  }

  gridSize =
      make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
  gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
  gridSizeShift =
      make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);

  numVoxels = gridSize.x * gridSize.y * gridSize.z;
  voxelSize =
      make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
  maxVerts = gridSize.x * gridSize.y * 100;

  printf("grid: %d x %d x %d = %d voxels\n", gridSize.x, gridSize.y, gridSize.z,
        numVoxels);
  printf("max verts = %d\n", maxVerts);

#if SAMPLE_VOLUME
  // load volume data
  char *path = sdkFindFilePath(volumeFilename, argv[0]);

  if (path == NULL) {
    fprintf(stderr, "Error finding file '%s'\n", volumeFilename);

    exit(EXIT_FAILURE);
  }

  int size = gridSize.x * gridSize.y * gridSize.z * sizeof(uchar);
  uchar *volume = loadRawFile(path, size);
  cudaMalloc((void **)&d_volume, size);
  cudaMemcpy(d_volume, volume, size, cudaMemcpyHostToDevice);
  free(volume);

  createVolumeTexture(d_volume, size);
#endif

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

  if (d_volume) {
    cudaFree(d_volume);
  }
}

void runGraphicsTest(int argc, char **argv) {
  printf("MarchingCubes\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
    printf("[%s]\n", argv[0]);
    printf("   Does not explicitly support -device=n in OpenGL mode\n");
    printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
    printf(" > %s -device=n -file=<reference> -dump=<0/1/2>\n", argv[0]);
    exit(EXIT_SUCCESS);
  }

  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA
  // interop.
  if (false == initGL(&argc, argv)) {
    return;
  }

  // findCudaDevice(argc, (const char **)argv);

  // register callbacks
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutReshapeFunc(reshape);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  initMenus();

  // Initialize CUDA buffers for Marching Cubes
  initMC(argc, argv);

  sdkCreateTimer(&timer);

  // start rendering mainloop
  glutMainLoop();
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
  launch_classifyVoxel(grid, threads, d_voxelVerts, d_voxelOccupied, d_volume,
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
                            d_voxelVertsScan, d_volume, gridSize, gridSizeShift,
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

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
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

void idle() {
  animation();
  glutPostRedisplay();
}

void reshape(int w, int h) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float)w / (float)h, 0.1, 10.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);
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
    sprintf(fps, "CUDA Marching Cubes: %3.1f fps", ifps);

    glutSetWindowTitle(fps);
    fpsCount = 0;

    fpsLimit = ftoi(MAX(1.f, ifps));
    sdkResetTimer(&timer);
  }
}


////////////////////////////////////////////////////////////////////////////////
// Load raw data from disk
////////////////////////////////////////////////////////////////////////////////

uchar *loadRawFile(char *filename, int size) {
  FILE *fp = fopen(filename, "rb");

  if (!fp) {
    fprintf(stderr, "Error opening file '%s'\n", filename);
    return 0;
  }

  uchar *data = (uchar *)malloc(size);
  size_t read = fread(data, 1, size, fp);
  fclose(fp);

  printf("Read '%s', %d bytes\n", filename, (int)read);
  return data;
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

