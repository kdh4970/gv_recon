#include <cstdlib>
#include <signal.h>
#include <stdlib.h>
#include "cuda/DeviceKernel.h"
#include "SyncedSubNode.h"

#include "marchingCubes.h"

//using std::placeholders::_1;



/////////////////////////////////////
// ROS2 gpu_voxels Variables Setup //
/////////////////////////////////////

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
  isMask = node->_isMask;
  if(!isMask) {
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
  DeviceKernel dk(isMask,depth_width, depth_height, numofcamera, ks, extrinsicInv, gpuvox_ExtrArr);
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
      
      MarchingCubes::getVoxelData(ptrbitVoxmap, numVoxels);
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
  std::thread glut_thread(MarchingCubes::run_glut_with_marchingCubes, argc, argv);

  glut_thread.join();
  ros2_thread.join();
  
  
  MarchingCubes::cleanup();
  rclcpp::shutdown();
  LOGGING_INFO(Gpu_voxels, "shutting down" << endl);
  exit(EXIT_SUCCESS);
}