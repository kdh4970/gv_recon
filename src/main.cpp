// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Christian Jülg
 * \date    2015-08-07
 * \author  Andreas Hermann
 * \date    2016-12-24
 *
 * This demo calcuates a distance field on the pointcloud
 * subscribed from a ROS topic.
 * Two virtual meausrement points are places in the scene
 * from which the clearance to their closest obstacle from the live pointcloud
 * is constantly measured (and printed on terminal).
 *
 * Place the camera so it faces you in a distance of about 1 to 1.5 meters.
 *
 * Start the demo and then the visualizer.
 * Example parameters:  ./build/bin/distance_ros_demo -e 0.3 -f 1 -s 0.008  #voxel_size 8mm, filter_threshold 1, erode less than 30% occupied  neighborhoods
 *
 * To see a small "distance hull":
 * Right-click the visualizer and select "Render Mode > Distance Maps Rendermode > Multicolor gradient"
 * Press "s" to disable all "distance hulls"
 * Press "Alt-1" an then "1" to enable drawing of SweptVolumeID 11, corresponding to a distance of "1"
 *
 * To see a large "distance hull":
 * Press "ALT-t" two times, then "s" two times.
 * You will see the Kinect pointcloud inflated by 10 Voxels.
 * Use "t" to switch through 3 slicing modes and "a" or "q" to move the slice.
 *
 */
//----------------------------------------------------------------------

#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/common_defines.h>

#include <pcl/point_types.h>
#include <icl_core_config/Config.h>
#include <chrono>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "cv_bridge/cv_bridge.h"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include <functional>
#include <memory>
#include <stdint.h>

#include "kernelCall.h"
#include "SyncedSubNode.h"

//using std::placeholders::_1;

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


int main(int argc, char* argv[])
{
  // initialize ROS
  printf("Program Start.\n");
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;

  for(int i{0};i<3;i++)
  {
    inputDepths[i] = new float[depth_width * depth_height];
    std::cout << "In Main : " << &inputDepths[i] << std::endl;
    inputMasks[i] = new uint8_t[mask_width * mask_height];
  }

  auto node = std::make_shared<SyncedSubNode>(inputDepths, inputMasks, depth_width, depth_height ,mask_width, mask_height);
  executor.add_node(node);
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