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
#include "kernels.h"
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

const int _depth_width = DepthWidth[AzureMode];
const int _depth_height = DepthHeight[AzureMode];

// replace vector to c array
float* inputDepths[3] {nullptr};
uint8_t* inputMasks[3] {nullptr};

bool cam0_data_received, cam1_data_received, cam2_data_received;
bool mask0_received, mask1_received, mask2_received;
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

class kfusionNode : public rclcpp::Node
{
public:
  bool _master_show, _sub1_show, _sub2_show;
  bool _isSync, _isMask, _isClearMap, _vis_unknown, _debug_chrono;
  int _usleep_time,_num_cameras,_map_size_x,_map_size_y,_map_size_z;

  float _voxel_side_length, _recon_threshold;
  std::string _master_cam_topic, _sub1_cam_topic, _sub2_cam_topic;
  std::string _master_yoso_topic, _sub1_yoso_topic, _sub2_yoso_topic;
  rclcpp::CallbackGroup::SharedPtr _parallel_group;
  boost::shared_ptr<voxelmap::BitVectorVoxelMap> _ptrBitVoxMap;
  std::chrono::duration<double> callback_duration;
  rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";
    for (const auto &parameter : parameters)
    {
      if (parameter.get_name() == "master_cam_show" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL){
        _master_show = parameter.as_bool();
      }
      if (parameter.get_name() == "sub1_cam_show" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL){
        _sub1_show = parameter.as_bool();
      }
      if (parameter.get_name() == "sub2_cam_show" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL){
        _sub2_show = parameter.as_bool();
      }
      if (parameter.get_name() == "usleep_time" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER){
        _usleep_time = parameter.as_int();
      }
      if (parameter.get_name() == "isClearMap" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL){
        _isClearMap = parameter.as_bool();
        RCLCPP_INFO(this->get_logger(), "Parameter 'isClearMap' changed: %s", _isClearMap?"True":"False");
      }
      if (parameter.get_name() == "recon_threshold" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE){
        _recon_threshold = parameter.as_double();
        _ptrBitVoxMap->updateReconThresh(_recon_threshold);
        RCLCPP_INFO(this->get_logger(), "Parameter 'recon_threshold' changed: %.5f", _recon_threshold);
      }
      if (parameter.get_name() == "vis_unknown" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL){
        _vis_unknown = parameter.as_bool();
        _ptrBitVoxMap->updateVisUnknown(_vis_unknown);
        RCLCPP_INFO(this->get_logger(), "Parameter 'vis_unknown' changed: %s", _vis_unknown?"True":"False");
      }

    }
    return result;
  }  

  kfusionNode()
  : Node("kfusion_node")
  {
    rclcpp::Parameter use_sim_time("use_sim_time", rclcpp::ParameterValue(true));
    this->set_parameter(use_sim_time);
    this->declare_parameter("num_cameras", 3);
    this->declare_parameter("master_cam_show", true);
    this->declare_parameter("sub1_cam_show", true);
    this->declare_parameter("sub2_cam_show", true);
    this->declare_parameter("usleep_time", 0);
    this->declare_parameter("map_size_x", 512);
    this->declare_parameter("map_size_y", 512);
    this->declare_parameter("map_size_z", 256);
    this->declare_parameter("isSync", true);
    this->declare_parameter("isMask", true);
    this->declare_parameter("isClearMap",false);
    this->declare_parameter("recon_threshold",0.03);
    this->declare_parameter("vis_unknown", false);
    this->declare_parameter("debug_chrono", false);
    
    _num_cameras = this->get_parameter("num_cameras").as_int();
    if(_num_cameras < 1){
      RCLCPP_ERROR(this->get_logger(), "num_cameras must be greater than 0");
      exit(EXIT_FAILURE);
    }
    if(_num_cameras >= 1) {
      _master_cam_topic = "/azure_kinect/master/depth_to_rgb/image_raw";
      _master_yoso_topic = "/yoso_node/master";
    }
    if(_num_cameras >= 2) {
      _sub1_cam_topic = "/azure_kinect/sub1/depth_to_rgb/image_raw";
      _sub1_yoso_topic = "/yoso_node/sub1";
    }
    if(_num_cameras >= 3) {
      _sub2_cam_topic = "/azure_kinect/sub2/depth_to_rgb/image_raw";
      _sub2_yoso_topic = "/yoso_node/sub2";
    }
    _map_size_x=this->get_parameter("map_size_x").as_int();
    _map_size_y=this->get_parameter("map_size_y").as_int();
    _map_size_z=this->get_parameter("map_size_z").as_int();
    _master_show = this->get_parameter("master_cam_show").as_bool();
    _sub1_show = this->get_parameter("sub1_cam_show").as_bool(); ;
    _sub2_show = this->get_parameter("sub2_cam_show").as_bool();
    _usleep_time = this->get_parameter("usleep_time").as_int();
    _isSync = this->get_parameter("isSync").as_bool();
    _isMask = this->get_parameter("isMask").as_bool();
    _isClearMap = this->get_parameter("isClearMap").as_bool();
    _recon_threshold = this->get_parameter("recon_threshold").as_double();
    _vis_unknown = this->get_parameter("vis_unknown").as_bool();
    _debug_chrono = this->get_parameter("debug_chrono").as_bool();

    if(!_isSync){
      if(_num_cameras >= 1){
        depthsub_0 = this->create_subscription<sensor_msgs::msg::Image>(_master_cam_topic, 20, std::bind(&kfusionNode::callback_from_image_topic, this, std::placeholders::_1));
        yososub_0= this->create_subscription<sensor_msgs::msg::Image>(_master_yoso_topic, 2, std::bind(&kfusionNode::yosoCallBack, this, std::placeholders::_1));
      }
      if(_num_cameras >= 2){
        depthsub_1 = this->create_subscription<sensor_msgs::msg::Image>(_sub1_cam_topic, 20, std::bind(&kfusionNode::callback_from_image_topic1, this, std::placeholders::_1));
        yososub_1= this->create_subscription<sensor_msgs::msg::Image>(_sub1_yoso_topic, 2, std::bind(&kfusionNode::yosoCallBack1, this, std::placeholders::_1));
      }
      if(_num_cameras >= 3){
        depthsub_2 = this->create_subscription<sensor_msgs::msg::Image>(_sub2_cam_topic, 20, std::bind(&kfusionNode::callback_from_image_topic2, this, std::placeholders::_1));
        yososub_2= this->create_subscription<sensor_msgs::msg::Image>(_sub2_yoso_topic, 2, std::bind(&kfusionNode::yosoCallBack2, this, std::placeholders::_1));
      }
    }
    if(_isSync){
      if(_num_cameras >= 1) mf_depthsub_0.subscribe(this, _master_cam_topic);
      if(_num_cameras >= 2) mf_depthsub_1.subscribe(this, _sub1_cam_topic);
      if(_num_cameras >= 3) mf_depthsub_2.subscribe(this, _sub2_cam_topic);
      if(_isMask){
        if(_num_cameras >= 1)mf_yososub_0.subscribe(this, _master_yoso_topic);
        if(_num_cameras >= 2)mf_yososub_1.subscribe(this, _sub1_yoso_topic);
        if(_num_cameras >= 3)mf_yososub_2.subscribe(this, _sub2_yoso_topic);
      }
      
      if(_isMask){
        if(_num_cameras == 1){
          syncT1.reset(new SyncT1(_SyncPolicyT1(60), mf_depthsub_0, mf_yososub_0));
          syncT1->registerCallback(std::bind(&kfusionNode::callbackT1, this, std::placeholders::_1, std::placeholders::_2));
        }
        if(_num_cameras == 2){
          syncT2.reset(new SyncT2(_SyncPolicyT2(60), mf_depthsub_0, mf_depthsub_1, mf_yososub_0, mf_yososub_1));
          syncT2->registerCallback(std::bind(&kfusionNode::callbackT2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        }
        if(_num_cameras == 3){
          syncT3.reset(new SyncT3(_SyncPolicyT3(60), mf_depthsub_0, mf_depthsub_1, mf_depthsub_2, mf_yososub_0, mf_yososub_1, mf_yososub_2));
          syncT3->registerCallback(std::bind(&kfusionNode::callbackT3, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
        }
      }
      else {
        if(_num_cameras == 1){
          depthsub_0 = this->create_subscription<sensor_msgs::msg::Image>(_master_cam_topic, 20, std::bind(&kfusionNode::callback_from_image_topic, this, std::placeholders::_1));
        }
        if(_num_cameras == 2){
          syncD2.reset(new SyncD2(_SyncPolicyD2(30),mf_depthsub_0, mf_depthsub_1));
          syncD2->registerCallback(std::bind(&kfusionNode::callbackD2, this, std::placeholders::_1, std::placeholders::_2));
        }
        if(_num_cameras == 3){
          syncD3.reset(new SyncD3(_SyncPolicyD3(30),mf_depthsub_0, mf_depthsub_1, mf_depthsub_2));
          syncD3->registerCallback(std::bind(&kfusionNode::callbackD3, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        }
      }
      
      
    }
    
    printf("============================== Node Info ===============================\n");
    printf("Num of Cam: %d\n", _num_cameras);
    printf("Depth topic : %s, %s, %s\n", _master_cam_topic.c_str(), _sub1_cam_topic.c_str(), _sub2_cam_topic.c_str());
    if(_isMask) printf("Yoso topic : %s, %s, %s\n", _master_yoso_topic.c_str(), _sub1_yoso_topic.c_str(), _sub2_yoso_topic.c_str());
    printf("Usleep time: %dms\n", static_cast<int>(_usleep_time/1000));
    printf("Voxel Map Size : %d X %d X %d\n", _map_size_x, _map_size_y, _map_size_z);
    printf("Use Synchronizer : %s\n", _isSync ? "True" : "False");
    printf("Use Mask : %s\n", _isMask ? "True" : "False");
    printf("Map Update Method : %s\n", _isClearMap ? "Clear Map" : "Kernel Function Procesing");
    printf("Projection Distance Error threshold : %.5f\n", _recon_threshold);
    printf("Visualize Unknown Voxels : %s\n", _vis_unknown ? "True" : "False");
    printf("Debug Chrono : %s\n", _debug_chrono ? "True" : "False");
    printf("========================================================================\n");
    this->set_on_parameters_set_callback(
            std::bind(&kfusionNode::parametersCallback, this, std::placeholders::_1));
  }
  void callback_from_image_topic(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    int depthRow = msg->height;
    int depthCol = msg->width;
    
    cv::Mat depth = cv_bridge::toCvShare(msg, "32FC1")->image;
    memset(inputDepths[0], 0, depthRow*depthCol*sizeof(float));
    memcpy(inputDepths[0], depth.data, depth.total()*sizeof(float));
    cam0_data_received = true; cam1_data_received = true; cam2_data_received = true;
    mask0_received = true; mask1_received = true; mask2_received = true;
  // std::cout << "depth 0 : " << msg->header.stamp.sec; std::cout << "." << msg->header.stamp.nanosec << std::endl;
  }

  void callback_from_image_topic1(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    int depthRow = msg->height;
    int depthCol = msg->width;
    
    cv::Mat depth = cv_bridge::toCvShare(msg, "32FC1")->image;
    memset(inputDepths[1], 0, depthRow*depthCol*sizeof(float));
    memcpy(inputDepths[1], depth.data, depth.total()*sizeof(float));
    cam1_data_received = true; 
  std::cout << "depth 1 : " << msg->header.stamp.sec; std::cout << "." << msg->header.stamp.nanosec << std::endl;
  }

  void callback_from_image_topic2(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    int depthRow = msg->height;
    int depthCol = msg->width;
    
    cv::Mat depth = cv_bridge::toCvShare(msg, "32FC1")->image;
    memset(inputDepths[2], 0, depthRow*depthCol*sizeof(float));
    memcpy(inputDepths[2], depth.data, depth.total()*sizeof(float));
    cam2_data_received = true; 
    std::cout << "depth 2 : " << msg->header.stamp.sec; std::cout << "." << msg->header.stamp.nanosec << std::endl;
  }

  void yosoCallBack(const sensor_msgs::msg::Image::SharedPtr msg){
    mask_width = msg->width;
    mask_height = msg->height;
    
    cv::Mat mask = cv_bridge::toCvShare(msg, "mono8")->image;
    memset(inputMasks[0], 0, mask_height*mask_width*sizeof(uint8_t));
    // memcpy(inputMasks[0], mask.data, mask.total());
    mask0_received = true;
  std::cout << "mask 0 : " << msg->header.stamp.sec; std::cout << "." << msg->header.stamp.nanosec << std::endl;

  }

  void yosoCallBack1(const sensor_msgs::msg::Image::SharedPtr msg){
    mask_width = msg->width;
    mask_height = msg->height;
    
    cv::Mat mask = cv_bridge::toCvShare(msg, "mono8")->image;
    memset(inputMasks[1], 0, mask_height*mask_width*sizeof(uint8_t));
    // memcpy(inputMasks[1], mask.data, mask.total());
    mask1_received = true;
    std::cout << "mask 1 : " << msg->header.stamp.sec; std::cout << "." << msg->header.stamp.nanosec << std::endl;
  }
  
  void yosoCallBack2(const sensor_msgs::msg::Image::SharedPtr msg){
    mask_width = msg->width;
    mask_height = msg->height;
    
    cv::Mat mask = cv_bridge::toCvShare(msg, "mono8")->image;
    memset(inputMasks[2], 0, mask_height*mask_width*sizeof(uint8_t));
    // memcpy(inputMasks[2], mask.data, mask.total());
    mask2_received = true;
    std::cout << "mask 2 : " << msg->header.stamp.sec; std::cout << "." << msg->header.stamp.nanosec << std::endl;
  }
  void callbackT1(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg)
  {
    std::chrono::system_clock::time_point callback_start;
    if(this->_debug_chrono) callback_start = std::chrono::system_clock::now();
    mask_width = yoso0_msg->width;
    mask_height = yoso0_msg->height;

    // get depth image and mask image
    cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
    cv::Mat mask0 = cv_bridge::toCvShare(yoso0_msg, "mono8")->image;
    std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

    // copy depth image and mask image to inputDepths and inputMasks
    for(int i=0; i<1; i++){
      memset(inputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
      memset(inputMasks[i], 0, mask_height*mask_width*sizeof(uint8_t));
    }

    memcpy(inputDepths[0], depth0.data, depth0.total()*sizeof(float));
    cam0_data_received = true; cam1_data_received = true; cam2_data_received = true;
    memcpy(inputMasks[0], mask0.data, mask0.total());
    mask0_received = true; mask1_received = true; mask2_received = true;
    if(this->_debug_chrono){
        std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
        std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
        std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
        this->callback_duration = copy_end - callback_start;
    }
  }
  void callbackT2(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso1_msg)
  {
    std::chrono::system_clock::time_point callback_start;
    if(this->_debug_chrono) callback_start = std::chrono::system_clock::now();
    mask_width = yoso0_msg->width;
    mask_height = yoso0_msg->height;

    // get depth image and mask image
    cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
    cv::Mat depth1 = cv_bridge::toCvShare(depth1_msg, "32FC1")->image;
    cv::Mat mask0 = cv_bridge::toCvShare(yoso0_msg, "mono8")->image;
    cv::Mat mask1 = cv_bridge::toCvShare(yoso1_msg, "mono8")->image;
    std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

    // copy depth image and mask image to inputDepths and inputMasks
    for(int i=0; i<2; i++){
      memset(inputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
      memset(inputMasks[i], 0, mask_height*mask_width*sizeof(uint8_t));
    }

    memcpy(inputDepths[0], depth0.data, depth0.total()*sizeof(float));
    memcpy(inputDepths[1], depth1.data, depth1.total()*sizeof(float));
    cam0_data_received = true; cam1_data_received = true; cam2_data_received = true;
    memcpy(inputMasks[0], mask0.data, mask0.total());
    memcpy(inputMasks[1], mask1.data, mask1.total());
    mask0_received = true; mask1_received = true; mask2_received = true;
    if(this->_debug_chrono){
      std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
      std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
      std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
      this->callback_duration = copy_end - callback_start;
    }
  }
  void callbackT3(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& depth2_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& yoso1_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso2_msg)
  {
    std::chrono::system_clock::time_point callback_start;
    if(this->_debug_chrono) callback_start = std::chrono::system_clock::now();
    mask_width = yoso0_msg->width;
    mask_height = yoso0_msg->height;

    // get depth image and mask image
    cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
    cv::Mat depth1 = cv_bridge::toCvShare(depth1_msg, "32FC1")->image;
    cv::Mat depth2 = cv_bridge::toCvShare(depth2_msg, "32FC1")->image;
    cv::Mat mask0 = cv_bridge::toCvShare(yoso0_msg, "mono8")->image;
    cv::Mat mask1 = cv_bridge::toCvShare(yoso1_msg, "mono8")->image;
    cv::Mat mask2 = cv_bridge::toCvShare(yoso2_msg, "mono8")->image;
    std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

    // copy depth image and mask image to inputDepths and inputMasks
    for(int i=0; i<3; i++){
      memset(inputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
      memset(inputMasks[i], 0, mask_height*mask_width*sizeof(uint8_t));
    }

    memcpy(inputDepths[0], depth0.data, depth0.total()*sizeof(float));
    memcpy(inputDepths[1], depth1.data, depth1.total()*sizeof(float));
    memcpy(inputDepths[2], depth2.data, depth2.total()*sizeof(float));
    cam0_data_received = true; cam1_data_received = true; cam2_data_received = true;
    memcpy(inputMasks[0], mask0.data, mask0.total());
    memcpy(inputMasks[1], mask1.data, mask1.total());
    memcpy(inputMasks[2], mask2.data, mask2.total());
    mask0_received = true; mask1_received = true; mask2_received = true;

    if(this->_debug_chrono){
      std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
      std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
      std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
      this->callback_duration = copy_end - callback_start;
    }
    // std::cout << "depth 0 : " << depth0_msg->header.stamp.sec; std::cout << "." << depth0_msg->header.stamp.nanosec << std::endl;
    // std::cout << "depth 1 : " << depth1_msg->header.stamp.sec; std::cout << "." << depth1_msg->header.stamp.nanosec << std::endl;
    // std::cout << "depth 2 : " << depth2_msg->header.stamp.sec; std::cout << "." << depth2_msg->header.stamp.nanosec << std::endl;
    // std::cout << "mask 0 : " << yoso0_msg->header.stamp.sec; std::cout << "." << yoso0_msg->header.stamp.nanosec << std::endl;
    // std::cout << "mask 1 : " << yoso1_msg->header.stamp.sec; std::cout << "." << yoso1_msg->header.stamp.nanosec << std::endl;
    // std::cout << "mask 2 : " << yoso2_msg->header.stamp.sec; std::cout << "." << yoso2_msg->header.stamp.nanosec << std::endl;
  }
  void callbackD2(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg)
  {
    std::chrono::system_clock::time_point callback_start;
    if(this->_debug_chrono) callback_start = std::chrono::system_clock::now();

    // get depth image and mask image
    cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
    cv::Mat depth1 = cv_bridge::toCvShare(depth1_msg, "32FC1")->image;
    std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

    // copy depth image and mask image to inputDepths and inputMasks
    for(int i=0; i<2; i++){
      memset(inputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
    }

    memcpy(inputDepths[0], depth0.data, depth0.total()*sizeof(float));
    memcpy(inputDepths[1], depth1.data, depth1.total()*sizeof(float));
    cam0_data_received = true; cam1_data_received = true; cam2_data_received = true;
    mask0_received = true; mask1_received = true; mask2_received = true;
    if(this->_debug_chrono){
      std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
      std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
      std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
      this->callback_duration = copy_end - callback_start;
    }
    // std::cout << "depth 0 : " << depth0_msg->header.stamp.sec; std::cout << "." << depth0_msg->header.stamp.nanosec << std::endl;
    // std::cout << "depth 1 : " << depth1_msg->header.stamp.sec; std::cout << "." << depth1_msg->header.stamp.nanosec << std::endl;
    // std::cout << "depth 2 : " << depth2_msg->header.stamp.sec; std::cout << "." << depth2_msg->header.stamp.nanosec << std::endl;
    // std::cout << "++ In Callback ====================================" << std::endl;
    // std::cout << "cvbridge time       : " << cvbridge_time.count()*1000 << "ms" << std::endl;
    // std::cout << "copy time          : " << copy_time.count()*1000 << "ms" << std::endl;
    // std::cout << "total time          : " << this->callback_duration.count()*1000 << "ms" << std::endl;
    // std::cout << "++ In Callback ====================================" << std::endl;
  }
  void callbackD3(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr& depth2_msg)
  {
    std::chrono::system_clock::time_point callback_start;
    if(this->_debug_chrono) callback_start = std::chrono::system_clock::now();

    // get depth image and mask image
    cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
    cv::Mat depth1 = cv_bridge::toCvShare(depth1_msg, "32FC1")->image;
    cv::Mat depth2 = cv_bridge::toCvShare(depth2_msg, "32FC1")->image;
    std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

    // copy depth image and mask image to inputDepths and inputMasks
    for(int i=0; i<3; i++){
      memset(inputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
    }

    memcpy(inputDepths[0], depth0.data, depth0.total()*sizeof(float));
    memcpy(inputDepths[1], depth1.data, depth1.total()*sizeof(float));
    memcpy(inputDepths[2], depth2.data, depth2.total()*sizeof(float));
    cam0_data_received = true; cam1_data_received = true; cam2_data_received = true;
    mask0_received = true; mask1_received = true; mask2_received = true;
    if(this->_debug_chrono){
      std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
      std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
      std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
      this->callback_duration = copy_end - callback_start;
    }
    // std::cout << "depth 0 : " << depth0_msg->header.stamp.sec; std::cout << "." << depth0_msg->header.stamp.nanosec << std::endl;
    // std::cout << "depth 1 : " << depth1_msg->header.stamp.sec; std::cout << "." << depth1_msg->header.stamp.nanosec << std::endl;
    // std::cout << "depth 2 : " << depth2_msg->header.stamp.sec; std::cout << "." << depth2_msg->header.stamp.nanosec << std::endl;
    // std::cout << "++ In Callback ====================================" << std::endl;
    // std::cout << "cvbridge time       : " << cvbridge_time.count()*1000 << "ms" << std::endl;
    // std::cout << "copy time          : " << copy_time.count()*1000 << "ms" << std::endl;
    // std::cout << "total time          : " << this->callback_duration.count()*1000 << "ms" << std::endl;
    // std::cout << "++ In Callback ====================================" << std::endl;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depthsub_0;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depthsub_1;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depthsub_2;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr yososub_0;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr yososub_1;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr yososub_2;
  message_filters::Subscriber<sensor_msgs::msg::Image> mf_depthsub_0, mf_depthsub_1, mf_depthsub_2;
  message_filters::Subscriber<sensor_msgs::msg::Image> mf_yososub_0, mf_yososub_1, mf_yososub_2;
  using _SyncPolicyT1 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using _SyncPolicyT2 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using _SyncPolicyT3 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using _SyncPolicyD2 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using _SyncPolicyD3 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using SyncT1 = message_filters::Synchronizer<_SyncPolicyT1>;
  using SyncT2 = message_filters::Synchronizer<_SyncPolicyT2>;
  using SyncT3 = message_filters::Synchronizer<_SyncPolicyT3>;
  using SyncD2 = message_filters::Synchronizer<_SyncPolicyD2>;
  using SyncD3 = message_filters::Synchronizer<_SyncPolicyD3>;
  std::shared_ptr<message_filters::Synchronizer<_SyncPolicyT1>> syncT1;
  std::shared_ptr<message_filters::Synchronizer<_SyncPolicyT2>> syncT2;
  std::shared_ptr<message_filters::Synchronizer<_SyncPolicyT3>> syncT3;
  std::shared_ptr<message_filters::Synchronizer<_SyncPolicyD2>> syncD2;
  std::shared_ptr<message_filters::Synchronizer<_SyncPolicyD3>> syncD3;
};



int main(int argc, char* argv[])
{
  // initialize ROS
  printf("Program Start.\n");
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<kfusionNode>();
  executor.add_node(node);
  const int numofcamera=node->_num_cameras;
  const int map_size_x=node->_map_size_x;
  const int map_size_y=node->_map_size_y;
  const int map_size_z=node->_map_size_z;
  const bool debug_chrono=node->_debug_chrono;
  float voxel_side_length;
  int usleep_time=node->_usleep_time;

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
		// inputDepths[i] = (float*)malloc(sizeof(float) * DepthWidth[AzureMode] * DepthHeight[AzureMode]);
    // malloc iinputDepths using new
    inputDepths[i] = new float[_depth_width * _depth_height];
    if(node->_isMask) inputMasks[i] = new uint8_t[mask_width * mask_height];

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

  DeviceKernel dk(node->_isMask,_depth_width, _depth_height, numofcamera, ks, extrinsicInv, gpuvox_ExtrArr);
  

  gvl->addMap(MT_BITVECTOR_VOXELMAP, "voxelmap_1");
  boost::shared_ptr<voxelmap::BitVectorVoxelMap> ptrbitVoxmap(gvl->getMap("voxelmap_1")->as<voxelmap::BitVectorVoxelMap>());
  std::cout << "[DEBUG] ptrbitVoxmap's m_dev_data (address) : " << ptrbitVoxmap->getVoidDeviceDataPtr() << std::endl;
  
  // update kernerl parameter and set constant memory
  ptrbitVoxmap->updateReconThresh(node->_recon_threshold);
  ptrbitVoxmap->updateVisUnknown(node->_vis_unknown);
  node->_ptrBitVoxMap = ptrbitVoxmap;
  ptrbitVoxmap->setConstMemory(_depth_width, _depth_height,mask_width,static_cast<float>(mask_width) / _depth_width);

  // visualize first iteration
  cam0_data_received = true; cam1_data_received = true; cam2_data_received = true;
  mask0_received = true; mask1_received = true; mask2_received = true;
  
  size_t iteration = 0;
  // double toDeviceMem, genPointCloud, reconTime, visualTime, transformTime, copyTime, generatemsgTime = 0.0;
  double copyTime, genPointCloud, reconTime,callbackTime,visualTime,syncTime = 0.0;

  LOGGING_INFO(Gpu_voxels, "start visualizing maps" << endl);
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
    if(cam0_data_received && cam1_data_received && cam2_data_received && mask0_received && mask1_received && mask2_received)
    // if(cam1_data_received && cam2_data_received )
    {
      is_loop = true;
      std::cout << "[DEBUG] Iteration =========> " << iteration << std::endl;
      if(debug_chrono){
        std::chrono::system_clock::time_point callback_end = std::chrono::system_clock::now();
        std::chrono::duration<double> callback_time = node->callback_duration;
        std::chrono::duration<double> synchronize_time = callback_end - loop_start;
        if(node->_isSync)std::cout << "[DEBUG] : (chrono::system_clock) Synchronization Time         : " << synchronize_time.count()*1000 - callback_time.count()*1000 << "ms" << std::endl;
        else std::cout << "[DEBUG] : (chrono::system_clock) Callback Processing Time     : " << synchronize_time.count()*1000 - callback_time.count()*1000 << "ms" << std::endl;
        timeMeasureTable[0][iteration] = synchronize_time.count()*1000 - callback_time.count()*1000;
        syncTime += synchronize_time.count()*1000 - callback_time.count()*1000;
        timeMeasureTable[1][iteration] = callback_time.count()*1000;
        callbackTime += callback_time.count() * 1000;
        if(node->_isSync) std::cout << "[DEBUG] : (chrono::system_clock) Callback Processing Time     : " << callback_time.count()*1000 << "ms" << std::endl;
      }
      cam0_data_received = false; cam1_data_received = false; cam2_data_received = false;
      mask0_received = false; mask1_received = false; mask2_received = false;
    
      
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
      else {usleep(usleep_time); gvl->visualizeMap("voxelmap_1");}
      
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