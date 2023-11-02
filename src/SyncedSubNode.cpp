#include "SyncedSubNode.h"


SyncedSubNode::SyncedSubNode(float* input_detph[3], uint8_t* input_mask[3], const int depth_width, const int depth_height, const int mask_width, const int mask_height)
: Node("kfusion_node"), _depth_width(depth_width), _depth_height(depth_height), _mask_width(mask_width), _mask_height(mask_height)
{
  // _InputDepths = input_detph;
  // _InputMasks = input_mask;
  for(int i{0};i<3;i++)
  {
    _InputDepths[i] = input_detph[i];
    std::cout << "In Node : " << &_InputDepths[i] << std::endl;
    _InputMasks[i] = input_mask[i];
  }
  // Set parameters
  std::string _master_cam_topic, _sub1_cam_topic, _sub2_cam_topic;
  std::string _master_yoso_topic, _sub1_yoso_topic, _sub2_yoso_topic;
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
  _isMask = this->get_parameter("isMask").as_bool();
  _isClearMap = this->get_parameter("isClearMap").as_bool();
  _recon_threshold = this->get_parameter("recon_threshold").as_double();
  _vis_unknown = this->get_parameter("vis_unknown").as_bool();
  _debug_chrono = this->get_parameter("debug_chrono").as_bool();


  // Set Synchronized Subcriber
  if(_isMask){
    switch (_num_cameras)
    {
      case (1):
      {
        mf_depthsub_0.subscribe(this, _master_cam_topic);
        mf_yososub_0.subscribe(this, _master_yoso_topic);
        syncT1.reset(new SyncT1(_SyncPolicyT1(60), mf_depthsub_0, mf_yososub_0));
        syncT1->registerCallback(std::bind(&SyncedSubNode::callbackT1, this, std::placeholders::_1, std::placeholders::_2));
        break;
      }
      case (2):
      {
        mf_depthsub_0.subscribe(this, _master_cam_topic);
        mf_depthsub_1.subscribe(this, _sub1_cam_topic);
        mf_yososub_0.subscribe(this, _master_yoso_topic);
        mf_yososub_1.subscribe(this, _sub1_yoso_topic);
        syncT2.reset(new SyncT2(_SyncPolicyT2(60), mf_depthsub_0, mf_depthsub_1, mf_yososub_0, mf_yososub_1));
        syncT2->registerCallback(std::bind(&SyncedSubNode::callbackT2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        break;
      }
      case (3):
      {
        mf_depthsub_0.subscribe(this, _master_cam_topic);
        mf_depthsub_1.subscribe(this, _sub1_cam_topic);
        mf_depthsub_2.subscribe(this, _sub2_cam_topic);
        mf_yososub_0.subscribe(this, _master_yoso_topic);
        mf_yososub_1.subscribe(this, _sub1_yoso_topic);
        mf_yososub_2.subscribe(this, _sub2_yoso_topic);
        syncT3.reset(new SyncT3(_SyncPolicyT3(60), mf_depthsub_0, mf_depthsub_1, mf_depthsub_2, mf_yososub_0, mf_yososub_1, mf_yososub_2));
        syncT3->registerCallback(std::bind(&SyncedSubNode::callbackT3, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
        break;
      }
      default:
      {
        RCLCPP_ERROR(this->get_logger(), "num_cameras must be 1, 2, or 3");
        exit(EXIT_FAILURE);
      }
    }
  }
  else {
    switch (_num_cameras)
    {
      case (1):
      {
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depthsub_0;
        depthsub_0 = this->create_subscription<sensor_msgs::msg::Image>(_master_cam_topic, 20, std::bind(&SyncedSubNode::callback_from_image_topic, this, std::placeholders::_1));
        break;
      }
      case (2):
      {
        mf_depthsub_0.subscribe(this, _master_cam_topic);
        mf_depthsub_1.subscribe(this, _sub1_cam_topic);
        syncD2.reset(new SyncD2(_SyncPolicyD2(30),mf_depthsub_0, mf_depthsub_1));
        syncD2->registerCallback(std::bind(&SyncedSubNode::callbackD2, this, std::placeholders::_1, std::placeholders::_2));
        break;
      }
      case (3):
      {
        mf_depthsub_0.subscribe(this, _master_cam_topic);
        mf_depthsub_1.subscribe(this, _sub1_cam_topic);
        mf_depthsub_2.subscribe(this, _sub2_cam_topic);
        syncD3.reset(new SyncD3(_SyncPolicyD3(30),mf_depthsub_0, mf_depthsub_1, mf_depthsub_2));
        syncD3->registerCallback(std::bind(&SyncedSubNode::callbackD3, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        break;
      }
      default:
      {
        RCLCPP_ERROR(this->get_logger(), "num_cameras must be 1, 2, or 3");
        exit(EXIT_FAILURE);
      }
    }
  }
  
  // Print Settings
  printf("============================== Node Info ===============================\n");
  printf("Num of Cam: %d\n", _num_cameras);
  printf("Depth topic : %s, %s, %s\n", _master_cam_topic.c_str(), _sub1_cam_topic.c_str(), _sub2_cam_topic.c_str());
  if(_isMask) printf("Yoso topic : %s, %s, %s\n", _master_yoso_topic.c_str(), _sub1_yoso_topic.c_str(), _sub2_yoso_topic.c_str());
  printf("Usleep time: %dms\n", static_cast<int>(_usleep_time/1000));
  printf("Voxel Map Size : %dX %d X %d\n", _map_size_x, _map_size_y, _map_size_z);
  printf("Use Mask : %s\n", _isMask ? "True" : "False");
  printf("Map Update Method : %s\n", _isClearMap ? "Clear Map" : "Kernel Function Procesing");
  printf("Projection Distance Error threshold : %.5f\n", _recon_threshold);
  printf("Visualize Unknown Voxels : %s\n", _vis_unknown ? "True" : "False");
  printf("Debug Chrono : %s\n", _debug_chrono ? "True" : "False");
  printf("========================================================================\n");

  // Set Parameter Reconfigure Callback 
  this->set_on_parameters_set_callback(
          std::bind(&SyncedSubNode::parametersCallback, this, std::placeholders::_1));
}

SyncedSubNode::~SyncedSubNode()
{
  RCLCPP_INFO(this->get_logger(), "Node is shutting down...");
}

rcl_interfaces::msg::SetParametersResult SyncedSubNode::parametersCallback(const std::vector<rclcpp::Parameter> &parameters)
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

void SyncedSubNode::callback_from_image_topic(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    int depthRow = msg->height;
    int depthCol = msg->width;
    
    cv::Mat depth = cv_bridge::toCvShare(msg, "32FC1")->image;
    memset(_InputDepths[0], 0, depthRow*depthCol*sizeof(float));
    memcpy(_InputDepths[0], depth.data, depth.total()*sizeof(float));
    _data_received = true;
  }

void SyncedSubNode::callbackT1(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg)
{
  std::chrono::system_clock::time_point callback_start;
  if(_debug_chrono) callback_start = std::chrono::system_clock::now();
  _mask_width = yoso0_msg->width;
  _mask_height = yoso0_msg->height;

  // get depth image and mask image
  cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
  cv::Mat mask0 = cv_bridge::toCvShare(yoso0_msg, "mono8")->image;
  std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

  // copy depth image and mask image to _InputDepths and _InputMasks
  for(int i=0; i<1; i++){
    memset(_InputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
    memset(_InputMasks[i], 0, _mask_height*_mask_width*sizeof(uint8_t));
  }

  memcpy(_InputDepths[0], depth0.data, depth0.total()*sizeof(float));
  memcpy(_InputMasks[0], mask0.data, mask0.total());
  if(_debug_chrono){
      std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
      std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
      std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
      callback_duration = copy_end - callback_start;
  }
  _data_received = true;
}

void SyncedSubNode::callbackT2(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
              const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso1_msg)
{
  std::chrono::system_clock::time_point callback_start;
  if(_debug_chrono) callback_start = std::chrono::system_clock::now();
  _mask_width = yoso0_msg->width;
  _mask_height = yoso0_msg->height;

  // get depth image and mask image
  cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
  cv::Mat depth1 = cv_bridge::toCvShare(depth1_msg, "32FC1")->image;
  cv::Mat mask0 = cv_bridge::toCvShare(yoso0_msg, "mono8")->image;
  cv::Mat mask1 = cv_bridge::toCvShare(yoso1_msg, "mono8")->image;
  std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

  // copy depth image and mask image to _InputDepths and _InputMasks
  for(int i=0; i<2; i++){
    memset(_InputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
    memset(_InputMasks[i], 0, _mask_height*_mask_width*sizeof(uint8_t));
  }

  memcpy(_InputDepths[0], depth0.data, depth0.total()*sizeof(float));
  memcpy(_InputDepths[1], depth1.data, depth1.total()*sizeof(float));
  memcpy(_InputMasks[0], mask0.data, mask0.total());
  memcpy(_InputMasks[1], mask1.data, mask1.total());
  if(_debug_chrono){
    std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
    std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
    std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
    callback_duration = copy_end - callback_start;
  }
  _data_received = true;
}

void SyncedSubNode::callbackT3(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
              const sensor_msgs::msg::Image::ConstSharedPtr& depth2_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg,
              const sensor_msgs::msg::Image::ConstSharedPtr& yoso1_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso2_msg)
{
  std::chrono::system_clock::time_point callback_start;
  if(_debug_chrono) callback_start = std::chrono::system_clock::now();
  _mask_width = yoso0_msg->width;
  _mask_height = yoso0_msg->height;

  // get depth image and mask image
  cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
  cv::Mat depth1 = cv_bridge::toCvShare(depth1_msg, "32FC1")->image;
  cv::Mat depth2 = cv_bridge::toCvShare(depth2_msg, "32FC1")->image;
  cv::Mat mask0 = cv_bridge::toCvShare(yoso0_msg, "mono8")->image;
  cv::Mat mask1 = cv_bridge::toCvShare(yoso1_msg, "mono8")->image;
  cv::Mat mask2 = cv_bridge::toCvShare(yoso2_msg, "mono8")->image;
  std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

  // copy depth image and mask image to _InputDepths and _InputMasks
  for(int i=0; i<3; i++){
    memset(_InputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
    memset(_InputMasks[i], 0, _mask_height*_mask_width*sizeof(uint8_t));
  }

  memcpy(_InputDepths[0], depth0.data, depth0.total()*sizeof(float));
  memcpy(_InputDepths[1], depth1.data, depth1.total()*sizeof(float));
  memcpy(_InputDepths[2], depth2.data, depth2.total()*sizeof(float));
  memcpy(_InputMasks[0], mask0.data, mask0.total());
  memcpy(_InputMasks[1], mask1.data, mask1.total());
  memcpy(_InputMasks[2], mask2.data, mask2.total());

  if(_debug_chrono){
    std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
    std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
    std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
    callback_duration = copy_end - callback_start;
  }
  _data_received = true;
}

void SyncedSubNode::callbackD2(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg)
{
  std::chrono::system_clock::time_point callback_start;
  if(_debug_chrono) callback_start = std::chrono::system_clock::now();

  // get depth image and mask image
  cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
  cv::Mat depth1 = cv_bridge::toCvShare(depth1_msg, "32FC1")->image;
  std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

  // copy depth image and mask image to _InputDepths and _InputMasks
  for(int i=0; i<2; i++){
    memset(_InputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
  }

  memcpy(_InputDepths[0], depth0.data, depth0.total()*sizeof(float));
  memcpy(_InputDepths[1], depth1.data, depth1.total()*sizeof(float));
  if(_debug_chrono){
    std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
    std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
    std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
    callback_duration = copy_end - callback_start;
  }
  _data_received = true;
}

void SyncedSubNode::callbackD3(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
              const sensor_msgs::msg::Image::ConstSharedPtr& depth2_msg)
{
  std::chrono::system_clock::time_point callback_start;
  if(_debug_chrono) callback_start = std::chrono::system_clock::now();

  // get depth image and mask image
  cv::Mat depth0 = cv_bridge::toCvShare(depth0_msg, "32FC1")->image;
  cv::Mat depth1 = cv_bridge::toCvShare(depth1_msg, "32FC1")->image;
  cv::Mat depth2 = cv_bridge::toCvShare(depth2_msg, "32FC1")->image;
  std::chrono::system_clock::time_point cvbridge_end = std::chrono::system_clock::now();

  // copy depth image and mask image to _InputDepths and _InputMasks
  for(int i=0; i<3; i++){
    memset(_InputDepths[i], 0, _depth_width*_depth_height*sizeof(float));
  }

  memcpy(_InputDepths[0], depth0.data, depth0.total()*sizeof(float));
  memcpy(_InputDepths[1], depth1.data, depth1.total()*sizeof(float));
  memcpy(_InputDepths[2], depth2.data, depth2.total()*sizeof(float));
  if(_debug_chrono){
    std::chrono::system_clock::time_point copy_end = std::chrono::system_clock::now();
    std::chrono::duration<double> cvbridge_time = cvbridge_end - callback_start;
    std::chrono::duration<double> copy_time = copy_end - cvbridge_end;
    callback_duration = copy_end - callback_start;
  }
  _data_received = true;
}

