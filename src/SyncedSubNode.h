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

/**
 * @brief This Node Subscribe depth image and mask image from 3 cameras synchronously.
 * 
 */
class SyncedSubNode : public rclcpp::Node
{
  public:
    // Choice whether to visualize the result of each camera. 
    bool _master_show;
    bool _sub1_show;
    bool  _sub2_show;

    // Launch option.
    bool _isMask;
    bool _isClearMap;
    bool _vis_unknown;
    bool _debug_chrono;
    int _usleep_time;
    int _num_cameras;
    int _map_size_x;
    int _map_size_y;
    int _map_size_z;
    float _voxel_side_length;
    float _recon_threshold;

    // Some Variables
    bool _data_received {false};
    float* _InputDepths[3];
    uint8_t* _InputMasks[3];
    uint16_t _depth_width;
    uint16_t _depth_height;
    uint16_t _mask_width;
    uint16_t _mask_height;

    rclcpp::CallbackGroup::SharedPtr _parallel_group;
    boost::shared_ptr<voxelmap::BitVectorVoxelMap> _ptrBitVoxMap;
    std::chrono::duration<double> callback_duration;

    /**
     * @brief Parameter reconfigure callback function
     * 
     * @param parameters 
     * @return rcl_interfaces::msg::SetParametersResult 
     */
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters);
    
    SyncedSubNode(float* input_detph[3], uint8_t* input_mask[3], const int depth_width, const int depth_height, const int mask_width, const int mask_height);
    ~SyncedSubNode();

    void callback_from_image_topic(const sensor_msgs::msg::Image::SharedPtr msg);
    /**
     * @brief Synchronized Callback function for just one set of depth and mask image. 
     * 
     * @param depth0_msg 
     * @param yoso0_msg 
     */
    void callbackT1(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg);
    
    /**
     * @brief Synchronized Callback function for two sets of depth and mask image. 
     * 
     * @param depth0_msg 
     * @param depth1_msg 
     * @param yoso0_msg 
     * @param yoso1_msg 
     */
    void callbackT2(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso1_msg);
    /**
     * @brief Synchronized Callback function for three sets of depth and mask image.
     * 
     * @param depth0_msg 
     * @param depth1_msg 
     * @param depth2_msg 
     * @param yoso0_msg 
     * @param yoso1_msg 
     * @param yoso2_msg 
     */
    void callbackT3(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr& depth2_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso0_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr& yoso1_msg, const sensor_msgs::msg::Image::ConstSharedPtr& yoso2_msg);
    /**
     * @brief Synchronized Callback function for two depth image.
     * 
     * @param depth0_msg 
     * @param depth1_msg 
     */
    void callbackD2(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg);
    /**
     * @brief Synchronized Callback function for three depth image.
     * 
     * @param depth0_msg 
     * @param depth1_msg 
     * @param depth2_msg 
     */
    void callbackD3(const sensor_msgs::msg::Image::ConstSharedPtr& depth0_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth1_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr& depth2_msg);

  private:
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
