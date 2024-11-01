#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>

#include <Eigen/Core>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>

#include "r3live.hpp"
#include "IMU_Processing.hpp"
#include "so3_math.h"
#include "common_lib.h"
#include "kd_tree/ikd_Tree.h"
#include "FOV_Checker/FOV_Checker.h"

#include "tools_logger.hpp"
#include "tools_color_printf.hpp"
#include "tools_eigen.hpp"
#include "tools_data_io.hpp"
#include "tools_timer.hpp"
#include "tools_openCV_3_to_4.hpp"

Camera_Lidar_queue g_camera_lidar_queue;
MeasureGroup Measures;
StatesGroup g_lio_state;
std::string data_dump_dir = std::string("/mnt/0B3B134F0B3B134F/color_temp_r3live/");

int main(int argc, char **argv)
{
    printf_program("R3LIVE: A Robust, Real-time, RGB-colored, LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package");
    Common_tools::printf_software_version();
    Eigen::initParallel();
    ros::init(argc, argv, "R3LIVE_main");
    R3LIVE *fast_lio_instance = new R3LIVE();
    ros::Rate rate(5000); // 无用
    bool status = ros::ok(); // 无用
    ros::spin();
}
