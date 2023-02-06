#pragma once
#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <condition_variable>

#include <Eigen/Eigen>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>

#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "common_lib.h"
#include "so3_math.h"

#define MAX_INI_COUNT (20)

const inline bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };
bool check_state(StatesGroup &state_inout);
void check_in_out_state(const StatesGroup &state_in, StatesGroup &state_inout);

class ImuProcess
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();
    void Process(const MeasureGroup &meas, StatesGroup &state, PointCloudXYZINormal::Ptr pcl_un_);
    void Reset();
    void IMU_Initial(const MeasureGroup &meas, StatesGroup &state, int &N);
    void lic_state_propagate(const MeasureGroup &meas, StatesGroup &state_inout);
    void lic_point_cloud_undistort(const MeasureGroup &meas, const StatesGroup &state_inout, PointCloudXYZINormal &pcl_out);
    StatesGroup imu_preintegration(const StatesGroup &state_inout, std::deque<sensor_msgs::Imu::ConstPtr> &v_imu, double end_pose_dt = 0);

    void IntegrateGyr(const std::vector<sensor_msgs::Imu::ConstPtr> &v_imu); // 无用
    void UndistortPcl(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZINormal &pcl_in_out); // 无用
    void Integrate(const sensor_msgs::ImuConstPtr &imu); // 无用
    void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu); // 无用

public:
    Eigen::Vector3d angvel_last;
    Eigen::Vector3d acc_s_last;
    Eigen::Matrix<double, DIM_OF_PROC_N, 1> cov_proc_noise; // 过程噪声的协方差矩阵
    Eigen::Vector3d cov_acc;
    Eigen::Vector3d cov_gyr;
    bool b_first_frame_ = true;                             // 判断是否是第1帧, 第1帧需要初始化
    bool imu_need_init_ = true;                             // 判断是否是第1帧, 第1帧需要初始化
    int init_iter_num = 1;
    Eigen::Vector3d mean_acc;
    Eigen::Vector3d mean_gyr;
    PointCloudXYZINormal::Ptr cur_pcl_un_;                  // 去畸变之后的点云
    sensor_msgs::ImuConstPtr last_imu_;                     // For timestamp usage
    double last_lidar_end_time_;
    double start_timestamp_;                                // 用于陀螺仪积分
    std::deque<sensor_msgs::ImuConstPtr> v_imu_;            // 确保v_imu_与v_rot_大小相等
    std::vector<Pose6D> IMU_pose;

    // std::ofstream fout; // 好像无用
    ros::NodeHandle nh; // 无用
    std::vector<Eigen::Matrix3d> v_rot_pcl_; // 无用
};
