#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <queue>
#include <deque>

#include <Eigen/Eigen>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "lib_sophus/se3.hpp"
#include "lib_sophus/so3.hpp"
#include "so3_math.h"
#include "tools_color_printf.hpp"
#include "tools_eigen.hpp"
#include "tools_ros.hpp"

#define USE_ikdtree
#define ESTIMATE_GRAVITY 1
#define ENABLE_CAMERA_OBS 1
#define printf_line std::cout << __FILE__ << " " << __LINE__ << std::endl;
#define PI_M (3.14159265358)
#define G_m_s2 (9.81) // Gravity const in Hong Kong SAR, China
#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)
#define INIT_COV (0.0001)
#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define CONSTRAIN(v, min, max) ((v > min) ? ((v < max) ? v : max) : min)
#define ARRAY_FROM_EIGEN(mat) mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat) std::vector<decltype(mat)::Scalar>(mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "Log/" + name))

// 18维: R p v bg ba g
// 24维: R p v bg ba g R_ic p_ic
// 29维: R p v bg ba g R_ic p_ic t fx fy cx cy
#if ENABLE_CAMERA_OBS
#define DIM_OF_STATES (29) // with vio obs
#else
#define DIM_OF_STATES (18) // For faster speed.
#endif
#define DIM_OF_PROC_N (12) // Dimension of process noise (Let Dim(SO(3)) = 3)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZINormal;

static const Eigen::Matrix3d Eye3d(Eigen::Matrix3d::Identity());
static const Eigen::Matrix3f Eye3f(Eigen::Matrix3f::Identity());
static const Eigen::Vector3d Zero3d(0, 0, 0);
static const Eigen::Vector3f Zero3f(0, 0, 0);
// Eigen::Vector3d Lidar_offset_to_IMU(0.05512, 0.02226, 0.0297); // Horizon // TODO 外参直接写到了代码里?
static const Eigen::Vector3d Lidar_offset_to_IMU(0.04165, 0.02326, -0.0284); // Avia

// 6自由度位姿, 偏移时间, 3轴加速度, 3轴角速度, 3维重力加速度封装在结构体Pose6D中
struct Pose6D
{
    typedef double data_type;
    data_type offset_time;
    data_type rot[9];
    data_type acc[3];
    data_type vel[3];
    data_type pos[3];
    data_type gyr[3];
};

// 计算向量的反对称矩阵
template <typename T = double>
inline Eigen::Matrix<T, 3, 3> vec_to_hat(Eigen::Matrix<T, 3, 1> &omega)
{
    Eigen::Matrix<T, 3, 3> res_mat_33;
    res_mat_33.setZero();
    res_mat_33(0, 1) = -omega(2);
    res_mat_33(1, 0) = omega(2);
    res_mat_33(0, 2) = omega(1);
    res_mat_33(2, 0) = -omega(1);
    res_mat_33(1, 2) = -omega(0);
    res_mat_33(2, 1) = omega(0);
    return res_mat_33;
}

// 计算角度的反正切
template <typename T = double>
T cot(const T theta)
{
    return 1.0 / std::tan(theta);
}

// 旋转矩阵的右雅克比
template <typename T = double>
inline Eigen::Matrix<T, 3, 3> right_jacobian_of_rotion_matrix(const Eigen::Matrix<T, 3, 1> &omega)
{
    // Barfoot, Timothy D, State estimation for robotics. Page 232-237
    // 机器人学中的状态估计, 203页, 公式7.77a
    Eigen::Matrix<T, 3, 3> res_mat_33;

    T theta = omega.norm();
    if (std::isnan(theta) || theta == 0)
        return Eigen::Matrix<T, 3, 3>::Identity();
    Eigen::Matrix<T, 3, 1> a = omega / theta;
    Eigen::Matrix<T, 3, 3> hat_a = vec_to_hat(a);
    res_mat_33 = sin(theta) / theta * Eigen::Matrix<T, 3, 3>::Identity() + (1 - (sin(theta) / theta)) * a * a.transpose() + ((1 - cos(theta)) / theta) * hat_a;
    // cout << "Omega: " << omega.transpose() << endl;
    // cout << "Res_mat_33:\r\n"  <<res_mat_33 << endl;
    return res_mat_33;
}

// 旋转矩阵的右雅克比逆
template <typename T = double>
Eigen::Matrix<T, 3, 3> inverse_right_jacobian_of_rotion_matrix(const Eigen::Matrix<T, 3, 1> &omega)
{
    // Barfoot, Timothy D, State estimation for robotics. Page 232-237
    // 机器人学中的状态估计, 203页, 公式7.76a
    Eigen::Matrix<T, 3, 3> res_mat_33;

    T theta = omega.norm();
    if (std::isnan(theta) || theta == 0)
        return Eigen::Matrix<T, 3, 3>::Identity();
    Eigen::Matrix<T, 3, 1> a = omega / theta;
    Eigen::Matrix<T, 3, 3> hat_a = vec_to_hat(a);
    res_mat_33 = (theta / 2) * (cot(theta / 2)) * Eigen::Matrix<T, 3, 3>::Identity() + (1 - (theta / 2) * (cot(theta / 2))) * a * a.transpose() + (theta / 2) * hat_a;
    // cout << "Omega: " << omega.transpose() << endl;
    // cout << "Res_mat_33:\r\n"  <<res_mat_33 << endl;
    return res_mat_33;
}

// 保存LiDAR数据队列
// 保存LiDAR, IMU, Camera的时间戳, 用于控制各传感器数据队列的对齐
struct Camera_Lidar_queue
{
    double m_first_imu_time = -3e88;   // 第1帧IMU数据时间戳, 第1帧IMU数据进来后赋值, 之后保持不变
    double m_last_imu_time = -3e88;    // 距当前最近的IMU帧时间戳, 最后的/最新的IMU帧时间戳
    double m_last_visual_time = -3e88; // 距当前最近的Camera帧时间戳, 最后的/最新的Camera帧时间戳
    double m_last_lidar_time = -3e88;  // 距当前最近的LiDAR帧时间戳, 最后的/最新的LiDAR帧时间戳
    int m_if_have_lidar_data = 0;      // 标志位, 表示当前是否有还未处理的LiDAR数据
    int m_if_have_camera_data = 0;     // 标志位, 表示当前是否有还未处理的Camera数据
    Eigen::Vector3d g_noise_cov_acc;   // 加速度计数据噪声的协方差矩阵
    Eigen::Vector3d g_noise_cov_gyro;  // 陀螺仪数据噪声的协方差矩阵
    int m_if_dump_log = 1;             // 标志位, 控制是否打印log
    int m_if_acc_mul_G = 0;            // 标志位, 控制是否将加速度计数据的单位从g转换为m/s^2

    double m_sliding_window_tim = 10000; // 好像无用
    double m_camera_imu_td = 0;          // 好像无用, 实际使用的Camera与IMU之间的时间偏移为td_ext_i2c //! m_camera_imu_td

    double m_visual_init_time = 3e88;  // 无用
    double m_lidar_drag_cam_tim = 5.0; // 无用
    double m_if_lidar_start_first = 1; // 无用
    int m_if_lidar_can_start = 1;      // 无用
    std::string m_bag_file_name;       // 无用

    // std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> *m_camera_frame_buf = nullptr;
    std::deque<sensor_msgs::PointCloud2::ConstPtr> *m_liar_frame_buf = nullptr; // LiDAR数据队列

    // 构造函数
    Camera_Lidar_queue()
    {
        m_if_have_lidar_data = 0;
        m_if_have_camera_data = 0;
    };
    // 析构函数
    ~Camera_Lidar_queue(){};

    // 更新m_first_imu_time与m_last_imu_time
    double imu_in(const double in_time)
    {
        if (m_first_imu_time < 0)
        {
            m_first_imu_time = in_time; // 只执行1次, 第1帧IMU数据进来后赋值, 之后保持不变
        }
        // 更新m_last_imu_time, 使其始终为距当前最近的IMU帧时间戳, 最后的/最新的IMU帧时间戳
        m_last_imu_time = std::max(in_time, m_last_imu_time);
        // m_last_imu_time = in_time;
        return m_last_imu_time;
    }

    // 将m_if_have_lidar_data置为true
    int lidar_in(const double &in_time)
    {
        // cout << "LIDAR in " << endl;
        if (m_if_have_lidar_data == 0)
        {
            m_if_have_lidar_data = 1; // 只执行1次, 第1帧LiDAR数据进来后将m_if_have_lidar_data置为true, 之后保持不变
            // cout << ANSI_COLOR_BLUE_BOLD << "Have LiDAR data" << endl;
        }
        if (in_time < m_last_imu_time - m_sliding_window_tim) // 好像无用
        {
            std::cout << ANSI_COLOR_RED_BOLD << "LiDAR incoming frame too old, need to be drop!!!" << ANSI_COLOR_RESET << std::endl;
            // TODO: Drop LiDAR frame
        }
        // m_last_lidar_time = in_time;
        return 1;
    }

    // 无用
    int camera_in(const double &in_time)
    {
        if (in_time < m_last_imu_time - m_sliding_window_tim)
        {
            std::cout << ANSI_COLOR_RED_BOLD << "Camera incoming frame too old, need to be drop!!!" << ANSI_COLOR_RESET << std::endl;
            // TODO: Drop camera frame
        }
        return 1;
    }

    double get_lidar_front_time()
    {
        if (m_liar_frame_buf != nullptr && m_liar_frame_buf->size())
        {
            // 更新m_last_lidar_time, 使其始终为距当前最近的LiDAR帧时间戳, 最后的/最新的LiDAR帧时间戳
            //? 为什么要加0.1
            m_last_lidar_time = m_liar_frame_buf->front()->header.stamp.toSec() + 0.1;
            return m_last_lidar_time;
        }
        else
        {
            return -3e88;
        }
    }

    double get_camera_front_time()
    {
        // 只是返回m_last_visual_time, 更新m_last_visual_time的步骤在其他位置
        // m_camera_imu_td好像无用, 实际使用的Camera与IMU之间的时间偏移为td_ext_i2c
        return m_last_visual_time + m_camera_imu_td;
    }

    // 控制是否可以处理Camera数据, 可以处理时返回true, 否则返回false
    bool if_camera_can_process()
    {
        m_if_have_camera_data = 1;
        double cam_last_time = get_camera_front_time();
        double lidar_last_time = get_lidar_front_time();

        if (m_if_have_lidar_data != 1)
        {
            return true;
        }

        if (cam_last_time < 0 || lidar_last_time < 0)
        {
            return false;
        }

        // 等到当前Camera数据之前的LiDAR数据全都处理完, 才可以开始处理Camera数据
        if (lidar_last_time <= cam_last_time)
        {
            // LiDAR data need process first.
            // return true;
            return false;
        }
        else
        {
            // scope_color(ANSI_COLOR_YELLOW_BOLD);
            // cout << "Camera can update, " << get_lidar_front_time() - m_first_imu_time << " | " << get_camera_front_time() - m_first_imu_time << endl;
            return true;
        }
        return false;
    }

    // 控制是否可以处理LiDAR数据, 可以处理时返回true, 否则返回false
    bool if_lidar_can_process()
    {
        // m_if_have_lidar_data = 1;
        double cam_last_time = get_camera_front_time();
        double lidar_last_time = get_lidar_front_time();
        if (m_if_have_camera_data == 0)
        {
            return true;
        }

        if (cam_last_time < 0 || lidar_last_time < 0)
        {
            // cout << "Cam_tim = " << cam_last_time << ", lidar_last_time = " << lidar_last_time << endl;
            return false;
        }

        // 等到当前LiDAR数据之前的Camera数据全都处理完, 才可以开始处理LiDAR数据
        if (lidar_last_time > cam_last_time)
        {
            // Camera data need process first.
            return false;
        }
        else
        {
            // scope_color(ANSI_COLOR_BLUE_BOLD);
            // cout << "LiDAR can update, " << get_lidar_front_time() - m_first_imu_time << " | " << get_camera_front_time() - m_first_imu_time << endl;
            // printf_line;
            return true;
        }
        return false;
    }

    // 无用
    double time_wrt_first_imu_time(double &time)
    {
        return time - m_first_imu_time;
    }

    // 无用
    void display_last_cam_LiDAR_time()
    {
        double cam_last_time = get_camera_front_time();
        double lidar_last_time = get_lidar_front_time();
        scope_color(ANSI_COLOR_GREEN_BOLD);
        cout << std::setprecision(15) << "Camera time = " << cam_last_time << ", LiDAR last time =  " << lidar_last_time << endl;
    }
};

// 当前处理的LiDAR帧与对应的若干帧IMU数据封装在结构体MeasureGroup中
struct MeasureGroup
{
    MeasureGroup()
    {
        this->lidar.reset(new PointCloudXYZINormal());
    };
    double lidar_beg_time;                      // 当前处理的LiDAR帧中第1个点的时间戳, 当前处理的LiDAR帧时间戳
    double lidar_end_time;                      // 当前处理的LiDAR帧中最后1个点的时间戳
    PointCloudXYZINormal::Ptr lidar;            // 当前处理的LiDAR帧
    std::deque<sensor_msgs::Imu::ConstPtr> imu; // 当前处理的若干帧IMU数据
};

struct StatesGroup
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 29维的状态向量
    Eigen::Matrix3d rot_end;     // [0-2] the estimated attitude (rotation matrix) at the end lidar point
    Eigen::Vector3d pos_end;     // [3-5] the estimated position at the end lidar point (world frame)
    Eigen::Vector3d vel_end;     // [6-8] the estimated velocity at the end lidar point (world frame)
    Eigen::Vector3d bias_g;      // [9-11] gyroscope bias
    Eigen::Vector3d bias_a;      // [12-14] accelerator bias
    Eigen::Vector3d gravity;     // [15-17] the estimated gravity acceleration
    Eigen::Matrix3d rot_ext_i2c; // [18-20] Extrinsic between IMU frame to Camera frame on rotation. IMU坐标系到Camera坐标系的基变换矩阵
    Eigen::Vector3d pos_ext_i2c; // [21-23] Extrinsic between IMU frame to Camera frame on position. 在IMU坐标系下取的IMU-Camera平移
    double td_ext_i2c_delta;     // [24]    Extrinsic between IMU frame to Camera frame on time-offset. IMU与Camera之间的时间偏移 //! td_ext_i2c_delta
    vec_4 cam_intrinsic;         // [25-28] Intrinsice of camera [fx, fy, cx, cy]

    Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> cov; // states covariance. 状态向量的协方差矩阵
    double last_update_time = 0;
    double td_ext_i2c;

    // 状态向量构造函数, 初始化状态向量
    StatesGroup()
    {
        rot_end = Eigen::Matrix3d::Identity();
        pos_end = vec_3::Zero();
        vel_end = vec_3::Zero();
        bias_g = vec_3::Zero();
        bias_a = vec_3::Zero();
        gravity = Eigen::Vector3d(0.0, 0.0, 9.805);
        // gravity = Eigen::Vector3d(0.0, 9.805, 0.0);
        rot_ext_i2c = Eigen::Matrix3d::Identity();
        pos_ext_i2c = vec_3::Zero();
        td_ext_i2c_delta = 0;

        cov = Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity() * INIT_COV;
        // cov.block(18, 18, 6,6) *= 0.1;
        last_update_time = 0;
        td_ext_i2c = 0;
    }

    // 状态向量析构函数
    ~StatesGroup() {}

    // 定义状态向量的"+"法
    StatesGroup operator+(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
    {
        StatesGroup a = *this;
        // a.rot_end = this->rot_end * Sophus::SO3d::exp(vec_3(state_add(0, 0), state_add(1, 0), state_add(2, 0) ) );
        a.rot_end = this->rot_end * Exp(state_add(0), state_add(1), state_add(2));
        a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
        a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
        a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
        a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);
#if ESTIMATE_GRAVITY
        a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
#endif
        a.cov = this->cov;
        a.last_update_time = this->last_update_time;
#if ENABLE_CAMERA_OBS
        a.rot_ext_i2c = this->rot_ext_i2c * Exp(state_add(18), state_add(19), state_add(20));
        a.pos_ext_i2c = this->pos_ext_i2c + state_add.block<3, 1>(21, 0);
        a.td_ext_i2c_delta = this->td_ext_i2c_delta + state_add(24);
        a.cam_intrinsic = this->cam_intrinsic + state_add.block(25, 0, 4, 1);
#endif
        return a;
    }

    // 定义状态向量的"+="法
    StatesGroup &operator+=(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
    {
        this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        this->pos_end += state_add.block<3, 1>(3, 0);
        this->vel_end += state_add.block<3, 1>(6, 0);
        this->bias_g += state_add.block<3, 1>(9, 0);
        this->bias_a += state_add.block<3, 1>(12, 0);
#if ESTIMATE_GRAVITY
        this->gravity += state_add.block<3, 1>(15, 0);
#endif
#if ENABLE_CAMERA_OBS
        this->rot_ext_i2c = this->rot_ext_i2c * Exp(state_add(18), state_add(19), state_add(20));
        this->pos_ext_i2c = this->pos_ext_i2c + state_add.block<3, 1>(21, 0);
        this->td_ext_i2c_delta = this->td_ext_i2c_delta + state_add(24);
        this->cam_intrinsic = this->cam_intrinsic + state_add.block(25, 0, 4, 1);
#endif
        return *this;
    }

    // 定义状态向量的"-"法
    Eigen::Matrix<double, DIM_OF_STATES, 1> operator-(const StatesGroup &b)
    {
        Eigen::Matrix<double, DIM_OF_STATES, 1> a;
        Eigen::Matrix3d rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3, 1>(0, 0) = SO3_LOG(rotd);
        a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
        a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
        a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
        a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
        a.block<3, 1>(15, 0) = this->gravity - b.gravity;
#if ENABLE_CAMERA_OBS
        Eigen::Matrix3d rotd_ext_i2c(b.rot_ext_i2c.transpose() * this->rot_ext_i2c);
        a.block<3, 1>(18, 0) = SO3_LOG(rotd_ext_i2c);
        a.block<3, 1>(21, 0) = this->pos_ext_i2c - b.pos_ext_i2c;
        a(24) = this->td_ext_i2c_delta - b.td_ext_i2c_delta;
        a.block<4, 1>(25, 0) = this->cam_intrinsic - b.cam_intrinsic;
#endif
        return a;
    }

    // 在终端打印状态向量
    static void display(const StatesGroup &state, std::string str = std::string("State: "))
    {
        vec_3 angle_axis = SO3_LOG(state.rot_end) * 57.3;
        printf("%s |", str.c_str());
        printf("[%.5f] | ", state.last_update_time);
        printf("(%.3f, %.3f, %.3f) | ", angle_axis(0), angle_axis(1), angle_axis(2));
        printf("(%.3f, %.3f, %.3f) | ", state.pos_end(0), state.pos_end(1), state.pos_end(2));
        printf("(%.3f, %.3f, %.3f) | ", state.vel_end(0), state.vel_end(1), state.vel_end(2));
        printf("(%.3f, %.3f, %.3f) | ", state.bias_g(0), state.bias_g(1), state.bias_g(2));
        printf("(%.3f, %.3f, %.3f) \r\n", state.bias_a(0), state.bias_a(1), state.bias_a(2));
    }
};

// 弧度转角度
template <typename T>
T rad2deg(T radians)
{
    return radians * 180.0 / PI_M;
}

// 角度转弧度
template <typename T>
T deg2rad(T degrees)
{
    return degrees * PI_M / 180.0;
}

// 将若干数据封装为Pose6D变量
template <typename T>
auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g,
                const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)
            rot_kp.rot[i * 3 + j] = R(i, j);
    }
    // Eigen::Map<Eigen::Matrix3d>(rot_kp.rot, 3,3) = R;
    return std::move(rot_kp); // 将左值转换为右值引用
}

#endif
