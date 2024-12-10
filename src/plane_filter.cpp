#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

class PointCloudProcessor
{
public:
    PointCloudProcessor()
    {
        // サブスクライバの設定（PointCloud2メッセージを受け取る）
        sub_ = nh_.subscribe("/camera/depth/color/points", 1, &PointCloudProcessor::pointCloudCallback, this);
        
        // パブリッシャの設定（平面除去後の点群をパブリッシュ）
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/processed_point_cloud", 10);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        // 受け取ったPointCloud2メッセージをPCL形式に変換
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        
        // 平面検出のためのセグメンテーションオブジェクトの設定
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.005); // 平面と見なす点の最大距離

        // 平面のインデックスと係数を保存するための変数
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        // 平面検出の実行
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty())
        {
            ROS_WARN("Could not estimate a planar model for the given dataset.");
            return;
        }

        // 平面以外の点を抽出
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);  // 平面以外の点を残す
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*cloud_filtered);

        // ボクセルグリッドによるダウンサンプリングの設定
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(cloud_filtered);
        voxel_filter.setLeafSize(0.003f, 0.003f, 0.003f); // ボクセルサイズ（調整可能）
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*cloud_downsampled);

        // 外れ値除去フィルタの設定
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_downsampled);
        sor.setMeanK(50);  // 各点に対して参照する近傍点の数
        sor.setStddevMulThresh(1.0);  // 標準偏差の倍率で除去を決定
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_denoised(new pcl::PointCloud<pcl::PointXYZ>);
        sor.filter(*cloud_denoised);

        // ノイズ除去後の点群をPointCloud2形式に変換
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_denoised, output);

        // フレームIDとタイムスタンプを設定
        output.header.frame_id = msg->header.frame_id;
        output.header.stamp = ros::Time::now();

        // パブリッシュ
        pub_.publish(output);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_processor");
    PointCloudProcessor processor;
    ros::spin();
    return 0;
}
