#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

class PointCloudProcessor
{
public:
    PointCloudProcessor()
    {
        // サブスクライバの設定（PointCloud2メッセージを受け取る）
        sub_ = nh_.subscribe("/camera/depth/color/points", 1, &PointCloudProcessor::pointCloudCallback, this);
        
        // パブリッシャの設定（フィルタ処理後の点群をパブリッシュ）
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/processed_point_cloud", 10);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        // 受け取ったPointCloud2メッセージをPCL形式に変換
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        
        // パススルーフィルタでz軸に基づくフィルタリング
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.30, 0.30 * 1.085); // z軸の範囲を設定
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pass.filter(*cloud_filtered);
        
        // フィルタ後の点群をPointCloud2形式に変換
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);
        
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
