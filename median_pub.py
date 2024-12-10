import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

class PointCloudMedianRViz:
    def __init__(self):
        rospy.init_node('point_cloud_median_rviz', anonymous=True)
        self.sub = rospy.Subscriber('/processed_point_cloud', PointCloud2, self.point_cloud_callback, queue_size=1)
        self.pub = rospy.Publisher('/median_point_cloud', PointCloud2, queue_size=10)

    def point_cloud_callback(self, msg):
        # 点群データを取り出す
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        # 点群データがない場合は無視
        if len(points) == 0:
            rospy.logwarn("Empty point cloud received.")
            return

        # 各軸の中央値を計算
        points_np = np.array(points)  # NumPy配列に変換
        median_point = np.median(points_np, axis=0)  # 中央値を計算

        # 中央値をRVizにパブリッシュ
        self.publish_to_rviz(median_point)

    def publish_to_rviz(self, median_point):
        # ヘッダー作成
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_depth_optical_frame"  # フレームIDを適切に設定

        # 中央値を含む1点のPointCloud2メッセージを作成
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        median_point_cloud = pc2.create_cloud(header, fields, [median_point])

        # トピックにパブリッシュ
        self.pub.publish(median_point_cloud)
        rospy.loginfo(f"Median point published: {median_point}")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        plotter = PointCloudMedianRViz()
        plotter.run()
    except rospy.ROSInterruptException:
        pass

