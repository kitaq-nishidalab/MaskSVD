import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class PointCloudMedianPlot:
    def __init__(self):
        rospy.init_node('point_cloud_median_plot', anonymous=True)
        self.sub = rospy.Subscriber('/processed_point_cloud', PointCloud2, self.point_cloud_callback, queue_size=1)
        self.median_points = []  # 中央値を保存するリスト
        self.max_points = 100  # 最大収集データ数

    def point_cloud_callback(self, msg):
        # 点群データを取り出す
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        # 点群データがない場合は無視
        if len(points) == 0:
            rospy.logwarn("Empty point cloud received.")
            return

        # 各軸の中央値を計算して保存
        points_np = np.array(points)  # NumPy配列に変換
        median_point = np.median(points_np, axis=0) * 1000  # 中央値を計算し、m から mm に変換
        self.median_points.append(median_point)

        rospy.loginfo(f"Median point calculated: {median_point}")

        # 収集が完了したらコールバックを終了して描画
        if len(self.median_points) >= self.max_points:
            rospy.loginfo("Finished collecting median points. Now saving the graph.")
            self.save_graph()
            rospy.signal_shutdown("Finished processing.")

    def save_graph(self):
        # 3Dグラフのセットアップ
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 中央値のプロット
        median_points = np.array(self.median_points)
        ax.scatter(median_points[:, 0], median_points[:, 1], median_points[:, 2], c='r')

        # 各中央値を結ぶ線を描画
        ax.plot(median_points[:, 0], median_points[:, 1], median_points[:, 2], c='b')

        # 軸ラベルを削除
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # グラフを画像ファイルに保存
        filename = "median_plot.png"
        plt.savefig(filename, dpi=300)  # 高解像度で保存
        rospy.loginfo(f"Graph saved as {filename}")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        plotter = PointCloudMedianPlot()
        plotter.run()
    except rospy.ROSInterruptException:
        pass
