import rospy
import tf2_ros
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
import numpy as np
import open3d as o3d
from sensor_msgs import point_cloud2
from scipy.spatial.transform import Rotation as R

def rpy_to_quaternion(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    q_w = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q_x = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q_y = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    q_z = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    return q_x, q_y, q_z, q_w

class TransformPublisher:
    def __init__(self):
        rospy.init_node('ICP_GT', anonymous=True)
        self.br = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber('/processed_point_cloud', PointCloud2, self.point_cloud_callback)

    def point_cloud_callback(self, point_cloud_msg):
        # 受信した点群をOpen3D形式に変換
        cloud_points = list(point_cloud2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(cloud_points)

        # CADモデル点群の読み込み（例としてload cad model cloud）
        target_cloud = o3d.io.read_point_cloud("TNUTEJN016_100000.pcd")  # CADモデルの点群ファイル

        # 手動で設定した初期位置と回転
        #PatternA
        #initial_translation = [0.1, -0.05, 0.322]
        #initial_rotation = rpy_to_quaternion(roll=-4, pitch=0, yaw=235)
        #PatternB
        initial_translation = [0.05, 0.05, 0.325]
        initial_rotation = rpy_to_quaternion(roll=-2, pitch=0, yaw=45)
        #PatternC
        #initial_translation = [-0.188, 0.073, 0.333]
        #initial_rotation = rpy_to_quaternion(roll=0, pitch=0, yaw=155)
        
        # 四元数を回転行列に変換
        rotation_matrix = R.from_quat(initial_rotation).as_matrix()
        
        # 初期変換行列の設定
        initial_transform = np.eye(4)
        initial_transform[:3, :3] = rotation_matrix
        initial_transform[:3, 3] = initial_translation
        initial_transform_inv = np.linalg.inv(initial_transform)
        
        #print(initial_transform)

        # ICPの実行
        icp_result = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, 0.0001, initial_transform_inv,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # ICP変換結果の取得
        icp_transform = icp_result.transformation
        icp_transform_inv = np.linalg.inv(icp_transform)
        #print(icp_transform)

        # 姿勢の四元数化
        rotation_matrix = icp_transform_inv[:3, :3]
        q_x, q_y, q_z, q_w = self.rotation_matrix_to_quaternion(rotation_matrix)
        #transformed_source_cloud = source_cloud.transform(icp_transform)
        #o3d.visualization.draw_geometries([transformed_source_cloud, target_cloud])
        
        # トランスフォームのパブリッシュ
        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = "camera_depth_optical_frame"
        transform_msg.child_frame_id = "Pattern+ICP"
        transform_msg.transform.translation.x = icp_transform_inv[0, 3]
        transform_msg.transform.translation.y = icp_transform_inv[1, 3]
        transform_msg.transform.translation.z = icp_transform_inv[2, 3]
        transform_msg.transform.rotation.x = q_x
        transform_msg.transform.rotation.y = q_y
        transform_msg.transform.rotation.z = q_z
        transform_msg.transform.rotation.w = q_w
        self.br.sendTransform(transform_msg)

    def rotation_matrix_to_quaternion(self, R):
        """回転行列から四元数を計算する"""
        q_w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q_x = (R[2, 1] - R[1, 2]) / (4 * q_w)
        q_y = (R[0, 2] - R[2, 0]) / (4 * q_w)
        q_z = (R[1, 0] - R[0, 1]) / (4 * q_w)
        return q_x, q_y, q_z, q_w

def main():
    transform_publisher = TransformPublisher()
    rospy.spin()

if __name__ == '__main__':
    main()
