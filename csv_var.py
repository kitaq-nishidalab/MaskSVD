import rospy
import numpy as np
import csv
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformListener
from tf2_msgs.msg import TFMessage
import tf_conversions
import os

class PoseEstimator:
    def __init__(self):
        self.pose_list = []
        self.max_samples = 1000
        self.sub = rospy.Subscriber('/tf_static', TFMessage, self.pose_callback)

        # CSVファイルを初期化してヘッダー行を追加
        self.csv_file = 'pose_data.csv'
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y', 'z', 'roll', 'pitch', 'yaw'])

    def pose_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "Posture_of_object":
                # 1. クォータニオンを取得
                quat = (
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                )

                # 2. クォータニオンをロール、ピッチ、ヨーに変換
                euler = tf_conversions.transformations.euler_from_quaternion(quat)
                roll, pitch, yaw = np.degrees(euler)

                # 3. 平行移動とオイラー角をリストに追加
                translation = [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    roll,
                    pitch,
                    yaw
                ]

                self.pose_list.append(translation)

                # CSVファイルにサンプルを保存
                with open(self.csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(translation)

                if len(self.pose_list) >= self.max_samples:
                    self.calculate_statistics()

    def calculate_statistics(self):
        poses = np.array(self.pose_list)

        # 平行移動とロール・ピッチ・ヨーの平均を計算
        mean_pose = np.mean(poses, axis=0)
        std_dev_pose = np.std(poses, axis=0)

        # 位置の標準偏差の合計と向きの標準偏差の合計を計算
        position_std_dev_sum = np.sum(std_dev_pose[:3])  # x, y, z
        orientation_std_dev_sum = np.sum(std_dev_pose[3:])  # roll, pitch, yaw

        # 3シグマの計算
        three_sigma_upper = mean_pose + 3 * std_dev_pose
        three_sigma_lower = mean_pose - 3 * std_dev_pose

        # 結果をログに表示
        rospy.loginfo(f"Mean Pose (x, y, z, roll, pitch, yaw): {mean_pose}")
        rospy.loginfo(f"Standard Deviation (Position): {std_dev_pose[:3]}")
        rospy.loginfo(f"Standard Deviation (Orientation): {std_dev_pose[3:]}")
        rospy.loginfo(f"Sum of Position Standard Deviations: {position_std_dev_sum}")
        rospy.loginfo(f"Sum of Orientation Standard Deviations: {orientation_std_dev_sum}")
        rospy.loginfo(f"3 Sigma Upper Bound: {three_sigma_upper}")
        rospy.loginfo(f"3 Sigma Lower Bound: {three_sigma_lower}")

        # ポーズリストをクリアして次のサンプルを収集
        self.pose_list.clear()

def main():
    rospy.init_node('Var', anonymous=True)
    estimator = PoseEstimator()
    rospy.spin()

if __name__ == '__main__':
    main()

