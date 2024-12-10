import rospy
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

class PoseEstimator:
    def __init__(self):
        self.pose_list = []
        self.max_samples = 1000
        self.sub = rospy.Subscriber('/tf_static', TFMessage, self.pose_callback)

    def pose_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "Posture_of_object":
                # クォータニオンをそのまま取得
                quat = np.array([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])

                # 平行移動とクォータニオンをリストに追加
                translation = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])

                # 平行移動とクォータニオンを結合して追加
                pose = np.concatenate((translation, quat))
                self.pose_list.append(pose)

                if len(self.pose_list) >= self.max_samples:
                    self.calculate_statistics()

    def calculate_statistics(self):
        poses = np.array(self.pose_list)

        # 平行移動とクォータニオンの平均と標準偏差を計算
        mean_pose = np.mean(poses, axis=0)
        std_dev_pose = np.std(poses, axis=0)

        # 位置の標準偏差の合計とクォータニオンの標準偏差の合計を計算
        position_std_dev_sum = np.sum(std_dev_pose[:3])  # x, y, z
        quaternion_std_dev_sum = np.sum(std_dev_pose[3:])  # クォータニオン (x, y, z, w)

        # 3シグマの計算
        three_sigma_upper = mean_pose + 3 * std_dev_pose
        three_sigma_lower = mean_pose - 3 * std_dev_pose

        # 結果をログに表示
        rospy.loginfo(f"Mean Pose (x, y, z, quat_x, quat_y, quat_z, quat_w): {mean_pose}")
        rospy.loginfo(f"Standard Deviation (Position): {std_dev_pose[:3]}")
        rospy.loginfo(f"Standard Deviation (Quaternion): {std_dev_pose[3:]}")
        rospy.loginfo(f"Sum of Position Standard Deviations: {position_std_dev_sum}")
        rospy.loginfo(f"Sum of Quaternion Standard Deviations: {quaternion_std_dev_sum}")
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
