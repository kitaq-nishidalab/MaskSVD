import rospy
import tf2_ros
from tf2_msgs.msg import TFMessage
import numpy as np
from geometry_msgs.msg import TransformStamped
def rpy_to_quaternion(roll, pitch, yaw):
    # RPY角をラジアンに変換
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # 四元数の計算
    q_w = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q_x = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q_y = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    q_z = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)

    return q_x, q_y, q_z, q_w
    
    

class TransformPublisher:
    def __init__(self):
        rospy.init_node('tf_transformer', anonymous=True)
        
        # トランスフォームのブロードキャスターを作成
        self.br = tf2_ros.StaticTransformBroadcaster()

        # それぞれのトランスフォームの情報をリストに格納
        self.transforms = [
            {
                "child_frame_id": "ground_truth_tf1",
                "translation": (0.1, -0.05, 0.32),
                "rotation": rpy_to_quaternion(roll=0, pitch=0, yaw=235),
            },
            {
                "child_frame_id": "ground_truth_tf2",
                "translation": (0.05, 0.05, 0.323),
                "rotation": rpy_to_quaternion(roll=5, pitch=0, yaw=45),
            },
            {
                "child_frame_id": "ground_truth_tf3",
                "translation": (-0.15, 0, 0.33),
                "rotation": rpy_to_quaternion(roll=0, pitch=0, yaw=135),
            }
        ]

        # Subscribe to tf_static topic
        self.sub = rospy.Subscriber('/tf_static', TFMessage, self.tf_callback)

    def tf_callback(self, msg):
        if not self.transforms:  # すでにトランスフォームを送信した場合
            return
        
        for transform in msg.transforms:
            if transform.child_frame_id == "camera_depth_optical_frame":
                # Publish the modified transforms
                self.publish_transforms()
                break  # 一度パブリッシュしたらループを抜ける

    def publish_transforms(self):
        parent_frame_id = "camera_depth_optical_frame"

        # TransformStampedのリストを作成
        transform_msgs = []
        
        for t in self.transforms:
            # Create a TransformStamped message
            transform_msg = TransformStamped()
            transform_msg.header.stamp = rospy.Time.now()
            transform_msg.header.frame_id = parent_frame_id
            transform_msg.child_frame_id = t["child_frame_id"]
            
            # Set translation and rotation
            transform_msg.transform.translation.x = t["translation"][0]
            transform_msg.transform.translation.y = t["translation"][1]
            transform_msg.transform.translation.z = t["translation"][2]
            transform_msg.transform.rotation.x = t["rotation"][0]
            transform_msg.transform.rotation.y = t["rotation"][1]
            transform_msg.transform.rotation.z = t["rotation"][2]
            transform_msg.transform.rotation.w = t["rotation"][3]
            
            # TransformStampedメッセージをリストに追加
            transform_msgs.append(transform_msg)

        # 一度にすべてのトランスフォームを送信
        self.br.sendTransform(transform_msgs)

def main():
    transform_publisher = TransformPublisher()
    rospy.spin()

if __name__ == '__main__':
    main()
