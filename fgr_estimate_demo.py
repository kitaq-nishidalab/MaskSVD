import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import Registration_test_jurai_fast as Registration
import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import PointCloud2
import copy
from geometry_msgs.msg import TransformStamped
import tf_conversions
import warnings
import os
import datetime
import tf2_ros
from tf2_msgs.msg import TFMessage
import time
import pandas as pd
import matplotlib.pyplot as plt


def create_pcd(xyz, color):
    n = xyz.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def inverse_transform_matrix(transform_matrix):
    # 変換行列の逆行列を計算
    return np.linalg.inv(transform_matrix)

class PointCloudProcessor:
	def __init__(self):
		rospy.init_node('posture_estimation', anonymous=True)
		#TFブロードキャスト
		self.br = tf2_ros.StaticTransformBroadcaster()
		self.frame_id = None
		self.child_frame_id = None
		self.sub_tf = rospy.Subscriber('/tf_static', TFMessage, self.tf_callback)

		self.measured_pcd = None
		#self.model_pcd = o3d.io.read_point_cloud("TNUTEJN016_10000_half.pcd")
		self.model_pcd = o3d.io.read_point_cloud("WMU2LR2020_10000_half4.pcd")
		#self.model_pcd = o3d.io.read_point_cloud("/home/nishidalab0/MaskNet/TNUTEJN016_half2.pcd")

		#self.masknet_load = torch.load("checkpoint/model_weight_epoch300_717_batchsize32.pth")
		#self.masknet_load = torch.load("checkpoint/model_weight_epoch300_batchsize32_plane.pth")
		#self.masknet_load = torch.load("checkpoint/pretrained.pth")
		#self.masknet_load = torch.load("checkpoint/first_train.pth")

		self.sub = rospy.Subscriber('/processed_point_cloud', PointCloud2, self.point_cloud_callback)
		#self.pub = rospy.Publisher('/estimated_transform', TransformStamped, queue_size=10)
		#self.pub2 = rospy.Publisher('/processed2_point_cloud', PointCloud2, queue_size=10)
		#self.pub2 = rospy.Publisher('/model_point_cloud', PointCloud2, queue_size=10)
		#self.pub3 = rospy.Publisher('/masked_point_cloud', PointCloud2, queue_size=10)

		# 現在の時刻を取得してディレクトリ名に追加
		#current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		#self.save_dir = f"saved_point_clouds/{current_time}"

		#if not os.path.exists(self.save_dir):
		#	os.makedirs(self.save_dir)

		# 点群の保存カウンター
		#self.counter = 0
		# 計測用のリスト
		self.processing_times = []
		
	

	def point_cloud_callback(self, msg):
		start_time = time.time()  # 開始時刻
		# Convert ROS PointCloud2 message to Open3D PointCloud
		pc_data = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
		points = np.array(list(pc_data))
		if points.size == 0:
			return
		xyz = points[:, :3]
		try:
			self.measured_pcd = create_pcd(xyz, [0, 1, 0])

			# 点群を保存
			#self.save_point_cloud(self.measured_pcd, "measure")

			self.process_point_cloud()
		except ValueError as e:
			rospy.logerr("ValueError: %s", str(e))
			
		# 処理時間の記録
		end_time = time.time()  # 終了時刻
		processing_time = end_time - start_time
			
		# 処理回数が1000回に到達したら保存してグラフを表示
		if len(self.processing_times) < 1000:
			self.processing_times.append(processing_time)
			#print(len(self.processing_times))
			
		if len(self.processing_times) == 1000:
			self.save_processing_times()
	
	def tf_callback(self, msg):
		if self.frame_id is not None:
			return
		for transform in msg.transforms:
			if transform.child_frame_id == "camera_depth_optical_frame":
				self.frame_id = transform.header.frame_id
				self.child_frame_id = transform.child_frame_id
				self.translation_x = transform.transform.translation.x
				self.translation_y = transform.transform.translation.y
				self.translation_z = transform.transform.translation.z
				self.rotation_x = transform.transform.rotation.x
				self.rotation_y = transform.transform.rotation.y
				self.rotation_z = transform.transform.rotation.z
				self.rotation_w = transform.transform.rotation.w
			
	def process_point_cloud(self):
		torch.cuda.empty_cache()
		#ここで, Proposed methodのプログラムを書く (self.)
		if self.measured_pcd is None:
			return
		numpy_model_pcd = np.array(self.model_pcd .points)
		numpy_measured_pcd = np.array(self.measured_pcd.points)
		
		model_pcd = copy.deepcopy(self.model_pcd)
		measured_pcd = copy.deepcopy(self.measured_pcd)
		model_pcd.paint_uniform_color([1.0, 0, 0])
		measured_pcd.paint_uniform_color([0, 1.0, 0])
		# テンプレート点群とソース点群の表示
		###o3d.visualization.draw_geometries([model_pcd, measured_pcd])

		template_tensor = torch.tensor(np.array(model_pcd.points), dtype=torch.float32).unsqueeze(0)
		source_tensor = torch.tensor(np.array(measured_pcd.points), dtype=torch.float32).unsqueeze(0)
		
		# 位置合わせ準備を行う（関数を呼び出す）
		registration_model = Registration.Registration()

		# 提案手法（MaskNet、SVD、ICP）の実行（実際のデータを代入）
		result_cheese = registration_model.register(template_tensor, source_tensor)
		est_T_cheese = result_cheese['est_T']     # est_T：ICPの変換行列
		self.publish_transform(est_T_cheese)
		
	
	def publish_transform(self, est_T_cheese):
		if est_T_cheese is None:
			print("est_T_cheese is None")
			return
		transform_msg = TransformStamped()
		
		# ROS Header
		transform_msg.header.stamp = rospy.Time.now()

		transform_msg.header.frame_id = self.child_frame_id
		transform_msg.child_frame_id = "Posture_of_object"  # ソースフレームの名前（適宜変更）
		
		# 変換行列をTransformに変換
		transform_matrix = est_T_cheese.cpu().numpy()[0]
		inverse_transform_matrix = np.linalg.inv(transform_matrix)

		translation = inverse_transform_matrix[:3, 3]
		#rotation_matrix = inverse_transform_matrix[:3, :3]
		
		# Quaternionに変換
		quaternion = tf_conversions.transformations.quaternion_from_matrix(inverse_transform_matrix)
		
		# TransformStampedメッセージに設定
		transform_msg.transform.translation.x = translation[0]
		transform_msg.transform.translation.y = translation[1]
		transform_msg.transform.translation.z = translation[2]
		transform_msg.transform.rotation.x = quaternion[0]
		transform_msg.transform.rotation.y = quaternion[1]
		transform_msg.transform.rotation.z = quaternion[2]
		transform_msg.transform.rotation.w = quaternion[3]
		
		#print(translation)
		# トランスフォームをブロードキャスト
		self.br.sendTransform(transform_msg)

		
    
	def publish_point_cloud(self, pcd, pcd2):
		if pcd is None:
			return
		
		# Open3DのPointCloudをROSのPointCloud2メッセージに変換
		points = np.asarray(pcd.points)
		header = rospy.Header()
		header.stamp = rospy.Time.now()
		header.frame_id = "camera_depth_optical_frame"  # 自分のフレームIDに置き換えてください
		
		pc_msg = pc2.create_cloud_xyz32(header, points)
		self.pub2.publish(pc_msg)

		# Open3DのPointCloudをROSのPointCloud2メッセージに変換
		points2 = np.asarray(pcd2.points)
		header2 = rospy.Header()
		header2.stamp = rospy.Time.now()
		header2.frame_id = "camera_depth_optical_frame"  # 自分のフレームIDに置き換えてください
		
		pc_msg2 = pc2.create_cloud_xyz32(header2, points2)
		self.pub3.publish(pc_msg)
	
	def save_point_cloud(self, pcd, char):
		# 点群の保存ファイル名
		filename = os.path.join(self.save_dir, f"point_cloud_{char}_{self.counter:05d}.pcd")
		o3d.io.write_point_cloud(filename, pcd)
		rospy.loginfo(f"Point cloud saved: {filename}")
		self.counter += 1
	def save_processing_times(self):
		# CSV保存
        	df = pd.DataFrame(self.processing_times, columns=["Processing Time"])
        	df.to_csv("fgr_processing_times.csv", index=False)
        	rospy.loginfo("Processing times saved to 'processing_times.csv'")

        	average_time = np.mean(self.processing_times)
        	print(f"Average execution time {average_time:.6f} seconds")
        	self.processing_times=[]



def main():
    processor = PointCloudProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
