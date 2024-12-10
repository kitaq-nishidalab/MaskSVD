import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import Registration
import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import PointCloud2
import copy
from geometry_msgs.msg import TransformStamped
import tf_conversions
import warnings
import os
import datetime
warnings.filterwarnings("ignore", category=UserWarning)  # UserWarningを無視

def create_pcd(xyz, color):
    n = xyz.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

class PointCloudProcessor:
	def __init__(self):
		rospy.init_node('posture_estimation', anonymous=True)

		self.measured_pcd = None
		#self.model_pcd = o3d.io.read_point_cloud("TNUTEJN016_100000.pcd")
		
		self.model_pcd = o3d.io.read_point_cloud("FilteredPointCloud.ply")
		# Y軸を中心に180度回転させる回転行列を定義
		R = self.model_pcd.get_rotation_matrix_from_axis_angle([0, np.pi, 0])
		# 点群に回転を適用
		self.model_pcd.rotate(R, center=(0, 0, 0))

		#self.model_pcd = o3d.io.read_point_cloud("WMU2LR2020.pcd")

		self.masknet_load = torch.load("checkpoint/model_weight_epoch300_batchsize32_plane.pth")

		self.sub = rospy.Subscriber('/processed2_point_cloud', PointCloud2, self.point_cloud_callback)
		self.pub = rospy.Publisher('/estimated_transform', TransformStamped, queue_size=10)
		#self.pub2 = rospy.Publisher('/processed2_point_cloud', PointCloud2, queue_size=10)
		self.pub2 = rospy.Publisher('/model_point_cloud', PointCloud2, queue_size=10)
		self.pub3 = rospy.Publisher('/masked_point_cloud', PointCloud2, queue_size=10)

		# 現在の時刻を取得してディレクトリ名に追加
		current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		self.save_dir = f"saved_point_clouds/{current_time}"

		#if not os.path.exists(self.save_dir):
		#	os.makedirs(self.save_dir)

		# 点群の保存カウンター
		self.counter = 0
		

	def point_cloud_callback(self, msg):
		# Convert ROS PointCloud2 message to Open3D PointCloud
		pc_data = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
		points = np.array(list(pc_data))
		if points.size == 0:
			return
		xyz = points[:, :3]
		try:
			self.measured_pcd = create_pcd(xyz, [0, 1, 0])

			# 点群を保存
			#self.save_point_cloud(self.measured_pcd)

			self.process_point_cloud()
		except ValueError as e:
			rospy.logerr("ValueError: %s", str(e))
			
	def process_point_cloud(self):
		torch.cuda.empty_cache()
		#ここで, Proposed methodのプログラムを書く (self.)
		if self.measured_pcd is None:
			return
		numpy_model_pcd = np.array(self.model_pcd .points)
		numpy_measured_pcd = np.array(self.measured_pcd.points)
		len_model_pcd = len(numpy_model_pcd)
		len_measured_pcd = len(numpy_measured_pcd)
		#print("モデルの点群数：", len_model_pcd)
		#print("計測の点群数：", len_measured_pcd)

		### 計測点群の密度に合わせる ###
		min_bound = np.min(numpy_measured_pcd, axis=0)
		max_bound = np.max(numpy_measured_pcd, axis=0)

		box_width = max_bound[0] - min_bound[0]
		box_height = max_bound[1] - min_bound[1]
		box_depth = max_bound[2] - min_bound[2]
		###print("ボクセルサイズの大きさ", box_width, box_height, box_depth)

		points_inside = numpy_measured_pcd[(numpy_measured_pcd >= min_bound) & (numpy_measured_pcd <= max_bound)].reshape(-1, 3)

		density = len_measured_pcd / ( box_width * box_height * box_depth )
		###print("密度：", density)   
		weight = 0.6
		voxel_size = weight * (1.0 / density) ** (1/3)
        ###print("ダウンサンプリングのボックスの大きさ：", voxel_size)

		#model_pcd = self.model_pcd.voxel_down_sample(voxel_size) #初期値0.006
		model_pcd = self.model_pcd #初期値0.006

		measured_pcd = copy.deepcopy(self.measured_pcd)
		model_pcd.paint_uniform_color([1.0, 0, 0])
		measured_pcd.paint_uniform_color([1.0, 0, 0])
		# テンプレート点群とソース点群の表示
		###o3d.visualization.draw_geometries([model_pcd, measured_pcd])

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.masknet_load.to(device)
		template_tensor = torch.tensor(np.array(model_pcd.points), dtype=torch.float32).unsqueeze(0)
		source_tensor = torch.tensor(numpy_measured_pcd, dtype=torch.float32).unsqueeze(0)
		template_tensor = template_tensor.to(device)
		source_tensor = source_tensor.to(device)
		
		# 位置合わせ準備を行う（関数を呼び出す）
		registration_model = Registration.Registration()

		#with torch.no_grad():
		#	masked_template_cheese, predicted_mask_cheese = self.masknet_load(template_tensor, source_tensor)

		# 提案手法（MaskNet、SVD、ICP）の実行（実際のデータを代入）
		result_cheese = registration_model.register(template_tensor, source_tensor)
		est_T_cheese = result_cheese['est_T']     # est_T：ICPの変換行列
		#print(est_T_cheese)
		#print(type(est_T_cheese))  # <class 'torch.Tensor'>

		# SVD+ICP処理、点群の表示
		#Registration.display_results_sample(
		#	template_tensor.detach().cpu().numpy()[0], 
		#	source_tensor.detach().cpu().numpy()[0], 
		#	est_T_cheese.detach().cpu().numpy()[0], 
		#	masked_template_cheese.detach().cpu().numpy()[0], 
		#)

		#masked_numpy = masked_template_cheese.detach().cpu().numpy()[0]
		#masked_pcd = o3d.geometry.PointCloud()
		#masked_pcd.points = o3d.utility.Vector3dVector(masked_numpy)

		self.publish_transform(est_T_cheese)
		self.publish_point_cloud(self.model_pcd)
	
	def publish_transform(self, est_T_cheese):
		if est_T_cheese is None:
			return
		transform_msg = TransformStamped()
		
		# ROS Header
		transform_msg.header.stamp = rospy.Time.now()
		transform_msg.header.frame_id = "camera_depth_optical_frame"  # ソースフレームの名前（適宜変更）
		
		# 変換行列をTransformに変換
		transform_matrix = est_T_cheese.cpu().numpy()[0]
		translation = transform_matrix[:3, 3]
		rotation_matrix = transform_matrix[:3, :3]
		
		# Quaternionに変換
		quaternion = tf_conversions.transformations.quaternion_from_matrix(transform_matrix)
		
		# TransformStampedメッセージに設定
		transform_msg.transform.translation.x = translation[0]
		transform_msg.transform.translation.y = translation[1]
		transform_msg.transform.translation.z = translation[2]
		transform_msg.transform.rotation.x = quaternion[0]
		transform_msg.transform.rotation.y = quaternion[1]
		transform_msg.transform.rotation.z = quaternion[2]
		transform_msg.transform.rotation.w = quaternion[3]
		
		# パブリッシュ
		self.pub.publish(transform_msg)

	def publish_point_cloud(self, pcd):
		if pcd is None:
			return
		
		# Open3DのPointCloudをROSのPointCloud2メッセージに変換
		points = np.asarray(pcd.points)
		header = rospy.Header()
		header.stamp = rospy.Time.now()
		header.frame_id = "camera_depth_optical_frame"  # 自分のフレームIDに置き換えてください
		
		pc_msg = pc2.create_cloud_xyz32(header, points)
		self.pub2.publish(pc_msg)

		"""
		# Open3DのPointCloudをROSのPointCloud2メッセージに変換
		points2 = np.asarray(pcd2.points)
		header2 = rospy.Header()
		header2.stamp = rospy.Time.now()
		header2.frame_id = "camera_depth_optical_frame"  # 自分のフレームIDに置き換えてください
		
		pc_msg2 = pc2.create_cloud_xyz32(header2, points2)
		self.pub3.publish(pc_msg2)
		"""
	
	def save_point_cloud(self, pcd):
		# 点群の保存ファイル名
		filename = os.path.join(self.save_dir, f"point_cloud_{self.counter:05d}.pcd")
		o3d.io.write_point_cloud(filename, pcd)
		rospy.loginfo(f"Point cloud saved: {filename}")
		self.counter += 1

		



def main():
    processor = PointCloudProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()

"""

# SVD+ICP処理、点群の表示
Registration_demo.display_results_sample(
	template_cheese.detach().cpu().numpy()[0], 
	source_cheese.detach().cpu().numpy()[0], 
	est_T_cheese.detach().cpu().numpy()[0], 
	masked_template_cheese.detach().cpu().numpy()[0],
	pattern)
	
"""
	
