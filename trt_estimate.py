import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import Registration_SVD as Registration
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
import tensorrt as trt


def convert_model_to_trt(masknet_load, template_tensor, source_tensor):
    # TensorRTのビルダーを作成
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    
    # ネットワーク定義を作成
    network =  builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Create optimization profile for dynamic batch size
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()

    profile.set_shape("template", (1, *template_tensor.shape[1:]), (8, *template_tensor.shape[1:]), (16, *template_tensor.shape[1:]))
    profile.set_shape("source", (1, *source_tensor.shape[1:]), (8, *source_tensor.shape[1:]), (16, *source_tensor.shape[1:]))
    config.add_optimization_profile(profile)s

    # Build the model with the specified configurations
    engine = builder.build_engine(network, config)
    
    return engine
    
def run_inference_with_trt(masknet_trt_model, template_tensor, source_tensor):
    # テンソルをGPU上に転送
    template_tensor = template_tensor.cuda()
    source_tensor = source_tensor.cuda()
    
    with torch.no_grad():
        # TensorRTエンジンで推論を実行
        masked_template_cheese, predicted_mask_cheese = masknet_trt_model(template_tensor, source_tensor)
    
    return masked_template_cheese, predicted_mask_cheese

def create_pcd(xyz):
    n = xyz.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

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
		self.model_pcd = o3d.io.read_point_cloud("TNUTEJN016_100000.pcd")
		#self.model_pcd = o3d.io.read_point_cloud("WMU2LR2020.pcd")
		
		#プロトタイプ
		#self.masknet_load = torch.load("checkpoint/model_weight_epoch300_717_batchsize32.pth")
		self.masknet_load = torch.load("checkpoint/model_weight_epoch300_batchsize32_plane.pth")
		
		
		#撮影対称オブジェクトを使ってデータを作成
		#self.masknet_load = torch.load("checkpoint/pretrained.pth")
		#self.masknet_load = torch.load("checkpoint/first_train.pth")
		#self.masknet_load = torch.load("checkpoint/transformed_learn.pth")
		#self.masknet_load = torch.load("checkpoint/unnoise_transformed_epoch300.pth")
		
		

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
		
	

	def point_cloud_callback(self, msg):
		# Convert ROS PointCloud2 message to Open3D PointCloud
		pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
		points = np.array(list(pc_data))
		#rospy.loginfo(f"Received point cloud with shape: {points.shape}")
		if points.size == 0:
			return
		xyz = points[:, :3]
		try:
			self.measured_pcd = create_pcd(xyz)

			# 点群を保存
			#self.save_point_cloud(self.measured_pcd, "measure")

			self.process_point_cloud()
		except ValueError as e:
			rospy.logerr("ValueError: %s", str(e))
	
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
		len_model_pcd = len(numpy_model_pcd)
		len_measured_pcd = len(numpy_measured_pcd)
		
		
		
		### 計測点群の密度に合わせる ###
		min_bound = np.min(numpy_measured_pcd, axis=0)
		max_bound = np.max(numpy_measured_pcd, axis=0)

		box_width = max_bound[0] - min_bound[0]
		box_height = max_bound[1] - min_bound[1]
		box_depth = max_bound[2] - min_bound[2]
		###print("ボクセルサイズの大きさ", box_width, box_height, box_depth)

		#points_inside = numpy_measured_pcd[(numpy_measured_pcd >= min_bound) & (numpy_measured_pcd <= max_bound)].reshape(-1, 3)

		density = len_measured_pcd / ( box_width * box_height * box_depth )
		###print("密度：", density)   
		weight = 0.7

		voxel_size = weight * (1.0 / density) ** (1/3)
		model_pcd = self.model_pcd.voxel_down_sample(voxel_size) #初期値0.006
		measured_pcd = copy.deepcopy(self.measured_pcd)
		device = torch.device( "cpu")
		#self.masknet_load.to(device)
		template_tensor = torch.tensor(np.array(model_pcd.points), dtype=torch.float32).unsqueeze(0)
		source_tensor = torch.tensor(numpy_measured_pcd, dtype=torch.float32).unsqueeze(0)
		template_tensor = template_tensor.to(device)
		source_tensor = source_tensor.to(device)
		
		
		# 位置合わせ準備を行う（関数を呼び出す）
		registration_model = Registration.Registration()
		print(template_tensor.size())
		print(source_tensor.size())
		masknet_trt_model = convert_model_to_trt(self.masknet_load, template_tensor, source_tensor)

		with torch.no_grad():
			masked_template_cheese, predicted_mask_cheese = run_inference_with_trt(masknet_trt_model, template_tensor, source_tensor)
        
		#マスクの結果保存
		#masked_pcd = o3d.geometry.PointCloud()
		#masked_pcd.points = o3d.utility.Vector3dVector(masked_template_cheese.detach().cpu().numpy()[0])
		#o3d.io.write_point_cloud("mask.pcd", masked_pcd)

		# 提案手法（MaskNet、SVD、ICP）の実行（実際のデータを代入）
		result_cheese = registration_model.register(masked_template_cheese, source_tensor)
		est_T_cheese = result_cheese['est_T']     # est_T：ICPの変換行列
		self.publish_transform(est_T_cheese)
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
		#self.publish_point_cloud(self.model_pcd, masked_pcd)
	
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



def main():
    processor = PointCloudProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
