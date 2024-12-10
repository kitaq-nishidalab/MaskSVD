import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation
from numpy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
import haris_img
import keypoint_fpfh
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
matplotlib.use('Agg')  # 非GUIモードのバックエンドを使用


def rotation_matrix_to_euler_angles(R):
    # Extract angles using trigonometric relations
    roll = np.arctan2(-R[1, 2], R[2, 2])
    pitch = np.arctan2(-(R[0, 2]*np.cos(roll)), R[2, 2])
    yaw = np.arctan2(-R[0, 1], R[0, 0])
    #yaw = np.arctan2(-1, -1)

    return np.array([roll, pitch, yaw])

def coordinate_3d(axes, range_x, range_y, range_z, grid = True):
    axes.set_xlabel("x", fontsize = 14)
    axes.set_ylabel("y", fontsize = 14)
    axes.set_zlabel("z", fontsize = 14)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    axes.set_zlim(range_z[0], range_z[1])
    if grid == True:
        axes.grid()

def visual_vector_3d(axes, loc, vector, color = "red"):
    axes.quiver(loc[0], loc[1], loc[2],
              vector[0], vector[1], vector[2],
              color = color, lw=3)

###################
#重心計算
###################
def calculate_centroid(pointcloud):
    # Calculate the centroid (mean) of the point cloud along each axis
    centroid = np.mean(pointcloud, axis=0)
    return centroid

def translate_to_origin(pointcloud):
    # Calculate the centroid of the point cloud
    centroid = calculate_centroid(pointcloud)

    # Translate the point cloud to move the centroid to the origin
    translated_pointcloud = pointcloud - centroid

    return translated_pointcloud

def adjust_principal_directions(points, principal_vectors, threshold=0.45):
    """
    主成分ベクトルの向きを判定し、逆方向と判定された場合は修正する関数。
    
    Parameters:
        points (np.ndarray): 点群 (N x 3)
        principal_vectors (np.ndarray): 主成分ベクトル (3 x 3)
        threshold (float): 符号が揃っている割合の閾値
        
    Returns:
        np.ndarray: 修正された主成分ベクトル (3 x 3)
        list: 各主成分ベクトルの判定結果（True: 正しい, False: 逆方向）
    """
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    results = []
    adjusted_vectors = principal_vectors.copy()  # 修正用のコピーを作成
    
    for i in range(principal_vectors.shape[1]):  # 各主成分ごとに判定
        principal_vector = principal_vectors[:, i]
        projections = shifted_points @ principal_vector
        # 正の符号の割合を計算
        positive_ratio = np.sum(projections > 0) / len(projections)
        
        if positive_ratio > threshold:
            results.append(True)  # 方向が正しい
        else:
            results.append(False)  # 方向が逆
            adjusted_vectors[:, i] *= -1  # 逆方向の場合は反転
            
    return adjusted_vectors, results
###################
#Registration
###################
# ICP registration module.
class SVD:
	def __init__(self, threshold=0.1, max_iteration=100):
		# threshold: 			Threshold for correspondences. (scalar)
		# max_iterations:		Number of allowed iterations. (scalar)
		self.threshold = threshold
		self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)

	def upsampling(self, pcd, number=1000):
		tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
		# 点群をメッシュ化（Convex Hullを使用）
		mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.01, tetra_mesh=tetra_mesh, pt_map=pt_map)
		# メッシュをサンプルして、点群をアップサンプリング
		pcd_upsampled = mesh.sample_points_uniformly(number_of_points=number)  # 1000点にアップサンプリング

		return pcd_upsampled

	# Preprocess template, source point clouds.
	def preprocess(self, template, source, target):
		if self.is_tensor: template, source = template.detach().cpu().numpy(), source.detach().cpu().numpy()	# Convert to ndarray if tensors.

		if len(template.shape) > 2: 						# Reduce dimension to [N, 3]
			template, source = template[0], source[0]

		#model = model.detach().cpu().numpy()[0]

		#model_pcd = o3d.geometry.PointCloud()
		template_pcd = o3d.geometry.PointCloud()
		source_pcd = o3d.geometry.PointCloud()

		#model_pcd.points = o3d.utility.Vector3dVector(model)
		template_pcd.points = o3d.utility.Vector3dVector(template)
		source_pcd.points = o3d.utility.Vector3dVector(source)

		#print("template_shape:", len(np.array(template_pcd.points)))
		#print("source_shape:", len(np.array(source_pcd.points)))

		#template_pcd = self.upsampling(template_pcd, len(np.array(template_pcd.points)) * 5)
		#source_pcd = self.upsampling(source_pcd, len(np.array(source_pcd.points)) * 5)
		
		# Find mean of template & source.
		self.template_mean = np.mean(np.array(template_pcd.points), axis=0, keepdims=True)
		self.source_mean = np.mean(np.array(source_pcd.points), axis=0, keepdims=True)
		#model_mean = np.mean(np.array(model_pcd.points), axis=0, keepdims=True)

		# Convert to open3d point clouds.
		template_ = o3d.geometry.PointCloud()
		source_ = o3d.geometry.PointCloud()
		#model_ = o3d.geometry.PointCloud()

		# Subtract respective mean from each point cloud.
		template_.points = o3d.utility.Vector3dVector(np.array(template_pcd.points) - self.template_mean)
		source_.points = o3d.utility.Vector3dVector(np.array(source_pcd.points) - self.source_mean)
		#model_.points = o3d.utility.Vector3dVector(np.array(model_pcd.points) - model_mean)

		#template_.paint_uniform_color([0, 1, 0])
		#source_.paint_uniform_color([0, 0, 1])
		#o3d.visualization.draw_geometries([source_, template_])

		### Voxel downsamplig
		#if target=="t":
		#	voxel_size = 0.004 #0.02(Haris)
		#	template_ = template_.voxel_down_sample(voxel_size)
		#	source_ = source_.voxel_down_sample(voxel_size)
		#voxel_size = 0.004 #0.02(Haris)
		#template_ = template_.voxel_down_sample(voxel_size)
		#source_ = source_.voxel_down_sample(voxel_size)
		### Haris3D
		#template_ = haris_img.haris3d(template_, blockSize=3, ksize=3, k=0.04, threshold_factor=0.5)
		#source_ = haris_img.haris3d(source_, blockSize=3, ksize=3, k=0.04, threshold_factor=0.5)
		#model_ = keypoint_fpfh.keypoint_fpfh(model_pcd, radius=0.55)
		### FPFH
		#template_ = keypoint_fpfh.keypoint_fpfh(template_, radius=0.5, threthold=0.3)
		#source_ = keypo/tf_staticint_fpfh.keypoint_fpfh(source_, radius=0.5, threthold=0.3)
		### ISS
		#template_ = o3d.geometry.keypoint.compute_iss_keypoints(template_,salient_radius=0.008,non_max_radius=0.008, gamma_21=0.975,gamma_32=0.975,min_neighbors=3)
		#source_ = o3d.geometry.keypoint.compute_iss_keypoints(source_,salient_radius=0.008,non_max_radius=0.008, gamma_21=0.975,gamma_32=0.975,min_neighbors=3)
		### Carvature
		#template_ = carvature(template_)
		#source_ = carvature(source_)

		return template_, source_

	# Postprocess on transformation matrix.
	def postprocess(self, res):
		# Way to deal with mean substraction
		# Pt = R*Ps + t 								original data (1)
		# Pt - Ptm = R'*[Ps - Psm] + t' 				mean subtracted from template and source.
		# Pt = R'*Ps + t' - R'*Psm + Ptm 				rearrange the equation (2)
		# From eq. 1 and eq. 2,
		# R = R' 	&	t = t' - R'*Psm + Ptm			(3)
		est_R = res.transformation[0:3, 0:3]		# ICP's rotation matrix (source -> template)
		t_ = np.array(res.transformation[0:3, 3]).reshape(1, -1)			# ICP's translation vector (source -> template)
		est_T = np.array(res.transformation)								# ICP's transformation matrix (source -> template)
		est_t = np.matmul(est_R, -self.source_mean.T).T + t_ + self.template_mean[0] 	# update predicted translation according to eq. 3
		est_T[0:3, 3] = est_t
		#print(self.source_mean)
		return est_R, est_t, est_T

	# Convert result to pytorch tensors.
	@staticmethod
	def convert2tensor(result):
		if torch.cuda.is_available(): device = 'cuda'
		else: device = 'cpu'
		result['est_R']=torch.tensor(result['est_R']).to(device).float().view(-1, 3, 3) 		# Rotation matrix [B, 3, 3] (source -> template)
		result['est_t']=torch.tensor(result['est_t']).to(device).float().view(-1, 1, 3)			# Translation vector [B, 1, 3] (source -> template)
		result['est_T']=torch.tensor(result['est_T']).to(device).float().view(-1, 4, 4)			# Transformation matrix [B, 4, 4] (source -> template)
		return result

	# icp registration.
	def __call__(self, template, source, target):
		self.is_tensor = torch.is_tensor(template)
		
		######## tensorからo3d #########
		template, source= self.preprocess(template, source, target)
		#source.paint_uniform_color([0, 1, 0])
		#template.paint_uniform_color([0, 0, 1])
		#o3d.visualization.draw_geometries([source, template])
		
		########################################################
		### Method using SVD（概略マッチング） ###
		########################################################
		######## マスクテンプレの主成分分析 #########
		T_trans = np.array(template.points)
		T_trans -= T_trans.mean(axis=0)
		# PCA実行
		T_pca = PCA(n_components=3)  # 3成分を取得するために n_components=3
		T_pca.fit(T_trans)
		# 固有値、固有ベクトルの取得
		#T_W = T_pca.explained_variance_  # 固有値
		T_V_pca = T_pca.components_.T  # 固有ベクトル (sklearnでは転置されている)
		######### ソースの主成分分析 ###########
		S_trans = np.array(source.points)
		S_trans -= S_trans.mean(axis=0)
		# PCA実行
		S_pca = PCA(n_components=3)  # 3成分を取得するために n_components=3
		S_pca.fit(S_trans)
		# 固有値、固有ベクトルの取得
		#S_W = S_pca.explained_variance_  # 固有値
		S_V_pca = S_pca.components_.T  # 固有ベクトル (sklearnでは転置されている)#
		########## 鏡像マッチング対策 ############
		#print("\n", np.linalg.det(S_V_pca @ T_V_pca.T))
		#print(np.linalg.det(S_V_pca @ T_V_pca.T))
		if np.linalg.det(S_V_pca @ T_V_pca.T) < 0:
			if target=="t":
				K = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]])     # T joint pipe
			elif target=="l":
				K = np.array([[1, 0, 0],[0, -1, 0],[0, 0, 1]])     # L joint pipe
			R = S_V_pca @ K @ T_V_pca.T
		else:
			R = S_V_pca @ T_V_pca.T
		#print(np.linalg.det(R))
		###print(R)
		###転置処理###
		R = R.T
		""""
		#############################################################################################################
		#############################################################################################################
		### 原点を表示 ###
		numpy_o_geo = np.array([[0, 0, 0]])
		o_geo = o3d.geometry.PointCloud()
		o_geo.points = o3d.utility.Vector3dVector(numpy_o_geo)
		o_geo.paint_uniform_color([1, 0, 0])
		### x軸の表示 ###
		numpy_ax_x = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [0.05, 0, 0], [0.06, 0, 0], [0.07, 0, 0], [0.08, 0, 0], [0.09, 0, 0], [0.1, 0, 0]])
		ax_x = o3d.geometry.PointCloud()
		ax_x.points = o3d.utility.Vector3dVector(numpy_ax_x)
		ax_x.paint_uniform_color([1, 0, 0])
		### y軸の表示 ###Screenshot from 2024-02-21 18-12-00
		numpy_ax_y = np.array([[0, 0.01, 0], [0, 0.02, 0], [0, 0.03, 0], [0, 0.04, 0], [0, 0.05, 0], [0, 0.06, 0], [0, 0.07, 0], [0, 0.08, 0], [0, 0.09, 0], [0, 0.1, 0]])
		ax_y = o3d.geometry.PointCloud()
		ax_y.points = o3d.utility.Vector3dVector(numpy_ax_y)
		ax_y.paint_uniform_color([1, 0, 0])
		### z軸の表示 ###
		numpy_ax_z = np.array([[0, 0, 0.01], [0, 0, 0.02], [0, 0, 0.03], [0, 0, 0.04], [0, 0, 0.05], [0, 0, 0.06], [0, 0, 0.07], [0, 0, 0.08], [0, 0, 0.09], [0, 0, 0.1]])
		ax_z = o3d.geometry.PointCloud()
		ax_z.points = o3d.utility.Vector3dVector(numpy_ax_z)
		ax_z.paint_uniform_color([1, 0, 0])
				
		pca_ut0_vector = T_V_pca[:, 0].reshape([3,1])
		pca_ut1_vector = T_V_pca[:, 1].reshape([3,1])
		pca_ut2_vector = T_V_pca[:, 2].reshape([3,1])
		pca_us0_vector = S_V_pca[:, 0].reshape([3,1])
		pca_us1_vector = S_V_pca[:, 1].reshape([3,1])
		pca_us2_vector = S_V_pca[:, 2].reshape([3,1])
		### 回転後ソースの主成分分析 ###
		ans = (R @ np.array(source.points).T).T
		ans_cov = ans.T @ ans
		# 固有値、固有ベクトルの取得
		ans_W, ans_V_pca = np.linalg.eig(ans_cov)
		# Sort eigenvectors with eigenvalues
		ans_index = ans_W.argsort()[::-1]
		ans_W = ans_W[ans_index]
		ans_V_pca = ans_V_pca[:, ans_index]

		ans0_vector = ans_V_pca[:, 0].reshape([3,1])
		ans1_vector = ans_V_pca[:, 1].reshape([3,1])
		ans2_vector = ans_V_pca[:, 2].reshape([3,1])
	

		#model_cov = np.array(model.points).T @ np.array(model.points)
		# 固有値、固有ベクトルの取得
		#model_W, model_V_pca = np.linalg.eig(model_cov)
		# Sort eigenvectors with eigenvalues
		#model_index = model_W.argsort()[::-1]
		#model_W = model_W[model_index]
		#model_V_pca = model_V_pca[:, model_index]

		#model0_vector = model_V_pca[:, 0].reshape([3,1])
		#model1_vector = model_V_pca[:, 1].reshape([3,1])
		#model2_vector = model_V_pca[:, 2].reshape([3,1])

		# 3Dベクトルを定義
		#ans_v1 = np.array(ans0_vector / 40)
		#ans_v2 = np.array(ans1_vector / 40)
		#ans_v3 = np.array(ans2_vector / 40)
		#model_v1 = np.array(model0_vector / 40)
		#model_v2 = np.array(model1_vector / 40)
		#model_v3 = np.array(model2_vector / 40)
		pca_v1 = np.array(pca_ut0_vector / 40)
		pca_v2 = np.array(pca_ut1_vector / 40)
		pca_v3 = np.array(pca_ut2_vector / 40)
		pca_v4 = np.array(pca_us0_vector / 40)
		pca_v5 = np.array(pca_us1_vector / 40)
		pca_v6 = np.array(pca_us2_vector / 40)
		### マスクテンプレの主成分ベクトルと点群を表示 ###
		fig = plt.figure(figsize = (6, 6))
		ax = fig.add_subplot(111, projection='3d')
		# 3D座標を設定
		coordinate_3d(ax, [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], grid = True)
		# 始点を設定
		o = [0, 0, 0]
		visual_vector_3d(ax, o, pca_v1, "red")
		visual_vector_3d(ax, o, pca_v2, "green")
		visual_vector_3d(ax, o, pca_v3, "blue")
		ax.scatter(np.asarray(template.points)[:,0], np.asarray(template.points)[:,1], np.asarray(template.points)[:,2], s = 3, c = "red")
		plt.savefig('/home/nishidalab0/vision_ws/src/MaskSVD/template_output_image.png')
		
		### ソースの主成分ベクトルと点群を表示 ###
		fig = plt.figure(figsize = (6, 6))
		ax = fig.add_subplot(111, projection='3d')
		# 3D座標を設定
		coordinate_3d(ax, [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], grid = True)
		# 始点を設定
		o = [0, 0, 0]
		visual_vector_3d(ax, o, pca_v4, "orange")
		visual_vector_3d(ax, o, pca_v5, "lime")
		visual_vector_3d(ax, o, pca_v6, "cyan")
		ax.scatter(np.asarray(source.points)[:,0], np.asarray(source.points)[:,1], np.asarray(source.points)[:,2], s = 3, c = "lime")
		
		fig.tight_layout() 
		plt.savefig('/home/nishidalab0/vision_ws/src/MaskSVD/source_output_image.png')
		###########################################################################################################
		###########################################################################################################
		"""
		transformation = np.eye(4)
		for i in range(3):
			for j in range(3):
				transformation[i][j] = R[i][j]
		#print(transformation)
		########################################################
		########################################################
		
		### ICPアルゴリズム（精密マッチング） ###
		res = o3d.pipelines.registration.registration_icp(source, template, self.threshold, transformation, criteria=self.criteria)	# icp registration in open3d.
		#print("transformation:\n", res.transformation)
		
		est_R, est_t, est_T = self.postprocess(res)
		result = {'est_R': est_R,
		          'est_t': est_t,
		          'est_T': est_T}
		
		
		if self.is_tensor: result = self.convert2tensor(result)
		return result

# Define Registration Algorithm.
def registration_algorithm():
  
  reg_algorithm = SVD()
  
  return reg_algorithm


# Register template and source pairs.
class Registration:
	def __init__(self):
		self.reg_algorithm = registration_algorithm()

	@staticmethod
	def pc2points(data):
		if len(data.shape) == 3:
			return data[:, :, :3]
		elif len(data.shape) == 2:
			return data[:, :3]

	def register(self, template, source, target):
		# template, source: 		Point Cloud [B, N, 3] (torch tensor)
		result = self.reg_algorithm(template, source, target)
		return result

def pc2open3d(data):
	if torch.is_tensor(data): data = data.detach().cpu().numpy()
	if len(data.shape) == 2:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data)
		return pc
	else:
		print("Error in the shape of data given to Open3D!, Shape is ", data.shape)


def display_results_sample(template, source, est_T, masked_template):
  transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]   # ※matmul：行列の積　　第一項：回転、第二項：平行移動、重心移動分も含まれる transformed =  R' * source + t
  #transformed_source = source + est_T[0:3, 3]
  #print(est_T[0:3, 0:3])
  #transformed_source = np.matmul(est_T[0:3, 0:3], (source - np.mean(source, axis=0, keepdims=True)).T).T
  #transformed_source = transformed_source + np.mean(source, axis=0, keepdims=True)
  #transformed_source = transformed_source + est_T[0:3, 3].T
  
  numpy_source_t = source + est_T[0:3, 3]
  source_t = o3d.geometry.PointCloud()
  source_t.points = o3d.utility.Vector3dVector(numpy_source_t)
  source_t.paint_uniform_color([0, 1, 0])
  
  ### x軸の表示 ###
  numpy_ax_x = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [0.05, 0, 0], [0.06, 0, 0], [0.07, 0, 0], [0.08, 0, 0], [0.09, 0, 0], [0.1, 0, 0]])
  ax_x = o3d.geometry.PointCloud()
  ax_x.points = o3d.utility.Vector3dVector(numpy_ax_x)
  ax_x.paint_uniform_color([1/3, 1/3, 1/3])
		
  ### y軸の表示 ###
  numpy_ax_y = np.array([[0, 0.01, 0], [0, 0.02, 0], [0, 0.03, 0], [0, 0.04, 0], [0, 0.05, 0], [0, 0.06, 0], [0, 0.07, 0], [0, 0.08, 0], [0, 0.09, 0], [0, 0.1, 0]])
  ax_y = o3d.geometry.PointCloud()
  ax_y.points = o3d.utility.Vector3dVector(numpy_ax_y)
  ax_y.paint_uniform_color([1/3, 1/3, 1/3])
		
  ### z軸の表示 ###
  numpy_ax_z = np.array([[0, 0, 0.01], [0, 0, 0.02], [0, 0, 0.03], [0, 0, 0.04], [0, 0, 0.05], [0, 0, 0.06], [0, 0, 0.07], [0, 0, 0.08], [0, 0, 0.09], [0, 0, 0.1]])
  ax_z = o3d.geometry.PointCloud()
  ax_z.points = o3d.utility.Vector3dVector(numpy_ax_z)
  ax_z.paint_uniform_color([1/3, 1/3, 1/3])
  
  ### 原点を表示 ###
  numpy_o = np.array([[0, 0, 0]])
  o = o3d.geometry.PointCloud()
  o.points = o3d.utility.Vector3dVector(numpy_o)
  o.paint_uniform_color([1, 0, 0])
  
  template = pc2open3d(template)
  source = pc2open3d(source)
  #transformed_source = copy.deepcopy(source)
  #transformed_source.transform(est_T)
  transformed_source = pc2open3d(np.array(transformed_source))
  masked_template = pc2open3d(masked_template)

  template.paint_uniform_color([1, 0, 0])
  source.paint_uniform_color([0, 1, 0])
  transformed_source.paint_uniform_color([0, 1, 0])
  masked_template.paint_uniform_color([0, 0, 1])

  #o3d.visualization.draw_geometries([template])                                    # テンプレ
  #o3d.visualization.draw_geometries([masked_template, source, ans_source])          # マスクテンプレ、ソース、正解ソース、原点
  #o3d.visualization.draw_geometries([template, ans_source])          # マスクテンプレ、ソース、正解ソース、原点
  #o3d.visualization.draw_geometries([masked_template, source, source_t, o, ax_x, ax_y, ax_z])
  #o3d.visualization.draw_geometries([masked_template, source, transformed_source])  # マスクテンプレ、ソース、変換後ソース
  
  #o3d.visualization.draw_geometries([template, source, transformed_source])        # テンプレ、ソース、変換後ソース
  
  o3d.visualization.draw_geometries([template, masked_template, source])           # テンプレ、マスクテンプレ、ソース
  #o3d.visualization.draw_geometries([template, source])                            # テンプレ、ソース
  #o3d.visualization.draw_geometries([masked_template, source])                     # マスクテンプレ、ソース
  ###masked_template.paint_uniform_color([0, 1, 0])
  ###o3d.visualization.draw_geometries([masked_template, source])                     # マスクテンプレ（green）
