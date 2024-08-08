import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation
from numpy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D



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

	# Preprocess template, source point clouds.
	def preprocess(self, template, source):
		if self.is_tensor: template, source = template.detach().cpu().numpy(), source.detach().cpu().numpy()	# Convert to ndarray if tensors.

		if len(template.shape) > 2: 						# Reduce dimension to [N, 3]
			template, source = template[0], source[0]

		# Find mean of template & source.
		self.template_mean = np.mean(template, axis=0, keepdims=True)
		self.source_mean = np.mean(source, axis=0, keepdims=True)

		# Convert to open3d point clouds.
		template_ = o3d.geometry.PointCloud()
		source_ = o3d.geometry.PointCloud()

		# Subtract respective mean from each point cloud.
		template_.points = o3d.utility.Vector3dVector(template - self.template_mean)
		source_.points = o3d.utility.Vector3dVector(source - self.source_mean)
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
	
	def postprocess_nonicp(self, res):
		# Way to deal with mean substraction
		# Pt = R*Ps + t 								original data (1)
		# Pt - Ptm = R'*[Ps - Psm] + t' 				mean subtracted from template and source.
		# Pt = R'*Ps + t' - R'*Psm + Ptm 				rearrange the equation (2)
		# From eq. 1 and eq. 2,
		# R = R' 	&	t = t' - R'*Psm + Ptm			(3)
		est_R = res[0:3, 0:3]		# ICP's rotation matrix (source -> template)
		t_ = np.array(res[0:3, 3]).reshape(1, -1)			# ICP's translation vector (source -> template)
		est_T = res							# ICP's transformation matrix (source -> template)
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
	def __call__(self, template, source, pattern):
		self.is_tensor = torch.is_tensor(template)
		
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
		
		### 重心移動の計算 ###
		source_a = source.detach().cpu().numpy()
		source_a = source_a[0]
		jusin_mae = np.mean(source_a, axis=0)
		
		# tensorからo3d
		template, source = self.preprocess(template, source)
		
		"""
		jusin_ato = np.mean(np.array(source.points), axis=0)
		jusin_ido = jusin_ato - jusin_mae
		print("\n移動前のソース点群の重心：\n", jusin_mae)
		print("\n移動後のソース点群の重心：\n", jusin_ato)
		print("\n重心移動：", jusin_ido)
		"""
		source.paint_uniform_color([0, 1, 0])
		template.paint_uniform_color([0, 0, 1])
		###o3d.visualization.draw_geometries([source, template])
		
		template_vis = template
		source_vis = source
		template_copy = copy.deepcopy(template)
		source_copy = copy.deepcopy(source)
		
		### ダウンサンプリング ###
		if pattern == "A":
			voxel_size = 0.008     # 0度
		elif pattern == "45":
			voxel_size = 0.011     # 45度
		elif pattern == "B":
			voxel_size = 0.015     # 90度
		elif pattern == "135":
			voxel_size = 0.015     # 180度
		elif pattern == "D":
			voxel_size = 0.011    # L90度
		elif pattern == "C":
			voxel_size = 0.011    # L180度
		
		template = template.voxel_down_sample(voxel_size)
		source = source.voxel_down_sample(voxel_size)
		
		### マスクテンプレの主成分分析 ###
		T_trans = np.array(template.points)
		#print(np.shape(T_trans)[0])
		#T_cov = T_trans.T @ T_trans / (np.shape(T_trans)[0] - 1)
		T_cov = T_trans.T @ T_trans
		# 固有値、固有ベクトルの取得
		T_W, T_V_pca = np.linalg.eig(T_cov)
		# Sort eigenvectors with eigenvalues
		T_index = T_W.argsort()[::-1]
		T_W = T_W[T_index]
		T_V_pca = T_V_pca[:, T_index]
		
		### ソースの主成分分析 ###
		S_trans = np.array(source.points)
		#print(np.shape(S_trans)[0])
		#S_cov = S_trans.T @ S_trans / (np.shape(S_trans)[0] - 1)
		S_cov = S_trans.T @ S_trans
		# 固有値、固有ベクトルの取得
		S_W, S_V_pca = np.linalg.eig(S_cov)
		# Sort eignvectors with eignvalues
		S_index = S_W.argsort()[::-1]
		S_W = S_W[S_index]
		S_V_pca = S_V_pca[:, S_index]
		
		#print("\nマスクテンプレ側の固有値：", T_W)
		#print('PCA_Template_左特異ベクトル：')
		#print('左特異ベクトル（マスク後テンプレ）：')
		#print(T_V_pca)
		#print("\nソース側の固有値：", S_W)
		#print('PCA_Source_左特異ベクトル：')
		#print('左特異ベクトル（ソース）：')
		#print(S_V_pca)
		
		pca_ut0_vector = T_V_pca[:, 0].reshape([3,1])
		pca_ut1_vector = T_V_pca[:, 1].reshape([3,1])
		pca_ut2_vector = T_V_pca[:, 2].reshape([3,1])
		pca_us0_vector = S_V_pca[:, 0].reshape([3,1])
		pca_us1_vector = S_V_pca[:, 1].reshape([3,1])
		pca_us2_vector = S_V_pca[:, 2].reshape([3,1])
		
		#R = pca_us0_vector @ pca_us0_vector.T + pca_us1_vector @ pca_us1_vector.T + pca_us2_vector @ pca_ut2_vector.T
		#R = pca_us0_vector @ pca_ut0_vector.T + pca_us1_vector @ pca_ut1_vector.T + pca_us2_vector @ pca_ut2_vector.T
		###print("\n", np.linalg.det(S_V_pca @ T_V_pca.T))
		###print(S_V_pca @ T_V_pca.T)
		if np.linalg.det(S_V_pca @ T_V_pca.T) < 0:
			K = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]])     # 0度、45度、90度
			R = S_V_pca @ K @ T_V_pca.T
		else:
			R = S_V_pca @ T_V_pca.T
		###print(np.linalg.det(R))
		###print(R)
		###転置処理###
		R = R.T
		
		#############################################################################################################
		#############################################################################################################
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
		# 3Dベクトルを定義
		ans_v1 = np.array(ans0_vector / 40)
		ans_v2 = np.array(ans1_vector / 40)
		ans_v3 = np.array(ans2_vector / 40)
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
		visual_vector_3d(ax, o, pca_v2, "blue")
		visual_vector_3d(ax, o, pca_v3, "green")
		ax.scatter(np.asarray(template.points)[:,0], np.asarray(template.points)[:,1], np.asarray(template.points)[:,2], s = 3, c = "red")
		### ソースの主成分ベクトルと点群を表示 ###
		fig = plt.figure(figsize = (6, 6))
		ax = fig.add_subplot(111, projection='3d')
		# 3D座標を設定
		coordinate_3d(ax, [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], grid = True)
		# 始点を設定
		o = [0, 0, 0]
		visual_vector_3d(ax, o, pca_v4, "orange")
		visual_vector_3d(ax, o, pca_v5, "cyan")
		visual_vector_3d(ax, o, pca_v6, "lime")
		ax.scatter(np.asarray(source.points)[:,0], np.asarray(source.points)[:,1], np.asarray(source.points)[:,2], s = 3, c = "lime")
		### SVD後ソース点群を表示 ###
		fig = plt.figure(figsize = (6, 6))
		ax = fig.add_subplot(111, projection='3d')
		# 3D座標を設定
		coordinate_3d(ax, [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], grid = True)
		# 始点を設定
		o = [0, 0, 0]
		visual_vector_3d(ax, o, ans_v1, "darkred")
		visual_vector_3d(ax, o, ans_v2, "darkblue")
		visual_vector_3d(ax, o, ans_v3, "darkgreen")
		ax.scatter(ans[:, 0], ans[:, 1], ans[:, 2], s = 3, c = "darkgreen")
		fig.tight_layout() 
		#plt.show()
		###########################################################################################################
		###########################################################################################################
		
		### SVDアルゴリズム（概略マッチング） ###
		transformation = np.eye(4)
		for i in range(3):
			for j in range(3):
				transformation[i][j] = R[i][j]
		#print(transformation)
		
		
		### ICPアルゴリズム（精密マッチング） ###
		res = o3d.pipelines.registration.registration_icp(source, template, self.threshold, transformation, criteria=self.criteria)	# icp registration in open3d.
		#print("transformation:\n", res.transformation)
		
		est_R, est_t, est_T = self.postprocess(res)
		result = {'est_R': est_R,
		          'est_t': est_t,
		          'est_T': est_T}
		
		#print(est_T)
		#print(est_R)      # 回転移動(3×3)
		#print(est_t)      # 平行移動
		#source_temp = copy.deepcopy(source)
		#target_temp = copy.deepcopy(template)
		#source_temp.paint_uniform_color([0, 1, 0])
		#target_temp.paint_uniform_color([1, 0, 0])
		#source_temp.transform(res.transformation)
		#source_temp.transform(transformation)
		#o3d.visualization.draw_geometries([source_temp, target_temp, o_geo])
		###print("\nSVD+ICPの変換行列：\n", est_T)      # 変換行列全体(4×4)、重心移動分も含まれる
		#print('\nfitness:', float(res.fitness))
		#print("source:", np.shape(np.array(source.points)))
		#print("template:", np.shape(np.array(template.points)))
		#print('\ninlier_rmse:', float(res.inlier_rmse))
		
		if self.is_tensor: result = self.convert2tensor(result)
		return result

# Define Registration Algorithm.
def registration_algorithm():
  
  reg_algorithm = SVD()
  
  return reg_algorithm


# Register template and source pairs.
class Registration:
	def __init__(self, pattern = "A"):
		#self.reg_algorithm = reg_algorithm
		#self.is_rpmnet = True if self.reg_algorithm == 'rpmnet' else False
		#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.reg_algorithm = registration_algorithm()
		self.pattern = pattern

	@staticmethod
	def pc2points(data):
		if len(data.shape) == 3:
			return data[:, :, :3]
		elif len(data.shape) == 2:
			return data[:, :3]

	def register(self, template, source):
		# template, source: 		Point Cloud [B, N, 3] (torch tensor)

		# No need to use normals. Only use normals for RPM-Net.
		#if not self.is_rpmnet == 'rpmnet':
		#	template, source = self.pc2points(template), self.pc2points(source)

		result = self.reg_algorithm(template, source, self.pattern)
		return result

def pc2open3d(data):
	if torch.is_tensor(data): data = data.detach().cpu().numpy()
	if len(data.shape) == 2:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data)
		return pc
	else:
		print("Error in the shape of data given to Open3D!, Shape is ", data.shape)


def display_results_sample(template, source, est_T, masked_template, pattern):
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
  
  ### 正解を定義 ###
  if pattern == "A":
  	## 0度 ##
  	ans_theta_x = np.radians(0)
  	ans_theta_y = np.radians(1)
  	ans_theta_z = np.radians(176)
  elif pattern == "45":
  	## 90度 ##
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(45)
  elif pattern == "B":
  	## 90度 ##
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(-90)
  elif pattern == "135":
  	## 90度 ##
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(-45)
  elif pattern == "D":
  	## 90度 ##
  	ans_theta_x = np.radians(-3)
  	ans_theta_y = np.radians(-1)
  	ans_theta_z = np.radians(-85)
  elif pattern == "C":
  	## 90度 ##
  	ans_theta_x = np.radians(-12)
  	ans_theta_y = np.radians(-2)
  	ans_theta_z = np.radians(176)
  # x軸方向に回転
  R_x = np.array(
         [[1, 0, 0], 
         [0, np.cos(ans_theta_x), -np.sin(ans_theta_x)], 
         [0, np.sin(ans_theta_x), np.cos(ans_theta_x)]])
  # y軸方向に回転
  R_y = np.array(
         [[np.cos(ans_theta_y), 0, np.sin(ans_theta_y)], 
         [0, 1, 0], 
         [-np.sin(ans_theta_y), 0, np.cos(ans_theta_y)]])
  # z軸方向に回転       
  R_z = np.array(
         [[np.cos(ans_theta_z), -np.sin(ans_theta_z), 0], 
         [np.sin(ans_theta_z), np.cos(ans_theta_z), 0], 
         [0, 0, 1]])
  # 回転行列を計算
  ans_R = R_x @ R_y @ R_z
  #print("ans_R:\n", ans_R)
  #print("est_R:\n", est_T[0:3, 0:3])
  
  # 平行移動
  ans_t_ = [0, 0.008, -0.011]
  if pattern == "D":
  
  	ans_t_ = [0.0055, 0.008, -0.011]
  if pattern == "C":
  
  	ans_t_ = [0.006, 0.008, -0.011]
    
  # 重心移動も含めた変換を行う
  ans_t = np.matmul(ans_R, -np.mean(source, axis=0).T).T + ans_t_
  numpy_ans_source = np.matmul(ans_R, source.T).T + ans_t
  ans_source = o3d.geometry.PointCloud()
  ans_source.points = o3d.utility.Vector3dVector(numpy_ans_source)
  
  ### 回転移動の差分 ###
  euler_angles = rotation_matrix_to_euler_angles(est_T[0:3, 0:3])
  rotation_angle_x = np.degrees(euler_angles[0])
  rotation_angle_y = np.degrees(euler_angles[1])
  rotation_angle_z = np.degrees(euler_angles[2])
  #print("\nRotation angle around x-axis:", rotation_angle_x, "degrees")
  #print("Rotation angle around y-axis:", rotation_angle_y, "degrees")
  #print("Rotation angle around z-axis:", rotation_angle_z, "degrees")
  ###print("\n回転移動の差：")
  diff_R_x = rotation_angle_x - np.degrees(ans_theta_x)
  diff_R_y = rotation_angle_y - np.degrees(ans_theta_y)
  diff_R_z = rotation_angle_z - np.degrees(ans_theta_z)
  ###print("x軸方向　", abs(diff_R_x), " ", "y軸方向　", abs(diff_R_y), " ", "z軸方向　", abs(diff_R_z))
  global diff_R
  diff_R = np.linalg.norm([diff_R_x, diff_R_y, diff_R_z])
  #print("\n回転移動の差（L2ノルム）：", diff_R)
  
  ### 平行移動の差分 ###
  global diff_t
  #print("est_T[0:3, 3]")
  #print(est_T[0:3, 3])
  #print("ans_t ")
  #print(ans_t)
  diff_t = np.linalg.norm(est_T[0:3, 3] - ans_t)
  #print("平行移動の差（L2ノルム）：", diff_t, "\n")
  
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
  ans_source.paint_uniform_color([1/3, 1/3, 1/3])

  #o3d.visualization.draw_geometries([template])                                    # テンプレ
  #o3d.visualization.draw_geometries([masked_template, source, ans_source])          # マスクテンプレ、ソース、正解ソース、原点
  #o3d.visualization.draw_geometries([template, ans_source])          # マスクテンプレ、ソース、正解ソース、原点
  #o3d.visualization.draw_geometries([masked_template, source, source_t, o, ax_x, ax_y, ax_z])
  #o3d.visualization.draw_geometries([masked_template, source, transformed_source])  # マスクテンプレ、ソース、変換後ソース
  o3d.visualization.draw_geometries([template, source, transformed_source])        # テンプレ、ソース、変換後ソース
  #o3d.visualization.draw_geometries([template, masked_template, source])           # テンプレ、マスクテンプレ、ソース
  #o3d.visualization.draw_geometries([template, source])                            # テンプレ、ソース
  #o3d.visualization.draw_geometries([masked_template, source])                     # マスクテンプレ、ソース
  ###masked_template.paint_uniform_color([0, 1, 0])
  ###o3d.visualization.draw_geometries([masked_template, source])                     # マスクテンプレ（green）
