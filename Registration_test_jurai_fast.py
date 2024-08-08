import numpy as np
import torch
import open3d as o3d
import Global_optimizer_fast
    
def rotation_matrix_to_euler_angles(R):
    # Extract angles using trigonometric relations
    roll = np.arctan2(-R[1, 2], R[2, 2])
    pitch = np.arctan2(-(R[0, 2]*np.cos(roll)), R[2, 2])
    yaw = np.arctan2(-R[0, 1], R[0, 0])

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
class ICP:
	def __init__(self, threshold=0.01, max_iteration=100):
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

		est_R = np.array(res.transformation[0:3, 0:3]) 						# ICP's rotation matrix (source -> template)
		t_ = np.array(res.transformation[0:3, 3]).reshape(1, -1)			# ICP's translation vector (source -> template)
		est_T = np.array(res.transformation)								# ICP's transformation matrix (source -> template)
		est_t = np.matmul(est_R, -self.source_mean.T).T + t_ + self.template_mean[0] 	# update predicted translation according to eq. 3
		#est_t = -self.source_mean[0] + t_ + self.template_mean[0]
		est_T[0:3, 3] = est_t
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

	# FGR + icp registration.
	def __call__(self, template, source, pattern):
		print(pattern)
		self.is_tensor = torch.is_tensor(template)
		
		### 原点を表示 ###
		numpy_o = np.array([[0, 0, 0]])
		o = o3d.geometry.PointCloud()
		o.points = o3d.utility.Vector3dVector(numpy_o)
		o.paint_uniform_color([0, 0, 1])
		
		### x軸の表示 ###
		numpy_ax_x = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [0.05, 0, 0], [0.06, 0, 0], [0.07, 0, 0], [0.08, 0, 0], [0.09, 0, 0], [0.1, 0, 0]])
		ax_x = o3d.geometry.PointCloud()
		ax_x.points = o3d.utility.Vector3dVector(numpy_ax_x)
		ax_x.paint_uniform_color([0, 0, 1])
		
		### y軸の表示 ###
		numpy_ax_y = np.array([[0, 0.01, 0], [0, 0.02, 0], [0, 0.03, 0], [0, 0.04, 0], [0, 0.05, 0], [0, 0.06, 0], [0, 0.07, 0], [0, 0.08, 0], [0, 0.09, 0], [0, 0.1, 0]])
		ax_y = o3d.geometry.PointCloud()
		ax_y.points = o3d.utility.Vector3dVector(numpy_ax_y)
		ax_y.paint_uniform_color([0, 0, 1])
		
		### z軸の表示 ###
		numpy_ax_z = np.array([[0, 0, 0.01], [0, 0, 0.02], [0, 0, 0.03], [0, 0, 0.04], [0, 0, 0.05], [0, 0, 0.06], [0, 0, 0.07], [0, 0, 0.08], [0, 0, 0.09], [0, 0, 0.1]])
		ax_z = o3d.geometry.PointCloud()
		ax_z.points = o3d.utility.Vector3dVector(numpy_ax_z)
		ax_z.paint_uniform_color([0, 0, 1])
  
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
		template.paint_uniform_color([1, 0, 0])
		###o3d.visualization.draw_geometries([source, template])
		
		template_vis = template
		source_vis = source
		
		if pattern == "A":
			voxel_size = 0.01  # 0度  0.009
		#elif pattern == 45:
		#	voxel_size = 0.011     # 45度
		elif pattern == "B":
			voxel_size = 0.011     # 90度
		#elif pattern == 135:
		#	voxel_size = 0.015     # 135度
		elif pattern == "D":
			voxel_size = 0.03    # 90度
		elif pattern == "C":
			voxel_size = 0.02   # 180度 0.08, 0.0321, 0.0711, 0.0721, 0.0731, 0.0741
		
		### 前処理 ###
		source, template, source_down, template_down, source_fpfh, template_fpfh = Global_optimizer_fast.prepare_dataset(voxel_size, template, source, pattern)
		#source, template, source_down, template_down, source_fpfh, template_fpfh = Global_optimizer_fast.prepare_dataset(p, template, source)
		###print("\n前処理後のテンプレートの点群数：", np.shape(np.array(template_down.points)))
		###print("前処理後のソースの点群数：", np.shape(np.array(source_down.points)), "\n")
		
		### 前処理後の色付け、表示 ###
		source_down.paint_uniform_color([0, 1, 0])
		template_down.paint_uniform_color([1, 0, 0])
		#o3d.visualization.draw_geometries([source_down, template_down])
		
		### FGRアルゴリズム（概略マッチング） ###
		#result_fgr = Global_optimizer_fast.execute_global_registration(source_down, template_down, source_fpfh, template_fpfh, voxel_size)
		result_fgr = Global_optimizer_fast.execute_fast_global_registration(source_down, template_down, source_fpfh, template_fpfh, voxel_size, pattern)
		
		source_vis_2 = o3d.geometry.PointCloud()
		#numpy_source_vis = result_fgr.transformation.T @ np.array(source_vis.points).T
		numpy_source_vis = np.matmul(result_fgr.transformation[0:3, 0:3], np.array(source_down.points).T).T + result_fgr.transformation[0:3, 3]
		source_vis_2.points = o3d.utility.Vector3dVector(numpy_source_vis)
		source_vis_2.paint_uniform_color([0, 1, 0])
		#o3d.visualization.draw_geometries([template_down, source_vis_2, o])     # テンプレ、FGR後ソース
		###print("\nFGRの変換行列：\n", result_fgr.transformation)
		
		### ICPアルゴリズム（精密マッチング） ###
		res = o3d.pipelines.registration.registration_icp(source, template, self.threshold, result_fgr.transformation, criteria=self.criteria)	# icp registration in open3d.
		
		est_R, est_t, est_T = self.postprocess(res)
		
		result = {'est_R': est_R,
		          'est_t': est_t,
		          'est_T': est_T}
		"""
		#print(est_R)      # 回転移動(3×3)
		#print(est_t)      # 平行移動
		"""
		###print("\nFGR+ICPの変換行列：\n", est_T)      # 変換行列全体(4×4)、重心移動分も含まれる
		
		#print('\nfitness:', float(res.fitness))
		#print("source:", np.shape(np.array(source.points)))
		#print("template:", np.shape(np.array(template.points)))
		#print('\ninlier_rmse:', float(res.inlier_rmse))
		
		if self.is_tensor: result = self.convert2tensor(result)
		return result

# Define Registration Algorithm.
def registration_algorithm():
  
  reg_algorithm = ICP()
  
  
  return reg_algorithm


# Register template and source pairs.
class Registration:
	def __init__(self, pattern="A"):
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

	def register(self, template, source, p):
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
  #print(pattern)
  transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]     # ※matmul：行列の積　　第一項：回転、第二項：平行移動、重心移動分も含まれる
  #print(est_T)
  
  numpy_source_t = source + est_T[0:3, 3]
  source_t = o3d.geometry.PointCloud()
  source_t.points = o3d.utility.Vector3dVector(numpy_source_t)
  source_t.paint_uniform_color([0, 1, 0])
  
  ### 原点を表示 ###
  numpy_o = np.array([[0, 0, 0]])
  o = o3d.geometry.PointCloud()
  o.points = o3d.utility.Vector3dVector(numpy_o)
  o.paint_uniform_color([1/3, 1/3, 1/3])
    
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
  
  ### 正解を定義 ###
  if pattern == "A":
  	ans_theta_x = np.radians(0)
  	ans_theta_y = np.radians(1)       
  	ans_theta_z = np.radians(176)
  elif pattern == "45":
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(45)
  elif pattern == "B":
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(-90)
  elif pattern == "135":
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(-45)
  elif pattern == "D":
  	ans_theta_x = np.radians(-3)
  	ans_theta_y = np.radians(-1)
  	ans_theta_z = np.radians(-85)
  elif pattern == "C":
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
  ans_R = R_z @ R_y @ R_x
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
  ###print("x軸方向　", abs(diff_R_x))
  ###print("y軸方向　", abs(diff_R_y))
  ###print("z軸方向　", abs(diff_R_z))
  global diff_R
  diff_R = np.linalg.norm([diff_R_x, diff_R_y, diff_R_z])
  print("\n回転移動の差（L2ノルム）：", diff_R)
  
  ### 平行移動の差分 ###
  global diff_t
  
  print("est_T[0:3, 3]")
  print(est_T[0:3, 3])
  print("ans_t ")
  print(ans_t)
  diff_t = np.linalg.norm(est_T[0:3, 3] - ans_t)
  print("平行移動の差（L2ノルム）：", diff_t, "\n")
  
  template = pc2open3d(template)
  source = pc2open3d(source)
  transformed_source = pc2open3d(transformed_source)

  template.paint_uniform_color([1, 0, 0])
  source.paint_uniform_color([0, 1, 0])
  transformed_source.paint_uniform_color([0, 1, 0])
  ans_source.paint_uniform_color([1/3, 1/3, 1/3])

  #o3d.visualization.draw_geometries([template])                                    # テンプレ
  ###o3d.visualization.draw_geometries([template, source, ans_source, o])                 # テンプレ、ソース、正解ソース、原点
  #o3d.visualization.draw_geometries([template, source, source_t, o, ax_x, ax_y, ax_z])
  o3d.visualization.draw_geometries([template, transformed_source])         # テンプレ、ソース、変換後ソース、原点
  #o3d.visualization.draw_geometries([template, masked_template, source])           # テンプレ、マスクテンプレ、ソース
  #o3d.visualization.draw_geometries([template, source])                            # テンプレ、ソース
  #o3d.visualization.draw_geometries([masked_template, source])                     # マスクテンプレ、ソース
