import open3d as o3d
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torch import sin, cos
import open3d as o3d
from tqdm import tqdm
import Registration_test
import Registration_test_ani
import copy

def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

###############
#データの作成
###############
# モデルのロード
save_path = R"C:\Users\IWAI\Documents\大学院_研究\program\checkpoint\model_weight_epoch300_batchsize32_plane.pth"
model_load = torch.load(save_path,  torch.device('cpu'))

# テンプレートデータのロード
pcd_file = R"C:\Users\IWAI\Documents\大学院_研究\program\TNUTEJN016.pcd"
# ソースデータのロード
if Registration_test_ani.theta == 0:   ## 0度 ##
	pcd_rot_file = R"C:\Users\IWAI\Documents\大学院_研究\program\sensor_cheese_noise.pcd"
elif Registration_test_ani.theta == 45:   ## 45度 ##
	pass
	pcd_rot_file = R"C:\Users\IWAI\Documents\大学院_研究\program\sensor_tpip_45_3.pcd"
elif Registration_test_ani.theta == 90:   ## 90度 ##
	pcd_rot_file = R"C:\Users\IWAI\Documents\大学院_研究\program\sensor_tpip_90_1.pcd"
elif Registration_test_ani.theta == 135:   ## 135度 ##
	pass
	pcd_rot_file = R"C:\Users\IWAI\Documents\大学院_研究\program\sensor_tpip_135_2.pcd"
elif Registration_test_ani.theta == "L_90":   ## L90 ##
	pcd_rot_file = R"C:\Users\IWAI\Documents\大学院_研究\program\sensor_Ljoint_90_4.pcd"
	pcd_file = R"C:\Users\IWAI\Documents\大学院_研究\program\WMU2LR2020.pcd"
elif Registration_test_ani.theta == "L_180":   ## L180 ##
	pcd_rot_file = R"C:\Users\IWAI\Documents\大学院_研究\program\sensor_Ljoint_180_2.pcd"
	pcd_file = R"C:\Users\IWAI\Documents\大学院_研究\program\WMU2LR2020.pcd"

# テンプレとソースを点群データに変換
# テンプレとソースを点群データに変換
pcd_cheese = o3d.io.read_point_cloud(pcd_file)
pcd_cheese_rot = o3d.io.read_point_cloud(pcd_rot_file)

numpy_cheese_points = np.array(pcd_cheese.points)
numpy_cheese_rot_points = np.array(pcd_cheese_rot.points)

###print("テンプレートの点群数：", np.shape(numpy_cheese_points))
###print("ソースの点群数：", np.shape(numpy_cheese_rot_points))

### 計測点群の密度に合わせる ###
points = np.asarray(pcd_cheese_rot.points)
num_points = len(points)
###print("ソースの点群数：", num_points)

min_bound = np.min(numpy_cheese_rot_points, axis=0)
max_bound = np.max(numpy_cheese_rot_points, axis=0)

box_width = max_bound[0] - min_bound[0]
box_height = max_bound[1] - min_bound[1]
box_depth = max_bound[2] - min_bound[2]
###print("ボクセルサイズの大きさ", box_width, box_height, box_depth)

points_inside = points[(points >= min_bound) & (points <= max_bound)].reshape(-1, 3)

density = num_points / ( box_width * box_height * box_depth )
###print("密度：", density)
if Registration_test_ani.theta == 0:   ## 0度 ##
	weight = 0.8   # 0度
elif Registration_test_ani.theta == 45:   ## 90度 ##
	weight = 0.8   # 90度
elif Registration_test_ani.theta == 90:   ## 90度 ##
	weight = 0.8   # 90度
elif Registration_test_ani.theta == 135:   ## 90度 ##
	weight = 0.8   # 90度
elif Registration_test_ani.theta == "L_90":   ## 90度 ##
	weight = 0.8   # 90度
elif Registration_test_ani.theta == "L_180":   ## 90度 ##
	weight = 0.8   # 90度
voxel_size = weight * (1.0 / density) ** (1/3)
###print("ダウンサンプリングのボックスの大きさ：", voxel_size)

#pcd_cheese = pcd_cheese.voxel_down_sample(voxel_size=0.0045) #初期値0.006
pcd_cheese = pcd_cheese.voxel_down_sample(voxel_size) #初期値0.006

numpy_cheese_points = np.array(pcd_cheese.points)
numpy_cheese_rot_points = np.array(pcd_cheese_rot.points)
print(np.shape(numpy_cheese_points))
print(np.shape(numpy_cheese_rot_points))

pcd_cheese_points = o3d.geometry.PointCloud()
pcd_cheese_points.points = o3d.utility.Vector3dVector(numpy_cheese_points)
pcd_cheese_rot_points = o3d.geometry.PointCloud()
pcd_cheese_rot_points.points = o3d.utility.Vector3dVector(numpy_cheese_rot_points)

pcd_cheese_points.paint_uniform_color([1.0, 0, 0])
pcd_cheese_rot_points.paint_uniform_color([0, 1.0, 0])

#o3d.visualization.draw_geometries_with_animation_callback([pcd_cheese_rot_points], rotate_view)

# テンプレート点群とソース点群の表示
###o3d.visualization.draw_geometries([pcd_cheese_points, pcd_cheese_rot_points])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_load.to(device)

template_cheese = torch.tensor(numpy_cheese_points, dtype=torch.float32).unsqueeze(0)
source_cheese = torch.tensor(numpy_cheese_rot_points, dtype=torch.float32).unsqueeze(0)
template_cheese = template_cheese.to(device)
source_cheese = source_cheese.to(device)

# 位置合わせ準備を行う（関数を呼び出す）
registration_model = Registration_test_ani.Registration()
masked_template_cheese, _ = model_load(template_cheese, source_cheese)
	
masked_template_numpy = masked_template_cheese.detach().cpu().numpy()[0]

masked_template_pcd = o3d.geometry.PointCloud()
masked_template_pcd.points = o3d.utility.Vector3dVector(masked_template_numpy)
masked_template_pcd.paint_uniform_color([0, 0, 1])

def register_view(vis):


	result_cheese = registration_model.register(masked_template_cheese, source_cheese, vis)

	#vis.poll_events()
	#vis.update_renderer()
	#time.sleep(1.5)
	return False


#vis = o3d.visualization.Visualizer()
#vis.create_window()        
result_cheese = registration_model.register(masked_template_cheese, source_cheese)
#o3d.visualization.draw_geometries_with_animation_callback([pcd_cheese_points, pcd_cheese_rot_points], register_view)

with torch.no_grad():
	masked_template_cheese, predicted_mask_cheese = model_load(template_cheese, source_cheese)

###print("\nマスクテンプレ点群の配列のサイズ：", masked_template_cheese.size())
###print("ソース点群の配列のサイズ：", source_cheese.size())

# 提案手法（MaskNet、SVD、ICP）の実行（実際のデータを代入）
#result_cheese = registration_model.register(masked_template_cheese, source_cheese)
est_T_cheese = result_cheese['est_T']     # est_T：ICPの変換行列

# SVD+ICP処理、点群の表示
Registration_test_ani.display_results_sample(
	template_cheese.detach().cpu().numpy()[0], 
	source_cheese.detach().cpu().numpy()[0], 
	est_T_cheese.detach().cpu().numpy()[0], 
	masked_template_cheese.detach().cpu().numpy()[0])





