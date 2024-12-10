import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import subprocess
import shlex
import json
import glob
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torch import sin, cos
import open3d as o3d
from tqdm import tqdm
import torchvision
import logging
import random
import Registration_test
import time

parser = argparse.ArgumentParser("Proposed method execute", add_help=True)
parser.add_argument("--save_path", "-s", type=str,default="checkpoint/model_weight_epoch300_batchsize32_plane.pth", required=True, help="path to save file")
parser.add_argument("--pattern", type=str, default="A", help="Target Pattern")

args = parser.parse_args()

# モデルのロード
save_path = args.save_path
model_load = torch.load(save_path)

pattern = args.pattern
if  pattern== "A":   ## Pattern A  (0度) ##
	pcd_file = "TNUTEJN016.pcd" # テンプレートデータのロード
	pcd_rot_file = "sensor_cheese_noise.pcd" # ソースデータのロード
elif pattern == "45":   ## 45度 ##
	pcd_file = "TNUTEJN016.pcd" # テンプレートデータのロード
	pcd_rot_file = "sensor_tpip_45_3.pcd" # ソースデータのロード
elif pattern == "B":   ## Pattern B  (90度) ##
	pcd_file = "TNUTEJN016.pcd" # テンプレートデータのロード
	pcd_rot_file = "sensor_tpip_90_1.pcd" # ソースデータのロード
elif pattern == "135":   ## 135度 ##
	pcd_file = "TNUTEJN016.pcd" # テンプレートデータのロード
	pcd_rot_file = "sensor_tpip_135_2.pcd" # ソースデータのロード
elif pattern == "D":   ## Pattern D  (L90) ##
	pcd_rot_file = "sensor_Ljoint_90_4.pcd" # テンプレートデータのロード
	pcd_file = "WMU2LR2020.pcd" # ソースデータのロード
elif pattern == "C":   ## Pattern C  (L180) ##
	pcd_rot_file = "sensor_Ljoint_180_2.pcd" # テンプレートデータのロード
	pcd_file = "WMU2LR2020.pcd" # ソースデータのロード

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
if pattern == "A":   
	weight = 0.8   
elif pattern == "45":  
	weight = 0.8   
elif pattern == "B":   
	weight = 0.8   
elif pattern == "135":   
	weight = 0.8   
elif pattern == "D":
	weight = 0.8   
elif pattern == "C":   
	weight = 0.8   
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

# テンプレート点群とソース点群の表示
###o3d.visualization.draw_geometries([pcd_cheese_points, pcd_cheese_rot_points])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_load.to(device)

template_cheese = torch.tensor(numpy_cheese_points, dtype=torch.float32).unsqueeze(0)
source_cheese = torch.tensor(numpy_cheese_rot_points, dtype=torch.float32).unsqueeze(0)
template_cheese = template_cheese.to(device)
source_cheese = source_cheese.to(device)

# 位置合わせ準備を行う（関数を呼び出す）
registration_model = Registration_test.Registration(pattern=pattern)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

#####計測開始#####
start.record()

n = 100
sum_diff_R = 0
sum_diff_t = 0
lists_R = []
lists_t = []
for i in range(n):
	with torch.no_grad():
		masked_template_cheese, predicted_mask_cheese = model_load(template_cheese, source_cheese)

	###print("\nマスクテンプレ点群の配列のサイズ：", masked_template_cheese.size())
	###print("ソース点群の配列のサイズ：", source_cheese.size())

	# 提案手法（MaskNet、SVD、ICP）の実行（実際のデータを代入）
	result_cheese = registration_model.register(masked_template_cheese, source_cheese)
	est_T_cheese = result_cheese['est_T']     # est_T：ICPの変換行列

	# SVD+ICP処理、点群の表示
	Registration_test.display_results_sample(
		template_cheese.detach().cpu().numpy()[0], 
		source_cheese.detach().cpu().numpy()[0], 
		est_T_cheese.detach().cpu().numpy()[0], 
		masked_template_cheese.detach().cpu().numpy()[0],
		pattern)
	
	sum_diff_R += Registration_test.diff_R
	sum_diff_t += Registration_test.diff_t
	
	lists_R.append(Registration_test.diff_R)
	lists_t.append(Registration_test.diff_t)
# 平均
mean_diff_R = sum_diff_R / n
mean_diff_t = sum_diff_t / n
print("\n回転行列のL2ノルムの平均：", mean_diff_R)
print("平行移動ベクトルのL2ノルムの平均：", mean_diff_t)
# 分散
dev_diff_R = 0
dev_diff_t = 0
for i in range(n):
	dev_diff_R += (lists_R[i] - mean_diff_R)**2
	dev_diff_t += (lists_t[i] - mean_diff_t)**2
var_diff_R = dev_diff_R / n
var_diff_t = dev_diff_t / n
print("\n回転行列のL2ノルムの分散：", var_diff_R)
print("平行移動ベクトルのL2ノルムの分散：", var_diff_t)

#print(len(source_cheese.detach().cpu().numpy()[0]))
#print(len(masked_template_cheese.detach().cpu().numpy()[0]))

#####計測終了#####
end.record()
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)
#print(elapsed_time / 1000, 'sec.')
