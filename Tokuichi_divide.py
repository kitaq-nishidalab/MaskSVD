import numpy as np
import subprocess
import shlex
import json
import glob
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial import cKDTree
import os
import open3d as o3d
from numpy.linalg import svd
from scipy.sparse.linalg import svds
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
	pointcloud1 = pointcloud1
	num_points = pointcloud1.shape[0]
	nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
	random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
	idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
	return pointcloud1[idx1, :]
	
num_bunny_points = 15000
bunny_test_batch_size = 1
bunny_workers = 1
###############
#データの作成
###############
# Read Stanford bunny's point cloud.
bunny_path = os.path.join('/home/nishidalab/MaskNet/bun_zipper.ply')
if not os.path.exists(bunny_path):
  print("Please download bunny dataset from http://graphics.stanford.edu/data/3Dscanrep/")
  print("Add the extracted folder in learning3d/data/")
bunny_data = o3d.io.read_point_cloud(bunny_path)
bunny_points = np.array(bunny_data.points)
bunny_idx = np.arange(bunny_points.shape[0])
np.random.shuffle(bunny_idx)
bunny_points = bunny_points[bunny_idx[:num_bunny_points]]
rotation_angle = np.radians(90)
#rotation_angle = 0.7
print('回転角度：', np.degrees(rotation_angle))
print('回転角度（radian）：', rotation_angle)
rotation_matrix = np.array([[np.cos(rotation_angle), np.sin(rotation_angle), 0],
                            [-np.sin(rotation_angle), np.cos(rotation_angle), 0],
                            [0, 0, 1]])
source = np.dot(rotation_matrix, bunny_points.T).T
###############
#重心合わせ
###############
def calculate_centroid(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    #print(centroid)
    return centroid
def translate_to_origin(pointcloud):
    centroid = calculate_centroid(pointcloud)
    translated_pointcloud = pointcloud - centroid
    #print(np.mean(translated_pointcloud, axis=0))
    return translated_pointcloud
source = translate_to_origin(source)
bunny_points = translate_to_origin(bunny_points)
###############
#点群の表示
###############
pcd_source = o3d.geometry.PointCloud()
pcd_source.points = o3d.utility.Vector3dVector(source)
pcd_template = o3d.geometry.PointCloud()
pcd_template.points = o3d.utility.Vector3dVector(bunny_points)
pcd_template.paint_uniform_color([1, 0, 0])
pcd_source.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pcd_template, pcd_source])

"""
######################
#特異値分解によるマッチング
######################
ut, st, vt = svd(bunny_points.T, full_matrices=False)
us, ss, vs = svd(source.T, full_matrices=False)

#Template
#左特異ベクトルと固有値
T_vals, T_vec =  np.linalg.eigh((bunny_points.T)@((bunny_points.T).T))
print('\nTemplate_固有値\n' , T_vals)
ut = T_vec.T

print('\nTemplate_左特異ベクトル\n' , ut)
#Source
#左特異ベクトルと固有値
S_vals, S_vec =  np.linalg.eigh((source.T)@((source.T).T))
print('\nSource_固有値\n' , S_vals)
us = S_vec.T
print('\nSource_左特異ベクトル\n' , us)

#４０度までだったらこれ
us = abs(us)
us[0, 0] = -1 * us[0, 0]
us[2, 0] = -1 * us[2, 0]
us[0, 1] = -1 * us[0, 1]
us[1, 1] = -1 * us[1, 1]
"""

######################
#PCAによるマッチング
######################
T_trans = bunny_points
print(np.shape(T_trans)[0])
# Perform PCA
T_cov = T_trans.T @ T_trans / (np.shape(T_trans)[0] - 1)
T_W, T_V_pca = np.linalg.eig(T_cov)
# Sort eigenvectors with eigenvalues
T_index = T_W.argsort()[::-1]
T_W = T_W[T_index]
T_V_pca = T_V_pca[:, T_index]

S_trans = source
print(np.shape(S_trans)[0])
# Perform PCA
S_cov = S_trans.T @ S_trans / (np.shape(S_trans)[0] - 1)
S_W, S_V_pca = np.linalg.eig(S_cov)
# Sort eignvectors with eignvalues
S_index = S_W.argsort()[::-1]
S_W = S_W[S_index]
S_V_pca = S_V_pca[:, S_index]
#print(T_W)
print('\nPCA_Template_左特異ベクトル\n', T_V_pca)
#print(S_W)
print('\nPCA_Source_左特異ベクトル\n', S_V_pca)

#print(ut, us)
R = S_V_pca @ T_V_pca.T
#print("shape", np.shape(ut), np.shape(st), np.shape(vt))
print("回転行列 R：")
print(R)
#print("シグマ")
#print(ss)
###############
#点群の表示
###############
pcd_source.points = o3d.utility.Vector3dVector((R.T @ source.T).T)
o3d.visualization.draw_geometries([pcd_template, pcd_source])

ut0_vector = T_V_pca[:, 0]
ut1_vector = T_V_pca[:, 1]
ut2_vector = T_V_pca[:, 2]
us0_vector = S_V_pca[:, 0]
us1_vector = S_V_pca[:, 1]
us2_vector = S_V_pca[:, 2]

#ut0_vector = ut[:, 0]
#ut1_vector = ut[:, 1]
#ut2_vector = ut[:, 2]
#us0_vector = us[:, 0]
#us1_vector = us[:, 1]
#us2_vector = us[:, 2]

# Plotlyでベクトルを表示
fig = go.Figure()

# 始点（原点）を作成
origin = np.zeros(3)
print(ut0_vector)
print(ut1_vector)
print(ut2_vector)
print(us0_vector)
print(us1_vector)
print(us2_vector)

#input
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
              
fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111, projection='3d')

# 3D座標を設定
coordinate_3d(ax, [-1, 1], [-1, 1], [-1, 1], grid = True)

# 始点を設定
o = [0, 0, 0]

# 3Dベクトルを定義
v1 = np.array(ut0_vector)
v2 = np.array(ut1_vector)
v3 = np.array(ut2_vector)
v4 = np.array(us0_vector)
v5 = np.array(us1_vector)
v6 = np.array(us2_vector)

# 3Dベクトルを配置
visual_vector_3d(ax, o, v1, "red")
visual_vector_3d(ax, o, v2, "blue")
visual_vector_3d(ax, o, v3, "green")
visual_vector_3d(ax, o, v4, "orange")
visual_vector_3d(ax, o, v5, "cyan")
visual_vector_3d(ax, o, v6, "lime")
plt.show()
