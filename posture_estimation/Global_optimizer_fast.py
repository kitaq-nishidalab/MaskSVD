import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import copy
import Registration_test_jurai_fast

############################################
# 位置合わせ描画
############################################
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

############################################
# 点群のダウンサンプリングと特徴量計算
############################################
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    #radius_normal = voxel_size * 2.0   # default
    radius_normal = voxel_size * 2.0   # T：1.7　, L：2.0
    
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)) # T：30　, L：30

    #radius_feature = voxel_size * 5.0   # default
    radius_feature = voxel_size * 5.0  # T：4.5　, L：5.0
    
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)) # T：100　, L：100
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, target, source):

    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


############################################
# Fast Global Registration(FGR)
############################################
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = 0.001  # T：0.001　, L：0.001
    decrease_mu = True
    division_factor = 0.005 #1.4　T：0.005　, L：0.002
    iteration_number = 100 #64　T：100　, L：64
    tuple_scale = 0.5 #0.95　T：0.5　, L：0.5
    
	
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            decrease_mu = decrease_mu,
            division_factor = division_factor, #1.4
            iteration_number = iteration_number, #64
            tuple_scale = tuple_scale, #95
            maximum_correspondence_distance=distance_threshold)) #0.025
    return result


