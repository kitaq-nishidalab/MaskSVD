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
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    """
    pcd_down = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius=voxel_size,
                                                       non_max_radius=0.012,
                                                        gamma_21=0.5,
                                                       gamma_32=0.5)
    """


    #radius_normal = voxel_size * 2.0   # default
    if Registration_test_jurai_fast.theta == 0:
    	radius_normal = voxel_size * 2.0   # 0度
    elif Registration_test_jurai_fast.theta == 90:
    	radius_normal = voxel_size * 2.0   # 90度
    elif Registration_test_jurai_fast.theta == "L_90":
    	radius_normal = voxel_size * 2.0   # L90度
    elif Registration_test_jurai_fast.theta == "L_180":
    	radius_normal = voxel_size * 2.0   # L180度
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #radius_feature = voxel_size * 5.0   # default
    if Registration_test_jurai_fast.theta == 0:
    	radius_feature = voxel_size * 4.0   # 0度
    elif Registration_test_jurai_fast.theta == 90:
    	radius_feature = voxel_size * 5.0   # 90度
    elif Registration_test_jurai_fast.theta == "L_90":
    	radius_feature = voxel_size * 5.0   # L90度
    elif Registration_test_jurai_fast.theta == "L_180":
    	radius_feature = voxel_size * 4.0   # L180度
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, target, source):

    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


############################################
# Fast Global Registration
############################################
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    if Registration_test_jurai_fast.theta == 0:
    	distance_threshold = voxel_size * 0.5   # 0度
    	decrease_mu = True
    	division_factor = 0.2 #1.4
    	iteration_number = 100 #64
    	tuple_scale = 0.3 #95
    elif Registration_test_jurai_fast.theta == 90:
    	distance_threshold = voxel_size * 0.02   # 90度
    	decrease_mu = True
    	division_factor = 0.2 #1.4
    	iteration_number = 100 #64
    	tuple_scale = 0.3 #95
    elif Registration_test_jurai_fast.theta == "L_90":
    	distance_threshold = voxel_size * 0.06    # L90度
    	decrease_mu = True
    	division_factor = 0.2 #1.4
    	iteration_number = 100 #64
    	tuple_scale = 0.3 #95
    elif Registration_test_jurai_fast.theta == "L_180":
    	distance_threshold = voxel_size * 0.5    # L180度
    	decrease_mu = True
    	division_factor = 0.2 #1.4
    	iteration_number = 100 #64
    	tuple_scale = 0.2 #95
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            decrease_mu = decrease_mu,
            division_factor = division_factor, #1.4
            iteration_number = iteration_number, #64
            tuple_scale = tuple_scale, #95
            maximum_correspondence_distance=distance_threshold)) #0.025
    return result


