import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import copy
import Registration_test_jurai

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

    #radius_normal = voxel_size * 2.0   # default
    if Registration_test_jurai.theta == 0:
    	radius_normal = voxel_size * 1.5   # 0度
    elif Registration_test_jurai.theta == 90:
    	radius_normal = voxel_size * 2.0   # 90度
    elif Registration_test_jurai.theta == "L_90":
    	radius_normal = voxel_size * 2.0   # L90度
    elif Registration_test_jurai.theta == "L_180":
    	radius_normal = voxel_size * 2.0   # L180度
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #radius_feature = voxel_size * 5.0   # default
    if Registration_test_jurai.theta == 0:
    	radius_feature = voxel_size * 9.5   # 0度
    elif Registration_test_jurai.theta == 90:
    	radius_feature = voxel_size * 7.0   # 90度
    elif Registration_test_jurai.theta == "L_90":
    	radius_feature = voxel_size * 5.0   # L90度
    elif Registration_test_jurai.theta == "L_180":
    	radius_feature = voxel_size * 5.0   # L180度
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
# RANSACによる概略マッチング
############################################
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    #distance_threshold = voxel_size * 1.5   # default
    if Registration_test_jurai.theta == 0:
    	distance_threshold = voxel_size * 0.5   # 0度
    elif Registration_test_jurai.theta == 90:
    	distance_threshold = voxel_size * 0.0075   # 90度
    elif Registration_test_jurai.theta == "L_90":
    	distance_threshold = voxel_size * 0.1    # L90度
    elif Registration_test_jurai.theta == "L_180":
    	distance_threshold = voxel_size * 0.0075    # L180度
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
         [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


