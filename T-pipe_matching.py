import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import copy

###########################
# 位置合わせ描画
############################
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

###########################
# 特徴量
############################
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 3
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, target, source):

    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

###########################
# RANSACによる概略マッチング
############################
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
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

###########################
# ICP
############################
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 3.0
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

pcd_target = o3d.io.read_point_cloud("/home/nishidalab/MaskNet/TNUTEJN016.pcd", remove_nan_points=True)
pcd_source = o3d.io.read_point_cloud("/home/nishidalab/MaskNet/sensor_cheese_ushiro.pcd", remove_nan_points=True)


#upsample_block = ml3d.models.NearestUpsampleBlock(pcd_cheese_rot)
#high_res_pcd_cheese_rot = upsample_block(pcd_cheese_rot)
add_list = []
kdtree = o3d.geometry.KDTreeFlann(pcd_source)
K = 3
for point in pcd_source.points:
  k, idx, _ = kdtree.search_knn_vector_3d(point, K)
  new_point = np.mean(np.array(pcd_source.points)[list(idx), :], axis=0)
  #print(new_point)
  add_list.append(new_point)

print(len(np.array(pcd_source.points)))
print(len(add_list))
print(len(np.insert(np.array(pcd_source.points), -1, add_list, axis=0)))
pcd_source.points = o3d.utility.Vector3dVector(np.insert(np.array(pcd_source.points), -1, add_list, axis=0))
print(len(np.array(pcd_source.points)))

pcd_target = pcd_target.voxel_down_sample(voxel_size=0.006)
pcd_source = pcd_source.voxel_down_sample(voxel_size=0.006)


###########################
# 特徴量
############################
voxel_size = 0.011
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, pcd_target, pcd_source)
###########################
# RANSACによる概略マッチング
############################
result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print(result_ransac)
draw_registration_result(source_down, target_down, result_ransac.transformation)
###########################
# ICP
############################
result_all_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print(result_all_icp)
draw_registration_result(source, target, result_all_icp.transformation)


print("変換行列")
print(result_all_icp.transformation)
rotation_matrix = result_all_icp.transformation[:3, :3]
print("回転行列")
print(rotation_matrix)


def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3x3 rotation matrix to Euler angles.

    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        numpy.ndarray: Euler angles [roll, pitch, yaw] in radians.
    """
    # Extract angles using trigonometric relations
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])
    
    
# Convert rotation matrix to Euler angles
euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

# Extract rotation angles around x, y, and z axes
rotation_angle_x = np.degrees(euler_angles[0])
rotation_angle_y = np.degrees(euler_angles[1])
rotation_angle_z = np.degrees(euler_angles[2])

print("Rotation angle around x-axis:", rotation_angle_x, "degrees")
print("Rotation angle around y-axis:", rotation_angle_y, "degrees")
print("Rotation angle around z-axis:", rotation_angle_z, "degrees")

