import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("WMU2LR2020.stl")

pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=5000)  # サンプリングのポイント数を調整
pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) * 0.001)

 # PCDファイルに保存
o3d.io.write_point_cloud("../WMU2LR2020_5000.ply", pcd)
