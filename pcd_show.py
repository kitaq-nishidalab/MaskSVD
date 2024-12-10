import open3d as o3d
import numpy as np

# PCDファイルを読み込む
pcd = o3d.io.read_point_cloud("/home/nishidalab0/vision_ws/src/MaskSVD/mask_T.pcd")
#pcd2 = o3d.io.read_point_cloud("/home/nishidalab0/vision_ws/src/MaskSVD/saved_point_clouds/20240926_153809/point_cloud_mask_00989.pcd")
print(np.shape(np.array(pcd.points)))
print(min(np.array(pcd.points)[:, 2]))
pcd.paint_uniform_color([0, 0, 1])
# ポイントクラウドを表示する
o3d.visualization.draw_geometries([pcd])
