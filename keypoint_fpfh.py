import open3d as o3d


def keypoint_fpfh(pcd, radius=0.7, threthold=0.01):
    # ボクセルダウンサンプリングを行う
    #voxel_size = 0.01
    #pcd = pcd.voxel_down_sample(voxel_size)

    # 法線推定
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # FPFH特徴量を計算する
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))

    # 特徴点を表示（ダウンサンプリングされた点群を色分けして表示）
    pcd.paint_uniform_color([0, 1, 0])  # 元の点群は灰色で表示
    keypoints_pcd = o3d.geometry.PointCloud()

    # 特徴量が高いポイントを抽出
    threshold = 0.01  # 特徴量のしきい値（適宜調整）
    keypoints = [pcd.points[i] for i in range(len(fpfh.data[0])) if fpfh.data[:, i].max() > threshold]

    # 特徴点を赤色に設定
    keypoints_pcd.points = o3d.utility.Vector3dVector(keypoints)
    keypoints_pcd.paint_uniform_color([1, 0, 0])

    return keypoints_pcd

if __name__=='__main__':
    # 点群データの読み込み
    pcd = o3d.io.read_point_cloud("TNUTEJN016.pcd")

    keypoints_pcd = keypoint_fpfh(pcd, radius=0.55, threthold=0.01)

    # 特徴点と元の点群を一緒に表示
    o3d.visualization.draw_geometries([keypoints_pcd, pcd])