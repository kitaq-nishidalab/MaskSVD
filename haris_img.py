import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt




def points_to_image(points_2d, image_size=(512, 512)):
    img = np.zeros(image_size, dtype=np.float32)
    pixel_to_point_map = {}
    for i, (x, y) in enumerate(points_2d):
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            img[y, x] = 255
            if (y, x) not in pixel_to_point_map:
                pixel_to_point_map[(y, x)] = []
            pixel_to_point_map[(y, x)].append(i)  # 保存するインデックス
    return img, pixel_to_point_map

def normalize_points(points_2d, image_size):
    min_vals = np.min(points_2d, axis=0)
    max_vals = np.max(points_2d, axis=0)
    scale = np.max(max_vals - min_vals)
    norm_points = (points_2d - min_vals) / scale * (image_size[0] - 1)
    return np.clip(norm_points, 0, image_size[0] - 1).astype(int)


def detect_harris_keypoints(image, blockSize=2, ksize=3, k=0.04, threshold_factor=0.3):
    # Harrisコーナー検出を実行
    harris_response = cv2.cornerHarris(image, blockSize, ksize, k)
    
    # 閾値を設定
    threshold = threshold_factor * harris_response.max()
    
    # コーナーが検出された点のインデックスを取得
    keypoints = np.argwhere(harris_response > threshold)
    
    return keypoints, harris_response

# 特徴点の逆変換
def backproject_keypoints_with_map(keypoints_2d_xy, keypoints_2d_yz, keypoints_2d_zx, points, map_xy, map_yz, map_zx):
    keypoints = []

    def find_3d_from_2d(yx_pair, map_):
        if yx_pair in map_:
            indices = map_[yx_pair]
            if indices:
                return points[indices[0]]  # 最初のインデックスを使用
        return None

    for y, x in keypoints_2d_xy:
        point = find_3d_from_2d((y, x), map_xy)
        if point is not None:
            keypoints.append(point)

    for y, x in keypoints_2d_yz:
        point = find_3d_from_2d((y, x), map_yz)
        if point is not None:
            keypoints.append(point)

    for x, z in keypoints_2d_zx:
        point = find_3d_from_2d((x, z), map_zx)
        if point is not None:
            keypoints.append(point)

    return np.array(keypoints)

def haris3d(pcd, blockSize=2, ksize=3, k=0.04, threshold_factor=0.3):
    points = np.asarray(pcd.points)

    # Z座標を無視してXY平面に投影
    points_2d_xy = points[:, :2].astype(np.float32)
    points_2d_xy = normalize_points(points_2d_xy, image_size=(512, 512))
    
    # YZ平面に投影
    points_2d_yz = points[:, 1:3].astype(np.float32)
    points_2d_yz = normalize_points(points_2d_yz, image_size=(512, 512))
    # ZX平面に投影
    points_2d_zx = points[:, [0, 2]].astype(np.float32)
    points_2d_zx = normalize_points(points_2d_zx, image_size=(512, 512))

    # 画像に変換
    img_xy, map_xy = points_to_image(points_2d_xy)
    img_yz, map_yz = points_to_image(points_2d_yz)
    img_zx, map_zx = points_to_image(points_2d_zx)

    # 画像のデータ型を8ビットに変換（OpenCVは8ビット画像を期待するため）
    img_xy = np.uint8(img_xy)
    img_yz = np.uint8(img_yz)
    img_zx = np.uint8(img_zx)

    # 画像を表示
    #cv2.imshow('2D Projection Image (XY Plane)', img_xy)
    #cv2.imshow('2D Projection Image (YZ Plane)', img_yz)
    #cv2.imshow('2D Projection Image (ZX Plane)', img_zx)

    # キー入力を待つ
    #cv2.waitKey(0)
    # ウィンドウを閉じる
    #cv2.destroyAllWindows()

    # Harrisコーナー検出を適用
    keypoints_xy, harris_response_xy = detect_harris_keypoints(img_xy, blockSize, ksize, k, threshold_factor)
    keypoints_yz, harris_response_yz = detect_harris_keypoints(img_yz, blockSize, ksize, k, threshold_factor)
    keypoints_zx, harris_response_zx = detect_harris_keypoints(img_zx, blockSize, ksize, k, threshold_factor)

    #print(f"Number of keypoints in XY plane: {len(keypoints_xy)}")
    #print(f"Number of keypoints in YZ plane: {len(keypoints_yz)}")
    #print(f"Number of keypoints in ZX plane: {len(keypoints_zx)}")

    # グレースケール画像をカラー画像に変換
    img_color_xy = cv2.cvtColor(img_xy, cv2.COLOR_GRAY2BGR)
    img_color_yz = cv2.cvtColor(img_yz, cv2.COLOR_GRAY2BGR)
    img_color_zx = cv2.cvtColor(img_zx, cv2.COLOR_GRAY2BGR)

    # 特徴点を赤色で描画
    for y, x in keypoints_xy:
        img_color_xy[y, x] = [0, 0, 255]  # 赤色で描画
    for y, x in keypoints_yz:
        img_color_yz[y, x] = [0, 0, 255]  # 赤色で描画
    for y, x in keypoints_zx:
        img_color_zx[y, x] = [0, 0, 255]  # 赤色で描画

    # 画像を表示
    #cv2.imshow('Harris Keypoints (XY Plane)', img_color_xy)
    #cv2.imshow('Harris Keypoints (YZ Plane)', img_color_yz)
    #cv2.imshow('Harris Keypoints (ZX Plane)', img_color_zx)
    cv2.imwrite('harris_keypoints_xy.png', img_color_xy)
    cv2.imwrite('harris_keypoints_yz.png', img_color_yz)
    cv2.imwrite('harris_keypoints_zx.png', img_color_zx)

    # キー入力を待つ
    #cv2.waitKey(0)
    # ウィンドウを閉じる
    #cv2.destroyAllWindows()

    # 特徴点の逆変換 (keypoints_2d_xy, keypoints_2d_yz, keypoints_2d_zx, points, pixel_to_point_map)
    keypoints_3d = backproject_keypoints_with_map(keypoints_xy, keypoints_yz, keypoints_zx, points, map_xy, map_yz, map_zx)

    # 3D点群として表示
    keypoints_pcd = o3d.geometry.PointCloud()
    if len(keypoints_3d) > 0:  # 空でないか確認
        keypoints_pcd.points = o3d.utility.Vector3dVector(keypoints_3d)
        keypoints_pcd.paint_uniform_color([1, 0, 0])  # 特徴点を赤色に設定
        #o3d.visualization.draw_geometries([keypoints_pcd])
        return keypoints_pcd
    else:
        print("No keypoints to display.")
        raise

if __name__=='__main__':
    # 点群の読み込み
    pcd = o3d.io.read_point_cloud("sensor_cheese_noise.pcd")
    #R = pcd.get_rotation_matrix_from_xyz((0, np.pi/45, np.pi/45))  # 45度回転
    #pcd.rotate(R, center=(0, 0, 0))

    keypoints_pcd = haris3d(pcd)

    o3d.visualization.draw_geometries([keypoints_pcd])