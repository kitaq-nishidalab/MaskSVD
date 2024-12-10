import open3d as o3d
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torch import sin, cos
import open3d as o3d
from tqdm import tqdm
import Registration_test
import Registration_test_ani
import copy

def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(3.0, 0.0)
        return False

mask_file = "/home/nishidalab0/vision_ws/src/MaskSVD/mask_T2.pcd"
model_file = "model_2.pcd"

pcd_mask = o3d.io.read_point_cloud(mask_file)
pcd_model = o3d.io.read_point_cloud(model_file)


pcd_mask.paint_uniform_color([0, 0, 1.0])
pcd_model.paint_uniform_color([1.0, 0, 0])

o3d.visualization.draw_geometries_with_animation_callback([pcd_model, pcd_mask], rotate_view)







