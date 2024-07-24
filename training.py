import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import open3d as o3d
from tqdm import tqdm

import network
import dataset

parser = argparse.ArgumentParser("MaskNet training", add_help=True)
parser.add_argument("--save_path", "-s", type=str, required=True, help="path to save file")
parser.add_argument("--epoch", type=int, default=300, help="epoch")
parser.add_argument("--batch", type=int, default=16, help="batch size")
args = parser.parse_args()
###################################
#--ネットワークの定義--
###################################
model = network.MaskNet()
#save_path = "/content/drive/MyDrive/ColabNotebooks/research/M1_object_detection/masknet_checkpoint/model_weight_epoch30.pth"
#model = torch.load(save_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

###################################
#--誤差関数--
###################################
# 誤差関数（平均二乗誤差）の定義
criterion = nn.MSELoss()

# 最適化器（Adam）の定義
# 引数1：最適化するネットワークのパラメータ parameters()で取得可能
learnable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(learnable_params, lr=0.0001)

###trainset = dataset.RegistrationData(algorithm='PointNetLK', data_class=dataset.ModelNet40Data(train=True, num_points=1024), partial_source=True, noise=False)
trainset = dataset.RegistrationData(algorithm='PointNetLK', data_class=dataset.ModelNet40Data(train=True, num_points=1024), partial_source=True, noise=False, num_subsampled_points=256)

index = 0

template_sample, source_sample, igt_sample, gt_mask_sample = trainset.__getitem__(index)

# template の値を確認
print("Template:", template_sample)
print(np.shape(template_sample.to('cpu').detach().numpy().copy()))

#pcd_template_sample = o3d.geometry.PointCloud()
#pcd_template_sample.points = o3d.utility.Vector3dVector(template_sample.to('cpu').detach().numpy())
#pcd_source_sample = o3d.geometry.PointCloud()
#pcd_source_sample.points = o3d.utility.Vector3dVector(source_sample.to('cpu').detach().numpy())

#o3d.visualization.draw_geometries([pcd_template_sample, pcd_source_sample])

batch_size = args.batch
test_batch_size = 8
workers = 2

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)

###################################
#--ネットワークの学習---
###################################
# エポック数：データセットを何周するか
epochs = args.epoch

for epoch in range(epochs):
  print(epoch) # epoch数表示
  model.train()
  train_loss = 0.0
  pred  = 0.0
  count = 0
  for i, data in enumerate(tqdm(train_loader)):
    template, source, igt, gt_mask = data

    # 実行するデバイスにデータを移動
    template = template.to(device)
    source = source.to(device)
    igt = igt.to(device)					# [source] = [igt]*[template]
    gt_mask = gt_mask.to(device)

    masked_template, predicted_mask = model(template, source)

    loss_mask = torch.nn.functional.mse_loss(predicted_mask, gt_mask)

    # forward + backward + optimize
    optimizer.zero_grad()

    #誤差逆伝播
    loss_mask.backward()

    #パラメータの更新
    optimizer.step()

    train_loss += loss_mask.item()
    count += 1

  train_loss = float(train_loss)/count

  print(f"Traininig Loss:{train_loss}\n")
  
###################################
#--ネットワークの保存--
###################################
save_path = args.save_path      #"/home/nishidalab0/MaskNet/checkpoint/model_weight_epoch300_batchsize16.pth"
torch.save(model, save_path)
print("Finish saving !!!!")

