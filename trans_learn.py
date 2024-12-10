import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from tqdm import tqdm

import network
import dataset_transform as dataset

def custom_collate_fn(batch):
    # 各タプルの template 点群と source 点群の最大サイズを取得
    max_size_template = max(t[0].size(0) for t in batch)  # template 点群の最大サイズ
    max_size_source = max(t[1].size(0) for t in batch)    # source 点群の最大サイズ
    max_size = max(max_size_template, max_size_source)    # 両方のうち、大きい方を選択

    padded_batch = []
    for template, source, gt_mask in batch:
        # template と source の点群をパディング
        padded_template = torch.cat([template, torch.zeros(max_size - template.size(0), template.size(1))], dim=0)
        padded_source = torch.cat([source, torch.zeros(max_size - source.size(0), source.size(1))], dim=0)
        
        # gt_mask はそのままでOK
        padded_batch.append((padded_template, padded_source, gt_mask))

    # パディングされたバッチをまとめる
    templates, sources, gt_masks = zip(*padded_batch)
    return torch.stack(templates), torch.stack(sources), torch.stack(gt_masks)

def main():
    parser = argparse.ArgumentParser("MaskNet training", add_help=True)
    parser.add_argument("--save_path", "-s", type=str,  default="checkpoint/unnoise_transformed_epoch200.pth", help="path to save file")
    parser.add_argument("--epoch", type=int, default=200, help="epoch")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    args = parser.parse_args()

    ###################################
    #--ネットワークの定義--
    ###################################
    #model = network.MaskNet()

    # 事前学習済みモデルを読み込み
    #model = torch.load("checkpoint/model_weight_epoch300_717_batchsize32.pth")
    #model = torch.load("checkpoint/model_weight_epoch300_batchsize32_plane.pth")
    model = torch.load("checkpoint/unnoise_transformed_epoch200.pth")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###################################
    #--誤差関数--
    ###################################
    criterion = nn.MSELoss()

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(learnable_params, lr=0.0001)
    
    #Tpipe
    #template_path = "TNUTEJN016_1024.ply"
    #source_dir = "/home/nishidalab0/vision_ws_blender/output/Tpipe_unnoise/FilterMasks"
    #gt_mask_path = "/home/nishidalab0/vision_ws_blender/output/Tpipe_unnoise/gtmask/masks.npy"
    #Lpipe
    template_path = "WMU2LR2020_1024.ply"
    source_dir = "/home/nishidalab0/vision_ws_blender/output/Lpipe/FilterMasks"
    gt_mask_path = "/home/nishidalab0/vision_ws_blender/output/Lpipe/gtmask/masks.npy"

    trainset = dataset.RegistrationData(template_path=template_path, source_dir=source_dir, gt_mask_path=gt_mask_path)
    
    index = 0
    template_sample, source_sample, igt_sample = trainset.__getitem__(index)
    
    # template の値を確認
    print("Template:", template_sample)
    print(np.shape(source_sample.to('cpu').detach().numpy().copy()))
    
    batch_size = args.batch
    workers = 2

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, collate_fn=custom_collate_fn)

    ###################################
    #--ネットワークの学習---
    ###################################
    epochs = args.epoch

    for epoch in range(epochs):
        print(epoch) # epoch数表示
        model.train()
        train_loss = 0.0
        count = 0
        for i, data in enumerate(tqdm(train_loader)):
            template, source, gt_mask = data

            template = template.to(device)
            source = source.to(device)
            gt_mask = gt_mask.to(device)

            masked_template, predicted_mask = model(template, source)

            loss_mask = torch.nn.functional.mse_loss(predicted_mask, gt_mask)

            optimizer.zero_grad()
            loss_mask.backward()
            optimizer.step()

            train_loss += loss_mask.item()
            count += 1

        train_loss = float(train_loss)/count
        print(f"Traininig Loss:{train_loss}\n")

    ###################################
    #--ネットワークの保存--
    ###################################
    save_path = args.save_path
    torch.save(model, save_path)
    print("Finish saving !!!!")

if __name__ == '__main__':
    main()
