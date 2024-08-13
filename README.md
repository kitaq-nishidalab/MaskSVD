# MaskSVD: The method using Masknet and SVD

わかりやすくてカッコイイ名前をつける
 
何かを簡潔に紹介する

 ![Fig 1_5](https://github.com/user-attachments/assets/e07e608d-5152-4a59-b39f-7f5b1bdd62c0)
 
# DEMO

Red PCD is Target. \
Green PCD is Measured PCD obtained by LiDAR.
 
* Point Cloud Dataset(PCD) 

<p align="center">
      <img src="https://github.com/user-attachments/assets/541bdc40-3e8d-4c9f-ae19-8cd6f88f562e" height="200">
      <img src="https://github.com/user-attachments/assets/0a3075bf-cf28-4b9f-8104-22c716e6283b" height="200">
</p>

<img src="https://github.com/user-attachments/assets/bcfac44b-8926-4a15-9538-be655536fde8" height="200" width="270">

# Features


セールスポイントや差別化などを説明する
 
# Requirement
 
MaskSVDを動かすのに必要なライブラリなどを列挙する

 
# Installation
 
Requirementで列挙したライブラリなどのインストール方法を説明する
 
```bash
pip install open3d
pip install torch
```
 
# Usage
## Demo MaskSVD

Demonstrate pattern "A", "B", "C", "D".

```bash
python3 T-pipe_demo.py -s checkpoint/model_weight_epoch300_batchsize32_plane.pth --pattern "A"
```

## Test MaskSVD（Proposed method）and Comparative method for Accuracy

 MaskSVD executes.

```bash
git clone https://github.com/Iwaiy/MaskSVD.git
cd MaskSVD
python3 T-pipe_test.py -s checkpoint/model_weight_epoch300_batchsize32_plane.pth --pattern "A"
```

Fast global registration executes.

```bash
python3 T-pipe_test_jurai_fast.py  --pattern "A"
```

PointNetLK executes.

```bash
python3 T-pipe_test_jurai_ptlk.py  --pattern "A"
```

--pattern "< A or B or C or D >"　can chenge.　

## Train Masknet
MaskNetの学習を実行する

```bash
python3 training.py -s "checkpoint/sample.pth" --epoch 300
```

# Note

# Author
 
* Yu Iwai
* The University of Kitakyushu
 
# License

This project is release under the MIT License.
