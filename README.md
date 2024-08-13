# MaskSVD: The method using Masknet and SVD

わかりやすくてカッコイイ名前をつける
 
何かを簡潔に紹介する

 ![Fig 1_5](https://github.com/user-attachments/assets/e07e608d-5152-4a59-b39f-7f5b1bdd62c0)
 
# DEMO
 
魅力が直感的に伝えわるデモ動画や図解を載せる

<p align="center">
      <img src="https://github.com/user-attachments/assets/5251691a-9ed5-46d8-9976-ee0eb002407e" height="200">
      <img src="https://github.com/user-attachments/assets/810f926a-996b-44f1-af77-6368efb4d406" height="200">
</p>
 
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
