# MaskSVD

わかりやすくてカッコイイ名前をつける
 
何かを簡潔に紹介する
 
# DEMO
 
魅力が直感的に伝えわるデモ動画や図解を載せる
 
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

 MaskSVD(提案手法)を実行する

```bash
git clone https://github.com/Iwaiy/MaskSVD.git
cd MaskSVD
python3 T-pipe_test.py -s checkpoint/model_weight_epoch300_batchsize32_plane.pth --pattern "A"
```
Fast global registrationを実行する

```bash
python3 training.py -s "checkpoint/sample.pth" --epoch 300
```

MaskNetの学習を実行する

```bash
python3 training.py -s "checkpoint/sample.pth" --epoch 300
```


# Note

# Author
 
* Yu Iwai
* The University of Kitakyushu
 
# License
