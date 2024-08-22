# MaskSVD: The method using Masknet and SVD

[**Paper**] [SICE2023_Iwai11.pdf](https://github.com/user-attachments/files/16595458/SICE2023_Iwai11.pdf)

In this work, a fast posture estimation method using SVD and MaskNet was proposed.
We confirmed that proposed method can execute from 3 to 50 times faster than conventional method.

### Processing flow of the proposed method:

<p align="center">
      <img src="https://github.com/user-attachments/assets/e07e608d-5152-4a59-b39f-7f5b1bdd62c0" height="300">
</p>

# DEMO

Red PCD is Target. \
Green PCD is Measured PCD obtained by LiDAR.
 
### Point Cloud Dataset(PCD):

<p align="center">
      <img src="https://github.com/user-attachments/assets/541bdc40-3e8d-4c9f-ae19-8cd6f88f562e" height="200">
      <img src="https://github.com/user-attachments/assets/0a3075bf-cf28-4b9f-8104-22c716e6283b" height="200">
</p>

PCD distributions are aligned by PCD density adjusting.

### PCD density adjusting:

<p align="center">
      <img src="https://github.com/user-attachments/assets/0b3c1752-225a-42c4-8fed-edc2080912be" height="250" >
</p>

MaskNet register shapes of model PCD and measured PCD.

### The function of MaskNet:

<p align="center">
      <img src="https://github.com/user-attachments/assets/bcfac44b-8926-4a15-9538-be655536fde8" height="300">
</p>

### Posture estimation: 

<p align="center">
      <img src="https://github.com/user-attachments/assets/bf8de7c7-8fbf-43e2-8356-c5917fcb2b14" height="200" >
</p>

# Features

The fast global registration and PointNetLK employed to compare the performance of the proposed method.

### A comparison of the time

<p align="center">
      <img src="https://github.com/user-attachments/assets/5b695d36-5611-431c-8b40-84fcbf2d45df" height="350" >
      <img src="https://github.com/user-attachments/assets/aa764a99-f6fc-485b-8f97-0aea9d1406e5" height="400" >
</p>
 
# Installation

```bash
pip install open3d
pip install torch
```

```bash
git clone https://github.com/Iwaiy/MaskSVD.git
cd MaskSVD
```

# Usage
## Demo MaskSVD

Demonstrate pattern "A", "B", "C", "D".

```bash
python3 T-pipe_demo.py -s checkpoint/model_weight_epoch300_batchsize32_plane.pth --pattern "A"
```
## Note
This program requires GPU.
So, if you have only CPU, please change program as shown below.
```bash
model_load = torch.load(save_path, map_location=torch.device('cpu'))
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

Learning of MaskNet executes.

```bash
python3 training.py -s "checkpoint/sample.pth" --epoch 300
```
# Acknowledgement

https://github.com/vinits5/masknet/tree/main

# Author
 
* Yu Iwai
* The University of Kitakyushu
 
# License

This project is release under the MIT License.
