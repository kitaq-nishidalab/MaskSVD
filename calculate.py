import numpy as np
import math

"""
  R = [[0.70460183, -0.70951649, 0.01107297], 
       [0.70824711,  0.70413413,  0.05080513], 
       [-0.04384393, -0.02795499,  0.9986472]]
"""
  
# x軸方向に回転
theta_x = math.radians(45)
R_x = np.array(
       [[1, 0, 0], 
       [0, math.cos(theta_x), -math.sin(theta_x)], 
       [0, math.sin(theta_x), math.cos(theta_x)]])
# y軸方向に回転       
theta_y = math.radians(45)
R_y = np.array(
       [[math.cos(theta_y), 0, math.sin(theta_y)], 
       [0, 1, 0], 
       [-math.sin(theta_y), 0, math.cos(theta_y)]])
# z軸方向に回転       
theta_z = math.radians(45)
R_z = np.array(
       [[math.cos(theta_z), -math.sin(theta_z), 0], 
       [math.sin(theta_z), math.cos(theta_z), 0], 
       [0, 0, 1]])
# 平行移動
t = [-0.02837952, 0.03793951, -0.53655746]
