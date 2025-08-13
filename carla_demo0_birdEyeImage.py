
"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref

import cv2
from queue import Queue
from queue import Empty
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass



import carla
import numpy as np
import cv2
import random
import networkx as nx
import matplotlib.pyplot as plt
label_name = np.array([
    "None",
    "Building",
    "Fences",
    "Other",
    "Pedestrian",
    "Pole",
    "RoadLines",
    "Road",
    "Sidewalk",
    "Vegetation",
    "Vehicle",
    "Wall",
    "TrafficSign",
    "Sky",
    "Ground",
    "Bridge",
    "RailTrack",
    "GuardRail",
    "TrafficLight",
    "Static",
    "Dynamic",
    "Water",
    "Terrain",
])

label_color = np.array([
    (255, 255, 255), # None 0
    (70, 70, 70),    # Building 1
    (100, 40, 40),   # Fences 2
    (55, 90, 80),    # Other 3
    (220, 20, 60),   # Pedestrian 4
    (153, 153, 153), # Pole 5
    (157, 234, 50),  # RoadLines 6 
    (128, 64, 128),  # Road 7 
    (244, 35, 232),  # Sidewalk 8
    (107, 142, 35),  # Vegetation 9
    (0, 0, 142),     # Vehicle 10
    (102, 102, 156), # Wall 11
    (220, 220, 0),   # TrafficSign 12
    (70, 130, 180),  # Sky 13
    (81, 0, 81),     # Ground 14
    (150, 100, 100), # Bridge 15
    (230, 150, 140), # RailTrack 16
    (180, 165, 180), # GuardRail 17
    (250, 170, 30),  # TrafficLight 18
    (110, 190, 160), # Static 19
    (170, 120, 50),  # Dynamic 20
    (45, 60, 150),   # Water 21
    (145, 170, 100), # Terrain 22
]) 


import numpy as np
import carla
import numpy as np
import carla
from visualize_networkx_occ import visualize_save_occ_as_vehlococcplt
from visualize_networkx_occ import visualize_save_occ_as_vehlocplt
from visualize_networkx_occ import visualize_save_occpath_as_vehlococcplt

# X_new = x - a
 #Y_new = b - y  # 等价于 -(y - b)
#a, b: 新坐标系的原点在原坐标系中的坐标（即平移向量）
def vehicleOCC_to_netwrokxGrid(vehiclepts,occ):

    #平移后新原点在原坐标系中的位置 (a, b)
    h,w = occ.shape
    a = 0
    b = h
    x_old = vehiclepts[0] 
    y_old = vehiclepts[1] 
    x_new = x_old -a 
    y_new = b - y_old # y变
    return np.array([x_new,y_new]).astype(np.int32)

# X_new = x - a
 #Y_new = b - y  # 等价于 -(y - b)
 # a, b: 新坐标系的原点在原坐标系中的坐标（即平移向量）
def netwrokxGrid_to_vehicleOCC(pts,occ):
    pts = np.array(pts)
    #print(pts)
    #平移后新原点在原坐标系中的位置 (a, b)
    h,w = occ.shape
    a = 0
    b = h
    x_old = pts[:,0] 
    y_old = pts[:,1] 
    x_new = x_old +a 
    y_new = b - y_old # y变
    pts[:,0] = x_new
    pts[:,1] = y_new
    return pts

def lidar_to_world(lidar_sensor, point_cloud):
    """
    将 LiDAR 获取的点云从传感器坐标系转换到世界坐标系。
    
    :param lidar_sensor: carla.Lidar 对象（传感器 Actor）
    :param point_cloud: list of carla.Location 或 (N, 3) numpy array
                        表示 LiDAR 本地坐标系下的点 [x_lidar, y_lidar, z_lidar]
    :return: (N, 3) numpy array，世界坐标系下的点 [x_world, y_world, z_world]
    """
    # 获取 LiDAR 到世界坐标系的变换矩阵
    lidar_2_world = lidar_sensor.get_transform().get_matrix()
    T = np.array(lidar_2_world)

    # 将输入转换为 NumPy 数组
    if isinstance(point_cloud, list):
        if isinstance(point_cloud[0], carla.Location):
            points = np.array([[p.x, p.y, p.z] for p in point_cloud])
        else:
            points = np.array(point_cloud)
    else:
        points = np.array(point_cloud)

    N = points.shape[0]

    # 构造齐次坐标: (N, 4)
    ones = np.ones((N, 1))
    points_homogeneous = np.hstack([points, ones])  # shape: (N, 4)

    # 应用变换：世界坐标 = 点云齐次坐标 @ 变换矩阵的转置（因矩阵是行主序）
    # 注意：CARLA 的 matrix 是 row-major，np.array 默认处理为 row-major，直接乘即可
    points_world = points_homogeneous @ T.T  # 或者使用 np.dot(points_homogeneous, T.T)

    # 返回前三维（x, y, z）
    return points_world[:, :3]



def world_to_vehicle_points(transform, world_points):
    """
    将大量世界坐标系下的点批量转换为车辆本地坐标系下的点。
    
    :param transform: carla.Transform, 车辆的 transform (e.g., vehicle.get_transform())
    :param world_points: numpy array of shape (N, 3) 或 list of carla.Location
                         表示 N 个世界坐标点 [x, y, z]
    :return: numpy array of shape (N, 3), 转换后的本地坐标 [x_local, y_local, z_local]
    """
    # 获取从世界坐标系 → 车辆本地坐标系的逆变换矩阵
    world_to_vehicle_matrix = transform.get_inverse_matrix()
    T = np.array(world_to_vehicle_matrix)

    # 将输入统一转换为 NumPy 数组
    if isinstance(world_points, list):
        if isinstance(world_points[0], carla.Location):
            world_points = np.array([[loc.x, loc.y, loc.z] for loc in world_points])
        else:
            world_points = np.array(world_points)

    N = world_points.shape[0]

    # 构造齐次坐标: (N, 4)
    ones = np.ones((N, 1))
    points_homogeneous = np.hstack([world_points, ones])  # (N, 4)

    # 批量应用逆变换: (N, 4) @ (4, 4).T → (N, 4)
    # 注意：np.dot 或 @ 要求矩阵维度匹配，T 是 4x4，我们转置后做右乘
    points_local_homogeneous = points_homogeneous @ T.T  # 更高效的方式

    pointVeh1 = np.array([transform.location.x, transform.location.y, transform.location.z,1])
    pointVeh2 = pointVeh1 @ T.T
    #print(pointVeh1)
    #print(pointVeh2)
    points_local_homogeneous = points_local_homogeneous[:, :3] - pointVeh2[:3]  # 减去车辆位置
    # 返回前三维
    return points_local_homogeneous[:, :3]  # (N, 3)

# 观察者位置计算,并且高处往下看，d=为距离车辆的直线距离，height为高度
def get_transform(vehicle_location, angle, d=6.4,height =2,pitchAngle = -15):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), height) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=+ angle, pitch=-pitchAngle))

import numpy as np

import numpy as np

def vehicle_to_occupancy(vehicle_points, resolution=0.5, x_range=(-100, 100), y_range=(-100, 100)):
    """
    将以车中心为原点的 XY 坐标转换为 OCC 图的坐标。

    :param vehicle_points: (2,)
    :param resolution: 网格分辨率（单位：米）
    :param x_range: X 轴范围 (min, max)
    :param y_range: Y 轴范围 (min, max)
    :return: (N, 2) 的 numpy array，表示 OCC 图的坐标 [grid_x, grid_y]
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # 计算网格坐标
 
    grid_x = np.floor((vehicle_points[0] - x_min) / resolution).astype(int)
    grid_y = np.floor((vehicle_points[1] - y_min) / resolution).astype(int)



    return np.column_stack((grid_x, grid_y))

def lidar_to_occupancy(lidar_points, labels, resolution=0.5, x_range=(-100, 100), y_range=(-100, 100)):
    """
    将LiDAR点云数据转换为占据栅格地图(Occupancy Grid Map)
    
    参数:
        lidar_points: (N,2) numpy数组, LiDAR点云坐标(车辆坐标系)
        labels: (N,) numpy数组, 每个点的标签(0:可通行, 1:障碍物)
        resolution: 栅格分辨率(米)
        x_range: X轴范围(min, max)
        y_range: Y轴范围(min, max)
        
    返回:
        occ_grid: 2D numpy数组, 占据栅格地图(0:可通行, 1:障碍物)
    """
    
    # 计算栅格地图尺寸
    x_min, x_max = x_range
    y_min, y_max = y_range
    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)
    
    # 初始化栅格地图(默认全0表示可通行)
    occ_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # 计算每个点的栅格坐标
    grid_x = np.floor((lidar_points[:, 0] - x_min) / resolution).astype(int)
    grid_y = np.floor((lidar_points[:, 1] - y_min) / resolution).astype(int)
    
    # 过滤超出边界的点
    valid_mask = (grid_x >= 0) & (grid_x < grid_width) & (grid_y >= 0) & (grid_y < grid_height)
    grid_x = grid_x[valid_mask]
    grid_y = grid_y[valid_mask]
    valid_labels = labels[valid_mask]
    
    # 使用向量化操作统计每个栅格的0和1数量
    unique_grids, counts = np.unique(np.column_stack((grid_y, grid_x, valid_labels)), 
                                   axis=0, return_counts=True)
    
    # 为每个栅格统计0和1的总数
    grid_stats = {}
    for (y, x, lbl), cnt in zip(unique_grids, counts):
        if (y, x) not in grid_stats:
            grid_stats[(y, x)] = [0, 0]  # [count_0, count_1]
        grid_stats[(y, x)][lbl] += cnt
    
    # 根据多数决定原则设置栅格值
    for (y, x), (count_0, count_1) in grid_stats.items():
        if count_1 > count_0:  # 1的数量多
            occ_grid[y, x] = 1
        # 否则保持默认值0
    
    return occ_grid
import numpy as np
import networkx as nx

#从一个二维的占用栅格数组创建一个networkx无向图

def grid_to_graph_vectorized(grid):
    """
    将占用栅格转换为无向图，使用 NumPy 向量化操作加速。
    假设 grid 中 0 表示自由空间，1 表示障碍物。
    节点为 (row, col) 坐标。
    """
    # 找出所有自由单元格的坐标
    
    free_coords = np.where(grid == 0)
    rows, cols = free_coords[0], free_coords[1]
    nodes = list(zip(rows, cols))
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    # 构建边：检查四个方向 (上、下、左、右)
    # 为每个自由点生成四个邻居坐标
    offsets = [(1, 0), (0, 1)]  # 只需检查右和下，避免重复添加边
    edges = []

    for dr, dc in offsets:
        # 当前点 (r, c)，邻居为 (r+dr, c+dc)
        r_shift = rows + dr
        c_shift = cols + dc
        validTmp = c_shift > grid.shape[1]-1
        c_shift[validTmp] = grid.shape[1]-1

        validTmp = r_shift > grid.shape[0]-1
        r_shift[validTmp] = grid.shape[0]-1

        # 检查移位后的坐标是否仍在范围内且也为自由空间
        valid = (
            (r_shift >= 0) &
            (r_shift < grid.shape[0]) &
            (c_shift >= 0) &
            (c_shift < grid.shape[1]) &
            (grid[r_shift, c_shift] == 0)
        )
        
        # 收集有效的边
        valid_nodes = list(zip(rows[valid], cols[valid]))
        valid_neighbors = list(zip(r_shift[valid], c_shift[valid]))
        edges.extend(zip(valid_nodes, valid_neighbors))
    
    G.add_edges_from(edges)
    return G


#   创建一个图形表示 Occupancy Grid Map，确保路径点与障碍物的最小距离大于 1 米
# rows, cols = occ2.shape
#   G = nx.grid_2d_graph(rows, cols) 
#1.基于occ和nx.grid_2d_graph，创建二维网格图，通常会看到 (0, 0) 在左上角，X轴向右，Y轴向下
#2.基于OCC中等于255的点，转换到G图坐标，并加入图中
#3.基于缓冲区距离obstacle_buffer，建立障碍物缓冲区，移除图中对应的节点
def create_graph_from_occupancy_grid(occ2, resolution=0.5, x_range=(-100, 100), y_range=(-100, 100),obstacle_buffer=0.5):
    occ3 = occ2.copy()
    rows, cols = occ2.shape
    
    # 计算缓冲距离对应的网格数
    buffer_cells = int(obstacle_buffer / resolution)
    
    # 使用NumPy快速找到所有障碍物位置,建立的坐标是网格坐标系， (0, 0) 在左上角，X轴向右，Y轴向下
    obstacle_positions = np.argwhere(occ2 == 255)
    
    # 预生成所有可能的偏移量
    dr = np.arange(-buffer_cells, buffer_cells + 1)
    dc = np.arange(-buffer_cells, buffer_cells + 1)
    dr_grid, dc_grid = np.meshgrid(dr, dc)
    offsets = np.column_stack((dr_grid.ravel(), dc_grid.ravel()))
    
    # 计算所有需要移除的节点
    all_nodes = set()
    for (r, c) in obstacle_positions:
        # 生成所有缓冲节点
        nodes = offsets + [r, c]
        # 过滤超出边界的节点
        valid_nodes = nodes[
            (nodes[:, 0] >= 0) & (nodes[:, 0] < rows) & 
            (nodes[:, 1] >= 0) & (nodes[:, 1] < cols)
        ]
        # 添加到待移除集合
        
        occ3[valid_nodes[:, 0], valid_nodes[:, 1]] = 255
    
    G=grid_to_graph_vectorized(occ3)  #将占用栅格转换为无向图，使用 NumPy 向量化操作加速。
    return G,occ3

 #输入startpts, goalpts为车本地坐标系
 #occ2 occ2,X轴向前指向车头，Y轴向上.右下角为原点
def find_path(occ2, startpts, goalpts, resolution=0.5,x_range=(-100, 100), y_range=(-100, 100)): #输入startpts, goalpts为车本地坐标系
    # 创建图（默认障碍物缓冲距离为 0.5 米）
    
    startpts2 = vehicle_to_occupancy(startpts, resolution, x_range, y_range) #将以车中心为原点的 XY 坐标转换为 OCC 图的坐标。
    goalpts2 =  vehicle_to_occupancy(goalpts,  resolution, x_range, y_range)
    startpts2 = startpts2[0]
    goalpts2 = goalpts2[0]

    #输入occ2为为车辆本地OCC坐标(左下角为原点，X轴向前指向车头，Y轴向上)
    # startpts2,endpts2 为车辆本地OCC坐标
    filename_startwith = 'out/occVehLoc_'
    visualize_save_occ_as_vehlococcplt(occ2, startpts2,goalpts2,filename_startwith)


  
    
    #输入occ2为为车辆本地OCC坐标(左下角为原点，X轴向前指向车头，Y轴向上)
    #startpts2,endpts2 为车辆本地OCC坐标
    #注意networkx的为(左上角为原点，X轴向前指向车头，Y轴向下)
    G,occ3= create_graph_from_occupancy_grid(occ2,resolution, x_range, y_range,obstacle_buffer=1)
    occ3Tmp = occ3.copy()
 
    filename_startwith = 'out/occVehLoc_obstaclebuffer_'
    visualize_save_occ_as_vehlococcplt(occ3Tmp, startpts2,goalpts2,filename_startwith)
    


    startpts3 = vehicleOCC_to_netwrokxGrid(startpts2,occ2)
    goalpts3 = vehicleOCC_to_netwrokxGrid(goalpts2,occ2)
    print("vehicle local OC,原点左上角,Y轴向上",startpts2,goalpts2)
    print("netwrokxGrid,原点左上角,Y轴向下:",startpts3,goalpts3)
    
    #print(G.nodes)
    startpts3 = (startpts3[0],startpts3[1])
    goalpts3 = (goalpts3[0],goalpts3[1])
    # 检查起点和终点是否有效
    if startpts3 not in G :
        raise ValueError("起点在障碍物或缓冲区内，无法规划路径。")
    if  goalpts3 not in G:
        raise ValueError("终点在障碍物或缓冲区内，无法规划路径。")

    # 使用 A* 算法进行路径规划
    try:
        #path = nx.astar_path(G, startpts3, goalpts3, heuristic=lambda a, b: np.linalg.norm(np.array(a) - np.array(b)))
        path = nx.dijkstra_path(G, startpts3, goalpts3)
        #path = nx.shortest_path(G, start, goal)
        #path = nx.bidirectional_dijkstra(G, start, goal)
        #path = nx.astar_path(G, startpts3, goalpts3, heuristic=lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1]))

        path2 = netwrokxGrid_to_vehicleOCC(path,occ3)
        print(f"找到路径，路径长度: {len(path2)}")
        
        return path2,G,occ3,startpts2,goalpts2
    except nx.NetworkXNoPath:
        print("无法找到路径。")
        return None, None, None, None, None


def lidar_data_callback(data,queue):
    queue.put(data)

def process_lidar_data2(point_cloud,lidar,vehicle,birdseye_image):
    
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)])) #处于lidar坐标系

    points = np.array([data['x'], data['y'], data['z']]).T    
    
    points_lidar2world = lidar_to_world(lidar, points)

    vehicle_transform = vehicle.get_transform()
    points_vehicleaxes = world_to_vehicle_points(vehicle_transform, points_lidar2world)

    objIndexs = np.array(data['ObjIdx'])
    labels = np.array(data['ObjTag'])
    
    #print(f"\npoints.shape is {points.shape}")
    #print(f"points_lidar2world.shape is {points_lidar2world.shape}")
    

    #生成和绘制车中心为原点的OCCupancy Grid Map
    birdseye_map = birdseye_image.copy()

    birdseye_points = points_vehicleaxes[:,:2]  # 只取x和y坐标

    lidar_points = birdseye_points
    
    labels2 = labels.copy()  # 复制标签数组以避免修改原始数据
    mask = (labels == 6) | (labels == 7) | (labels == 14) | (labels == 22) | (labels == 8) 
    labels2[mask] = 0  # 将道路、路面、地面和地形和人型到的标签设置为0 可通行
    labels2[~mask] = 1  # 其余设置为1（障碍）
    
    #目标车区域设定为0
    bounding_box = vehicle.bounding_box
    bbox_extent = bounding_box.extent
    mask = (lidar_points[:, 0] < bbox_extent.x) & (lidar_points[:, 1] < bbox_extent.y) & (lidar_points[:, 0] > -bbox_extent.x) & (lidar_points[:, 1] > -bbox_extent.y)
    labels[mask] = 6 
    labels2[mask] = 0 

    resolution=0.5
    x_range=(-100, 100)
    y_range=(-100, 100)

    occ = lidar_to_occupancy(lidar_points, labels2, resolution,x_range, y_range)

    occ2 = occ.astype(np.uint8)*255  # 存储类别 ID
    occ2 = np.flipud(occ2)  # 上下翻转图像数据,保证occ2,X轴向前指向车头，Y轴向上.右下角为原点

    ##显示和保存坐标，坐标轴和车辆位置，轮廓
    filename_startwith = 'out/occ2_lidar_image_'
    originpts = [0,0]#

    #输入点为车辆本地坐标系，0,0 为车中心，X轴向前指向车头，Y轴向上。转换为车辆本地OCC坐标左下角为原点，X轴向前指向车头，Y轴向上
    visualize_save_occ_as_vehlocplt(occ2,vehicle,originpts,filename_startwith,resolution, x_range, y_range)
    
    # 示例起点和终点
    start = np.array([0,0]) #车辆本地坐标系，0,0 为车中心，X轴向前指向车头，Y轴向上
    goal = np.array([20,5]) #车辆本地坐标系，0,0 为车中心，X轴向前指向车头，Y轴向上
  

    #输入是车本地坐标系，start, goal在内部转为网格坐标系统
    #输入occ矩阵是车本地坐标local系,左下角为原点，X轴向前指向车头，Y轴向上(已经经过翻转)
    # 输出的path是networkx网格坐标系(等同opencv图像坐标系)
    # startpts3,goalpts3 为networkx网格坐标系(等同opencv图像坐标系)
    # startpts2,goalpts2 为车本地网格坐标系
    # 输出occ矩阵是车本地坐标local系,左下角为原点，X轴向前指向车头，Y轴向上(已经经过翻转)
    path2,G,occ3,startpts2,goalpts2 = find_path(occ2, start, goal) #输入是车本地坐标系，在内部转为网格坐标系统，输出的path是车本地坐标系
    #内部转为networkx网格坐标系(等同opencv图像坐标系)

    # 可视化结果并保存图像
    if path2 is not None:
        print("path2 is not Non")
        filename_startwith = 'out/occ3_pathAstar_'
        visualize_save_occpath_as_vehlococcplt(occ3,path2,startpts2,goalpts2,filename_startwith)
        time.sleep
        
  
    



    #生成绘制车中心为原点的鸟瞰图
    for i, point in enumerate(birdseye_points):
        point[1] = -point[1]  # 翻转y轴，使得y轴向下
        x, y = int(point[0] * 10 + 1000), int(point[1] * 10 + 1000)  # 缩放并平移到图像中心
        
        label = int(labels[i])  # 确保标签是整数
        

        if label ==  6 or   label ==  7:
            color = (0, 255, 0)  # 绿色表示道路和路面

        elif label ==  14 or label == 22 :  #    # Ground or Terrain               
            color = (0, 200, 0) 
        elif label == 8: #Sidewalk
            color = (255, 0, 0)
        elif label == 0 or label == 10:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255) 
       
        cv2.circle(birdseye_map, (x, y), 2, color, -1)
    
    cv2.circle(birdseye_map, (1000, 1000), 2, (255, 0, 0), -1)

    bounding_box = vehicle.bounding_box
    bbox_extent = bounding_box.extent
    pos1 = int(-bbox_extent.x * 10 + 1000), int(-bbox_extent.y * 10 + 1000)
    pos2 = int(bbox_extent.x * 10 + 1000), int(bbox_extent.y * 10 + 1000)

    cv2.rectangle(birdseye_map, pos1,pos2, (0, 255, 0), 2)

    filename = 'out/birdseye_image_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
    cv2.imwrite(filename, birdseye_map)

'''废弃
# 将3D点投影到车辆中心为原点的2D平面
def project_to_birdseye(points, vehicle_transform,lidar_2_world):

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    local_lidar_points = np.array(points).T
    #print(local_lidar_points.shape)

    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[local_lidar_points, [np.ones(local_lidar_points.shape[1])]]


    birdseye_points = []
    world_points = np.dot(lidar_2_world, local_lidar_points)
    world_points = world_points.T
    #print("World points shape:", world_points.shape)

    for point in world_points:
        # 转换到车辆坐标系
        relative_point = point[0:3] - np.array([vehicle_transform.location.x, vehicle_transform.location.y, 0])
        birdseye_points.append((relative_point[0], relative_point[1]))
    #print("Birdseye points shape:", len(birdseye_points))
    return birdseye_points
def process_lidar_data(data,vehicle,lidar,birdseye_image):
    points = []
    labels = []
    objIndexs = []
    object_points = {}

    # 收集点云数据和物体标签
    for detection in data:
        #if detection.object_tag != 0:  # 忽略背景
            #print(detection)
        point = np.array([detection.point.x, detection.point.y, detection.point.z])
        label = str(detection.object_tag)
        objIndex = detection.object_idx
        points.append(point)
        labels.append(label)
        objIndexs.append(objIndex)
        if label not in object_points:
            object_points[label] = []
        object_points[label].append(point)
 
    vehicle_transform = vehicle.get_transform()
    lidar_2_world = lidar.get_transform().get_matrix()
    birdseye_points = project_to_birdseye(points,vehicle_transform,lidar_2_world)
    
    points_vehicleaxes = world_to_vehicle_points(vehicle_transform, points)
    # 绘制鸟瞰图
    birdseye_map = birdseye_image.copy()
    #print("Processing {} points".format(len(birdseye_points)))
    #print("Processing {} points".format(len(points)))

    # 绘制每个物体的边界框和类别
   
    """
    
    for label, obj_points in object_points.items():
        label = int(label)  # 确保标签是整数
        #print("label,label_name, obj_points:", label,label_name[label], len(obj_points))
        obj_birdseye_points = project_to_birdseye(obj_points, vehicle_transform,lidar_2_world)
        obj_birdseye_points = np.array(obj_birdseye_points) * 1 + 250  # 缩放并平移到图像中心
        obj_birdseye_points = obj_birdseye_points.astype(int)

        # 计算边界框
        x_min, y_min = np.min(obj_birdseye_points, axis=0)
        x_max, y_max = np.max(obj_birdseye_points, axis=0)

        # 绘制边界框
        #cv2.rectangle(birdseye_map, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # 在边界框内标注类别
        #cv2.putText(birdseye_map, label_name[label], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    """
    # 绘制所有点
    for i, point in enumerate(birdseye_points):
        x, y = int(point[0] * 10 + 1000), int(point[1] * 10 + 1000)  # 缩放并平移到图像中心
        objIndex= int(objIndexs[i])
        label = int(labels[i])  # 确保标签是整数
        color= (label_color[label][0],label_color[label][1], label_color[label][2])
        color = (int(color[0]), int(color[1]), int(color[2]))

        if label ==  6 or   label ==  7:
            color = (0, 255, 0)  # 绿色表示道路和路面

        elif label ==  14 or label == 22 :  #    # Ground or Terrain               
            color = (0, 200, 0) 
        elif label == 8:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255) 
       
        cv2.circle(birdseye_map, (x, y), 2, color, -1)

    # 绘制车辆位置
    #cv2.circle(birdseye_map, (1000, 1000), 10, (255, 0, 0), -1)
  
    # 显示鸟瞰图
   #cv2.imshow('Birdseye View', birdseye_map)
   # cv2.waitKey(1)
    filename = 'out/birdseye_image_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
    cv2.imwrite(filename, birdseye_map)

'''

# 主函数
def main():
    pygame.init()
    pygame.font.init()
    world = None
    lidar = None
 
   
  
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)

    available_maps = client.get_available_maps()
    print(available_maps) 
    client.load_world("Town04")

    world = client.get_world()
    clock = pygame.time.Clock()

    ##设置同步模式
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)


    ################################
    ##定位观察者位置
    spectator = world.get_spectator()
    location = carla.Location()
    location.x = 284
    location.y = -172
    location.z = 0.25
    angle = 0
    spectator.set_transform(get_transform(location, angle,height = 80,pitchAngle = 90))
    
 

    world = client.get_world()
    clock = pygame.time.Clock()




            ################################
    ##把车放在停车场，不急于waypoint
    ###1 目的停车位
    vehicle = None
    location = carla.Location(
        x=280.5,  # 示例X范围
        y=-223.2,  # 示例Y范围
        z=0.26  # 固定Z值，考虑路面高度
    )
    rotation = carla.Rotation(
        pitch=0,  # 默认俯仰角
        yaw=0,  # 随机偏航角
        roll=0   # 默认滚转角
    )
    pos =  carla.Transform(location, rotation)
    posPark1 = pos 

    ###2: 生成目的车开始位置
    location = carla.Location(
        x=285.5-15,  # 示例X范围
        y=-172.2,  # 示例Y范围
        z=0.26  # 固定Z值，考虑路面高度
    )
    rotation = carla.Rotation(
        pitch=0,  # 默认俯仰角
        yaw=0,  # 随机偏航角
        roll=0   # 默认滚转角
    )
    pos =  carla.Transform(location, rotation)
    posGenVehile = pos
    
    blueprint = world.get_blueprint_library().filter('model3')[0]
    vehicle = world.try_spawn_actor(blueprint, posGenVehile)



    # 在车辆上方放置语义激光雷达
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic') 
    lidar_bp.set_attribute('upper_fov', str(15.0))
    lidar_bp.set_attribute('lower_fov', str(-25.0))
    lidar_bp.set_attribute('channels', str(64.0))
    lidar_bp.set_attribute('range', str(100.0))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / 0.05))
    lidar_bp.set_attribute('points_per_second', str(500000))

    lidar_transform = carla.Transform(carla.Location(x=1.5,y=0,z=2.0))

    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)



    # 在车辆上方放置语义相机
    camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 创建鸟瞰图
    birdseye_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
    
     # This (4, 4) matrix transforms the points from lidar space to world space.
    


    lidar_queue = Queue()
    lidar.listen(lambda data: lidar_data_callback(data, lidar_queue))

    try:
        while True:
            world.tick()
            
            try:
                # Get the data once it's received.
               
                lidar_data = lidar_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            #process_lidar_data(lidar_data, lidar,vehicle, birdseye_image)
            process_lidar_data2(lidar_data,lidar,vehicle,birdseye_image)
            control= carla.VehicleControl(throttle=0.2, steer=-0.2, brake=0.0, hand_brake=False, reverse=False)
            control.manual_gear_shift = False
            vehicle.apply_control(control)
    finally:
        lidar.destroy()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()






plt.title("中文标题", fontproperties='SimHei')
