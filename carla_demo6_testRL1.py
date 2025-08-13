#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

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
import open3d as o3d
import cv2
from matplotlib import cm

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

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


def show_save_lidar(vis,point_list,frame):
    
    if  vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270,
            visible=False)  # 关键：设置 visible=False 以进行离屏渲染)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True
        return vis
    
    if frame == 1:
            vis.add_geometry(point_list)
   #print(point_list)
    vis.update_geometry(point_list)
    vis.poll_events()
    vis.update_renderer()
    filename = "out/carla_lidar_frame_{:04d}.jpg".format(frame)
    #vis.capture_screen_image(filename, do_render=True)#保存为文件，根据需要开启

    return vis

import matplotlib.pyplot as plt
def semantic_lidar_callback(point_cloud, point_list,frame):
    
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    #print(point_cloud)
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)

    points_xyz = np.vstack((data['x'], data['y'], data['z'])).T
    obj_tags = data['ObjTag']

    # 可视化
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2],
                         c=obj_tags, cmap='tab20')  # 使用不同的颜色表示不同的对象类别
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.title('Semantic LIDAR Point Cloud')
    plt.savefig('out/semantic_lidar%06d.png' %frame)  # 保存图片
    plt.close(fig)  # 关闭图形以释放内存



#跟踪算法
#https://blog.csdn.net/WaiNgai1999/article/details/132062188
def pure_pursuit(tar_location, v_transform):
    L = 2.875
    yaw = v_transform.rotation.yaw * (math.pi / 180)
    x = v_transform.location.x - L / 2 * math.cos(yaw)
    y = v_transform.location.y - L / 2 * math.sin(yaw)
    dx = tar_location.x - x
    dy = tar_location.y - y
    ld = math.sqrt(dx*dx + dy*dy)
    alpha = math.atan2(dy, dx) - yaw
    delta = math.atan(2 * math.sin(alpha) * L / ld) * 180 / math.pi
    steer = delta/90
    if steer > 1:
        steer = 1
    elif steer < -1:
        steer = -1
    return steer


# 观察者位置计算,并且高处往下看，d=为距离车辆的直线距离，height为高度
def get_transform(vehicle_location, angle, d=6.4,height =2,pitchAngle = -15):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), height) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-pitchAngle))

##图像传感器回调函数，调用opencv                
def img_process(image):
    pass
##深度图像传感器回调函数，调用opencv ，直接保存               
def depth_img_process(image):
    pass
    #image.convert(carla.ColorConverter.Depth)
    #image.convert(carla.ColorConverter.LogarithmicDepth)
    #image.save_to_disk('out/depth_%06d.jpg' % image.frame)
    
   
##语法分割图像传感器回调函数，调用opencv ，直接保存               
def semantic_img_process(image):
    pass
    #image.convert(carla.ColorConverter.Depth)
    #image.save_to_disk('out/semantic_%06d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette)

##instance分割图像传感器回调函数，调用opencv ，直接保存               
def instance_img_process(image):
    #image.convert(carla.ColorConverter.Depth)
    image.save_to_disk('out/instance_%06d.jpg' % image.frame)

def game_loop():
    """ Main loop for agent"""

    pygame.init()
    pygame.font.init()
    world = None
    lidar = None
    vis = None
    frame = 0
   
    
    try:
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
        #settings.synchronous_mode = True
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
        

        #################################
        ##加RGB传感器
        bp_lib = world.get_blueprint_library()
        
        camerargb = bp_lib.filter("sensor.camera.rgb")[0]
        camerargb.set_attribute('image_size_x', '800')
        camerargb.set_attribute('image_size_y', '600')
        camerargb.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        camerargb.set_attribute('sensor_tick', '1.0')    
        camera = world.spawn_actor(
            blueprint=camerargb,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle)

        camera.listen(lambda data: img_process(data))
        ###################################################
        ##加深度相机，sensor.camera.depth
        depth_camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_camera_bp.set_attribute('image_size_x', '800')
        depth_camera_bp.set_attribute('image_size_y', '600')
        depth_camera_bp.set_attribute('fov', '110')
        depth_camera_bp.set_attribute('sensor_tick', '1.0')
        depth_camera_transform = carla.Transform(carla.Location(x=1.6, z=1.6))
        depth_camera = world.spawn_actor(depth_camera_bp, depth_camera_transform, attach_to=vehicle)

        depth_camera.listen(lambda data: depth_img_process(data))

        ###################################################
        ##加语法分割相机。Semantic segmentation
        semantic_camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        semantic_camera_bp.set_attribute('image_size_x', '800')
        semantic_camera_bp.set_attribute('image_size_y', '600')
        semantic_camera_bp.set_attribute('fov', '110')
        semantic_camera_bp.set_attribute('sensor_tick', '1.0')
        semantic_camera_transform = carla.Transform(carla.Location(x=1.6, z=1.6))
        semantic_camera = world.spawn_actor(semantic_camera_bp, semantic_camera_transform, attach_to=vehicle)

        semantic_camera.listen(lambda data: semantic_img_process(data))


        ###################################################
        ##加Instance segmentation camera。Instance segmentation camera,0.9.11版本没有，可以用于安装在高处和交通灯上
        #instance_camera_bp = world.get_blueprint_library().find('sensor.camera.instance_segmentation')
        #instance_camera_bp.set_attribute('image_size_x', '800')
        #instance_camera_bp.set_attribute('image_size_y', '600')
        #instance_camera_bp.set_attribute('fov', '110')
        #instance_camera_bp.set_attribute('sensor_tick', '1.0')
        #instance_camera_transform = carla.Transform(carla.Location(x=1.6, z=1.6))
        #instance_camera = world.spawn_actor(instance_camera_bp, instance_camera_transform, attach_to=vehicle)

        #instance_camera.listen(lambda data: instance_img_process(data))

      
      

        ############
        # 加ray_cast_semantic
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic') 
        lidar_bp.set_attribute('upper_fov', str(15.0))
        lidar_bp.set_attribute('lower_fov', str(-25.0))
        lidar_bp.set_attribute('channels', str(64.0))
        lidar_bp.set_attribute('range', str(100.0))
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / 0.05))
        lidar_bp.set_attribute('points_per_second', str(500000))

        lidar_transform = carla.Transform(carla.Location(x=1.5,y=0,z=2.4))

        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        point_list = o3d.geometry.PointCloud()
        lidar.listen(lambda data: semantic_lidar_callback(data, point_list,frame))
        

        ############
        frame = 0
        vis = None
        while True:
         
            world.tick()#同步模式
            world.render(display)
            #clock.tick_busy_loop(20) 
            # As soon as the server is ready continue!
            #if not world.wait_for_tick(10.0):
            #    continue
           
            
            control= carla.VehicleControl(throttle=0.2, steer=-0.5, brake=0.0, hand_brake=False, reverse=False)
            control.manual_gear_shift = False
            
            control.steer = pure_pursuit(posPark1.location, vehicle.get_transform())
            vehicle.apply_control(control)

            vis= show_save_lidar(vis,point_list,frame)
            frame = frame+1

            if frame > 600:
                break
       


    finally:
     
        pygame.quit()
        lidar.destroy()
        camera.destroy()
        depth_camera.destroy()    
        #vis.destroy_window()
        

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
import subprocess
import time
import os





def main():

    # --- 配置 ---
    game_loop()



if __name__ == '__main__':
    main()


