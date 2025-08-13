#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

#修改来源自vehcle_gallery.py的代码，然后最简化代码，删除自动化代码部分


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import math
import random
import time
import cv2
import numpy as np


# 观察者位置计算,并且高处往下看，d=为距离车辆的直线距离，height为高度
def get_transform(vehicle_location, angle, d=6.4,height =2,pitchAngle = -15):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), height) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-pitchAngle))

def img_process(image):

    image.save_to_disk('_out/%06d.png' % image.frame)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA格式
    array = array[:, :, :3]  # 去除Alpha通道，保留RGB
    #array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)  # CARLA使用RGB，OpenCV默认BGR,OPENCV高版本，这条代码可以注释掉。

    # 显示图像
    ##cv2.imshow('CARLA RGB Sensor', array)
    ##cv2.waitKey(1)
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


def main():

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    
    map = world.get_map()

    vehicle_blueprints = world.get_blueprint_library().filter('vehicle')
    vehicle_1 = world.get_blueprint_library().filter('model3')[0]
    
    location = random.choice(world.get_map().get_spawn_points()).location
    #location = carla.Location(x=0,y=0,z=0.26)
    transform = carla.Transform(location, carla.Rotation(yaw=0.0))

    vehicle = world.spawn_actor(vehicle_1, transform)

    spectator = world.get_spectator()



  
    ##测试1
   

    print(vehicle.type_id)
    print(location)
    angle = 0
    vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0, brake=0.0, hand_brake=False, reverse=False))

    # 观察者位置计算,并且高处往下看，d=为距离车辆的直线距离，height为高度
    spectator.set_transform(get_transform(vehicle.get_location(), angle,height = 100,pitchAngle = 70))

    #time.sleep(5)  # 等待10秒钟，观察车辆行驶情况

    
         
        



    ##测试2
    camerargb = world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    camerargb.set_attribute('image_size_x', '800')
    camerargb.set_attribute('image_size_y', '600')
    camerargb.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    camerargb.set_attribute('sensor_tick', '1.0')    

    transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    camerargbsensor = world.spawn_actor(camerargb, transform, attach_to=vehicle)
    camerargbsensor.listen(lambda data: img_process(data))


    while True:
        spectator.set_transform(get_transform(vehicle.get_location(), angle,height = 100,pitchAngle = 70))

        # Nearest waypoint in the center of a Driving or Sidewalk lane.
        waypoint01 = map.get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=carla.LaneType.Driving)

        #Nearest waypoint 2 meter 
        #away from the vehicle in the center of a Driving lane.
        v_trans = vehicle.get_transform()
        waypoints = waypoint01.next(6.0)
        waypoint02 = waypoints[0]
        tar_loc = waypoint02.transform.location
        steer = pure_pursuit(tar_loc, v_trans)
        vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=steer, brake=0.0, hand_brake=False, reverse=False))
        if cv2.waitKey(1) == 27:
            break


 
    cv2.destroyAllWindows()
    camera.destroy()
    vehicle.destroy()
    
if __name__ == '__main__':

    main()
