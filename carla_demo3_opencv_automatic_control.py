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

import cv2
from lanedetectionOpencv  import  process_image,simple_process_image 

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
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

import time




# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world,  args):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
       
        self.player = None
        self._actor_filter = args.filter
        self.restart(args)


    def restart(self, args):
        """Restart the world"""
       
        
        if args.seed is not None:
            random.seed(args.seed)

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        print("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()[0:1]#只取前10个点
                    
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

            
            print(f"只取前{len(spawn_points)}个物体")
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
    
        
       


  

    

    def destroy(self):
        """Destroys all actors"""
        actors = [self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

##图像传感器回调函数，调用opencv                
def img_process(image):

    image.save_to_disk('out/input%06d.jpg' % image.frame)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA格式
    array = array[:, :, :3]  # 去除Alpha通道，保留RGB
    #array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)  # CARLA使用RGB，OpenCV默认BGR,OPENCV高版本，这条代码可以注释掉。

    result = simple_process_image(array)  # 调用图像处理函数
    # 显示图像
    #cv2.imshow('CARLA RGB Sensor', array)
    #cv2.waitKey(100)
    if result is not None:
        str1 = 'out/output%06d.jpg' % image.frame
        cv2.imwrite(str1, result)

def game_loop(args):
    """ Main loop for agent"""

    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21

    try:
        client = carla.Client("127.0.0.1", args.port)
        client.set_timeout(4.0)

     

        world = World(client.get_world(), args)
  
        if args.agent == "Roaming":
            agent = RoamingAgent(world.player)
        elif args.agent == "Basic":
            agent = BasicAgent(world.player)
            spawn_point = world.map.get_spawn_points()[0]
            agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)

            spawn_points = world.map.get_spawn_points()
            random.shuffle(spawn_points)

            if spawn_points[0].location != agent.vehicle.get_location():
                destination = spawn_points[0].location
            else:
                destination = spawn_points[1].location

            agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        clock = pygame.time.Clock()


        #################################
        bp_lib = client.get_world().get_blueprint_library()
        
        camerargb = bp_lib.filter("sensor.camera.rgb")[0]
        camerargb.set_attribute('image_size_x', '800')
        camerargb.set_attribute('image_size_y', '600')
        camerargb.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        camerargb.set_attribute('sensor_tick', '1.0')    

        camera = client.get_world().spawn_actor(
            blueprint=camerargb,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=world.player)

         
        camera.listen(lambda data: img_process(data))

        ################################

        while True:
            clock.tick_busy_loop(60)
            # As soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue

            if args.agent == "Roaming" or args.agent == "Basic":
                
                # as soon as the server is ready continue!
                world.world.wait_for_tick(10.0)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
            else:
                agent.update_information()
               

                # Set new destination when target has been reached
                if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints and args.loop:
                    agent.reroute(spawn_points)
                    tot_target_reached += 1
                elif len(agent.get_local_planner().waypoints_queue) == 0 and not args.loop:
                    print("Target reached, mission accomplished...")
                    break

                speed_limit = world.player.get_speed_limit()
                agent.get_local_planner().set_speed(speed_limit)

                
                control = agent.run_step()
                #print(control)
                
                #0：获取当前ego车辆的位置和航点
                ego_vehicle_loc = world.player.get_location()
                ego_vehicle_wp = world.map.get_waypoint(ego_vehicle_loc)

                # 1: 画前面0.1米的waypoints
                distance = 0.1  # 每个航点之间的间隔距离
                waypoints = ego_vehicle_wp.next(distance)
                life_time = 0
                for waypoint in waypoints:
                   client.get_world().debug.draw_string(
                        waypoint.transform.location, 
                        'O',  # 标记符号
                        draw_shadow=False,
                        color=carla.Color(r=0, g=255, b=0),  # 绿色标记
                        life_time=life_time,  # 显示时间（秒）
                        persistent_lines=True
                    )
                
                # 2：获取waypoints的类型
                #print(f"incoming_waypoint.is_junction:{incoming_waypoint.is_junction}")
                

                # 3:画全局waypoints
                life_time = 10
                global_route_trace = agent.get_global_route_trace()
                for i in range(len(global_route_trace) - 1):
                    waypoint, road_option = global_route_trace[i]
                    client.get_world().debug.draw_string(
                        waypoint.transform.location, 
                        'O',  # 标记符号
                        draw_shadow=False,
                        color=carla.Color(r=255, g=0, b=0),  # 绿色标记
                        life_time=life_time,  # 显示时间（秒）
                        persistent_lines=True
                    )

                # 4:画localplan当前的目标点
                
                life_time = 10
                waypoint = agent._local_planner.target_waypoint

                client.get_world().debug.draw_string(
                    waypoint.transform.location, 
                    'O',  # 标记符号
                    draw_shadow=False,
                    color=carla.Color(r=0, g=0, b=255),  # 绿色标记
                    life_time=life_time,  # 显示时间（秒）
                    persistent_lines=True
                )

                
                world.player.apply_control(control)

             


                

    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: cautious) ',
        default="cautious")
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()
    game_loop(args)



if __name__ == '__main__':
    main()
