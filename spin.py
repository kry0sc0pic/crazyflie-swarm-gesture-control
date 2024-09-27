from utils.manager import SwarmManager
from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
import math
import random
import time
URIS = [
    "radio://0/80/2M/E7E7E7E7E5",
]
POSITION = None
TIME = 10
# manager = SwarmManager(URIS,dry_run=False)

init_drivers()





cf = Crazyflie(rw_cache='./cache')
with SyncCrazyflie(URIS[0],cf=cf) as scf:
    scf.cf.param.set_value('ring.effect', 9)
    scf.cf.param.set_value('ring.headlightEnable', 1)
    if POSITION is None:
        log_config = LogConfig(name='stateEstimate', period_in_ms=10)
        log_config.add_variable('stateEstimate.x', 'float')
        log_config.add_variable('stateEstimate.y', 'float')
        log_config.add_variable('stateEstimate.z', 'float')
        log_config.add_variable('stateEstimate.yaw','float')
        with SyncLogger(scf, log_config) as logger:
            scf.cf.param.set_value('ring.effect', 9)
            scf.cf.param.set_value('ring.headlightEnable', 255)
            
            for entry in logger:
                x = entry[1]['stateEstimate.x']
                y = entry[1]['stateEstimate.y']
                z = entry[1]['stateEstimate.z']
                yaw = entry[1]['stateEstimate.yaw']
                POSITION = (x,y,z,yaw)
                break
    
        

        
        # angle = 180 / (math.pi * radius)

        with MotionCommander(scf) as cmdr:
            # cmdr.take_off(0.5)
            time.sleep(3)
            if(yaw < 0):
                cmdr.turn_left(abs(yaw))
            else:
                cmdr.turn_right(abs(yaw))

            radius = math.sqrt(x*x + y*y)
            """
            2   |  1
                |
            ---------

            3   |  0
            """
            quadrant = 0
            if(x > 0 and y > 0):
                quadrant = 1
            elif(x < 0 and y > 0):
                quadrant = 2
            elif(x < 0 and y < 0):
                quadrant = 3

            local_yaw = (180/math.pi) * math.atan(abs(y)/abs(x) if quadrant % 2 != 0 else abs(x)/abs(y))
            abs_yaw = (quadrant*90) + local_yaw
            cmdr.turn_left(abs_yaw)
            time.sleep(0.5)
            cmdr.circle_left(radius,velocity=1.5,angle_degrees=360 * 4)
            time.sleep(1)
            # cmdr.down(0.5)
            # cmdr.start_circle_left(radius,velocity=1)
            # time.sleep(5)
            cmdr.stop()
            # cmdr.land()
        
        scf.cf.param.set_value('ring.effect', 0)
        scf.cf.param.set_value('ring.headlightEnable', 0)