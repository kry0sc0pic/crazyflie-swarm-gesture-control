from cflib.crazyflie.swarm import CachedCfFactory,Swarm
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.positioning.motion_commander import MotionCommander
import cflib.crtp
import threading
import time
import math
import typing
from utils import wait_for_position_estimator, get_full_state_estimate, calculate_total_yaw

class SwarmManager:
    def __init__(self,URIS,dry_run=False,use_low_level=False) -> None:
        self._uris = URIS
        self.full_state_estimates: typing.Dict[str,list] = {}
        self.hl_commanders: typing.Dict[str,PositionHlCommander] = {}
        self.motion_commanders: typing.Dict[str,MotionCommander] = {}
        self.state = 'not_setup'
        self.swarm: Swarm = Swarm(URIS,factory=CachedCfFactory('./cache'))
        self.dry_run = dry_run
        self.use_low_level = use_low_level
        self.desired_height = 0.5
        self.height_lock = threading.Lock()
        self.circle_setup = False    
        self.circle_state: typing.Literal['inactive','active'] = 'inactive'

    def stop_all(self):
        if self.state in ['flying']:
            print("[SwarmManager] Stopping All Drones")
            if self.use_low_level:
               if not self.dry_run:
                   self.swarm.parallel_safe(lambda scf: self._get_commander_dict().get(scf.cf.link_uri).stop())
            self.circle_state = 'inactive'
        else:

            print("[SwarmManager] Error: Drones are not flying")

    def _go_to(self,scf,x,y,z) -> None:
        # return
        uri = scf.cf.link_uri
        commander: PositionHlCommander = self.hl_commanders.get(uri)
        print(f"[SwarmManager] Moving {uri} to {x},{y},{z}")
        commander.go_to(x=x,y=y,z=z)
        print(f"[SwarmManager] Updating Position for {uri} to {x},{y},{z}")
        self.full_state_estimates[uri][0] = x
        self.full_state_estimates[uri][1] = y
        self.full_state_estimates[uri][2] = z
    
    def _create_motion_commander(self,scf):
        uri = scf.cf.link_uri
        position = self.full_state_estimates[uri]
        self.motion_commanders[uri] = MotionCommander(scf,
                                                      default_height=position[2])
        
    def _enable_INDI(self,scf):
        scf.cf.param.set_value('stabilizer.controller','1')

    def _create_hl_commander(self,scf):
        uri = scf.cf.link_uri
        position = self.full_state_estimates[uri]
        self.hl_commanders[uri] = PositionHlCommander(
            scf,
            x=position[0],
            y=position[1],
            z=position[2],
            controller=PositionHlCommander.CONTROLLER_PID,
        )
        

    def _get_commander_dict(self) -> dict:
        if self.use_low_level:
            return self.motion_commanders
        else:
            return self.hl_commanders
        
    def _set_height(self,scf,x,y,z):
        if(self.use_low_level):
            commander = self.motion_commanders.get(scf.cf.link_uri)
            if commander is None:
                print("[SwarmManager] Error: Motion Commander not found")
            else:
                curr_height = self.full_state_estimates.get(scf.cf.link_uri)[2]
                delta = z - curr_height
                if delta > 0:
                    commander.up(abs(delta))
                    self.full_state_estimates[scf.cf.link_uri][2] = z
                elif delta < 0:
                    commander.down(abs(delta))
                    self.full_state_estimates[scf.cf.link_uri][2] = z
        else:
            self._go_to(scf,x,y,z)
    
    def set_position(self,scf,position):
        self._go_to(scf,position)

    def _save_full_state_estimates(self,scf):
        uri = scf.cf.link_uri
        self.full_state_estimates[uri] = get_full_state_estimate(scf)

    def setup(self) -> bool:
        if(self.dry_run):
            print("[SwarmManager] Dry Run Setup")
            
        print("[SwarmManager] Initialising CRTP Drivers")
        if(not self.dry_run):
            cflib.crtp.init_drivers()
        print("[SwarmManager] Opening Links")
        if(not self.dry_run):
            self.swarm.open_links()
        print("[SwarmManger] Waiting for Position Estimation")
        if(not self.dry_run):
            self.swarm.parallel(wait_for_position_estimator)

            
        print("[SwarmManager] Saving Estimated Positions")
        if not self.dry_run:
            self.swarm.parallel_safe(self._save_full_state_estimates)
        print("[SwarmManager] Creating Commanders")
        if(self.use_low_level):
            print("[SwarmManager] Creating Motion Commanders")
            if(not self.dry_run):
                self.swarm.parallel(self._create_motion_commander)
        else:
            print("[SwarmManager] Creating High Level Commanders")
            if(not self.dry_run):
                self.swarm.parallel(self._create_hl_commander)
        
        print("[SwarmManager] Updating state to not_flying")
        self.state = 'not_flying'
        print("[SwarmManger] Setup Complete")   

    def cleanup(self) -> bool:
        
        if self.state in ["not_flying"]:
            print("[SwarmManager] Closing Links")
            if(not self.dry_run):
                self.swarm.close_links()

        if self.state in ["flying"]:
            print("[SwarmManager] Error: Drone is still flying")
            return
        
        print("[SwarmManager] Updating state to not_setup")
        self.state = 'not_setup'

        print("[SwarmManger] Cleanup Complete")
    
    def takeoff(self,height: float = 0.5) -> None:
        if self.state in ['not_flying']:
            print("[SwarmManager] Taking Off")
            if(not self.dry_run):
                self.swarm.parallel_safe(lambda scf: self._get_commander_dict().get(scf.cf.link_uri).take_off(height))
                for uri in self._uris:
                    self.full_state_estimates[uri][2] = height 
                time.sleep(2)
            print("[SwarmManager] Updating state to flying")
            self.state = 'flying'
            
    def land(self) -> None:
        if self.state in ['flying']:
            print("[SwarmManager] Landing")
            if(not self.dry_run):
                self.swarm.parallel_safe(lambda scf: self._get_commander_dict().get(scf.cf.link_uri).land())
            print("[SwarmManager] Updating state to not_flying")
            self.state = 'not_flying'

    def set_multiple_heights(self,height_map: typing.Dict[str,float]) -> None:
        if self.state in ['flying']:
            arg_height_map = {}
            for uri,height in height_map.items():
                if uri not in self._uris:
                    print(f"[SwarmManager] Error: {uri} not in URIS")
                    continue
                
                x,y = self.full_state_estimates.get(uri)[0],self.full_state_estimates.get(uri)[1]
                arg_height_map[uri] = (x,y,height)
                print(f"[SwarmManager] Moving {uri} to {x},{y},{height}")
            if not self.dry_run:
                self.swarm.parallel_safe(self._set_height,args_dict=arg_height_map)

    def set_uniform_height(self,height):
        if self.state in ['flying']:
            desired_positions = {}
            for k,v in self.full_state_estimates.items():
                desired_positions[k] = (v[0],v[1],height)
            print(f"[SwarmManger] Moving to desired height {height}")
            if(not self.dry_run):
                self.swarm.parallel_safe(self._set_height,args_dict=desired_positions)

        else:
            print("[SwarmManager] Error: Drones are not flying")

    def _setup_circle(self,scf):
        uri = scf.cf.link_uri
        motion_commander = self._get_commander_dict().get(uri)
        full_state_estimate = self.full_state_estimates.get(uri)
        x,y = full_state_estimate[0],full_state_estimate[1]
        curr_yaw = full_state_estimate[3]
        total_yaw = calculate_total_yaw(x,y,curr_yaw)
        if total_yaw < 0:
            motion_commander.turn_right(abs(total_yaw))
            print(f"[SwarmManager] {uri} Turned Right by {total_yaw} degrees")
        else:
            motion_commander.turn_left(total_yaw) 
            print(f"[SwarmManager] {uri} Turned Left by {total_yaw} degrees")

    def _run_async_circle(self,scf,duration: float):
        uri = scf.cf.link_uri
        motion_commander = self._get_commander_dict().get(uri)
        full_state_estimate = self.full_state_estimates.get(uri)
        print(f"[{uri}] Calculating Circle")
        x,y = full_state_estimate[0],full_state_estimate[1]
        print(f"[{uri}] State: x:{x},y:{y}")
        radius = math.sqrt(x*x + y*y)
        print(f"[{uri}] Radius: {radius}")
        dist = 2 * math.pi * radius
        print(f"[{uri}] Distance: {dist}")
        print(f"[{uri}] {uri} Velocity: {dist/duration}")
        print(f"[{uri}] Running Circle")
        motion_commander.start_circle_left(radius,velocity=dist/duration)
        print(f"[SwarmManager] {uri} is running circle")

    def _run_circle(self,scf,duration: float):
        uri = scf.cf.link_uri
        motion_commander = self._get_commander_dict().get(uri)
        full_state_estimate = self.full_state_estimates.get(uri)
        
        print(f"[{uri}] Calculating Circle")
        x,y = full_state_estimate[0],full_state_estimate[1]
        print(f"[{uri}] State: x:{x},y:{y}")
        radius = math.sqrt(x*x + y*y)
        print(f"[{uri}] Radius: {radius}")
        dist = 2 * math.pi * radius
        print(f"[{uri}] Distance: {dist}")
        print(f"[{uri}] {uri} Velocity: {dist/duration}")
        print(f"[{uri}] Running Circle")
        motion_commander.circle_left(radius,velocity=dist/duration,angle_degrees=360)
        print(f"[SwarmManager] {uri} Ran Circle")
     
    def rotate_swarm(self,duration: float = 10,stage=-1,circle_mode='sync') -> None:
        print("[SwarmManager] Rotating Swarm")
        if self.state in ['flying']:
            if stage == 0 or stage == -1:
                print("[SwarmManager] Setting Up Circle")
                if not self.dry_run:
                    self.swarm.parallel_safe(self._setup_circle)
                self.circle_setup = True
            
            if stage == -1: time.sleep(0.3)
            
            if self.circle_state == 'inactive' and (stage == -1 or stage == 1):
                self.circle_state == 'active'
                if(circle_mode == 'sync'):
                    if not self.dry_run:
                        self.swarm.parallel_safe(self._run_circle,args_dict={k:(duration,) for k in self._uris})
                    else:
                        time.sleep(duration)
                    self.circle_state = 'inactive'
                else:
                    if not self.dry_run:
                        self.swarm.parallel_safe(self._run_async_circle,args_dict={k:(duration,) for k in self._uris})
        else:
            print("[SwarmManager] Error: Drones are not flying")
    