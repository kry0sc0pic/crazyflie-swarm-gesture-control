from cflib.crazyflie.swarm import CachedCfFactory,Swarm
from cflib.positioning.position_hl_commander import PositionHlCommander
import cflib.crtp

class SwarmManager:
    def __init__(self,URIS,dry_run=False) -> None:
        self.positions = {}
        self.commanders = {}
        self.state = 'not_setup'
        self.swarm = Swarm(URIS,factory=CachedCfFactory('./cache'))
        self.dry_run = dry_run

    def _go_to(self,scf,position) -> None:
        uri = scf.cf.link_uri
        commander: PositionHlCommander = self.commanders.get(uri)
        print(f"[SwarmManager] Moving {uri} to {position}")
        commander.go_to(x=position[0],y=position[1],z=position[2])
        print(f"[SwarmManager] Updating Position for {uri} to {position}")
        self.positions[uri] = position

    def _create_commander(self,scf):
        
        uri = scf.cf.link_uri
        position = self.positions[uri]
        self.commanders[uri] = PositionHlCommander(
            scf,
            x=position[0],
            y=position[1],
            z=position[2],
            controller=PositionHlCommander.CONTROLLER_PID,
        )

    def _set_height(self,scf,position):
        self._go_to(scf,position)
    
    def set_position(self,scf,position):
        self._go_to(scf,position)

    def setup(self) -> bool:
        if(self.dry_run):
            print("[SwarmManager] Updating state to not_flying")
            self.state = 'not_flying'
            print("[SwarmManager] Dry Run Setup")
            return
        print("[SwarmManager] Initialising CRTP Drivers")
        cflib.crtp.init_drivers()
        print("[SwarmManager] Opening Links")
        self.swarm.open_links()
        print("[SwarmManger] Waiting for Position Estimation")
        self.swarm.__wait_for_position_estimator()
        print("[SwarmManager] Saving Estimated Positions")
        est_positions = self.swarm.get_estimated_positions()
        for k,v in est_positions.items():
            self.positions[k] = (v.x,v.y,v.z)
        print("[SwarmManager] Creating Commanders")
        self.swarm.parallel(self._create_commander)
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
                self.swarm.parallel(lambda scf: self.commanders.get(scf.cf.link_uri).takeoff(height))
            print("[SwarmManager] Updating state to flying")
            self.state = 'flying'
            
    def land(self) -> None:
        if self.state in ['flying']:
            print("[SwarmManager] Landing")
            if(not self.dry_run):
                self.swarm.parallel(lambda scf: scf.cf.land())
            print("[SwarmManager] Updating state to not_flying")
            self.state = 'not_flying'

    def set_height(self,height):
        if self.state in ['flying']:
            desired_positions = {}
            for k,v in self.positions:
                desired_positions[k] = (v[0],v[1],height)
            print("[SwarmManger] Moving to desired height")
            if(self.dry_run):
                print("[SwarmManger] Dry Run Moved")
                return
            else:
                self.swarm.parallel_safe(self._set_height,desired_positions)

        else:
            print("[SwarmManager] Error: Drones are not flying")

