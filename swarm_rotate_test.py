from utils.manager import SwarmManager
URIS = [
    # "radio://0/20/2M/E7E7E7E7E5",
    # "radio://0/30/2M/E7E7E7E7E5",
    # "radio://0/40/2M/E7E7E7E7E5",
    # "radio://0/60/2M/E7E7E7E7E5",
    # "radio://0/80/2M/E7E7E7E7E5",
    # "radio://0/90/2M/E7E7E7E7E5",
    # "radio://0/10/2M/E7E7E7E7E7",
    # "radio://0/30/2M/E7E7E7E7E7",
    # "radio://0/40/2M/E7E7E7E7E7",
    "radio://0/50/2M/E7E7E7E7E7",
    # "radio://0/70/2M/E7E7E7E7E7",
    # "radio://0/80/2M/E7E7E7E7E7",
    ] 

HEIGHT_MAP = {
    "radio://0/30/2M/E7E7E7E7E7": 0.5,
    "radio://0/40/2M/E7E7E7E7E7": 0.5,
    "radio://0/50/2M/E7E7E7E7E7": 0.5
}



manager = SwarmManager(URIS,dry_run=False,use_low_level=True)
print("Setting Up Swarm")
manager.setup(enableINDI=True)

print("Taking Off")
manager.takeoff()

# print("Set Heights")
# manager.set_multiple_heights(HEIGHT_MAP)


print("Rotating Swarm")
manager.rotate_swarm(duration=5,stage=-1)

print("Landing")
manager.land()

print("Cleaning")
manager.cleanup()