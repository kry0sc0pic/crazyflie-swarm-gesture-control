from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
import math 

def wait_for_position_estimator(scf):
    print(f"🔵 Waiting for Position Estimation -> {scf.cf.link_uri} ")
    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')
    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10
    threshold = 0.001
    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)
            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)
            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break
    print(f"🟢 Position Estimation Complete -> {scf.cf.link_uri} ")

def get_full_state_estimate(scf):
    print(f"🔵 Getting Full State Estimate -> {scf.cf.link_uri} ")
    log_config = LogConfig(name='stateEstimate', period_in_ms=10)
    log_config.add_variable('stateEstimate.x', 'float')
    log_config.add_variable('stateEstimate.y', 'float')
    log_config.add_variable('stateEstimate.z', 'float')
    log_config.add_variable('stateEstimate.yaw', 'float')
    with SyncLogger(scf, log_config) as logger:
        for entry in logger:
            x = entry[1]['stateEstimate.x']
            y = entry[1]['stateEstimate.y']
            z = entry[1]['stateEstimate.z']
            yaw = entry[1]['stateEstimate.yaw']
            break
    print(f"🟢 Full State Estimate Complete -> {scf.cf.link_uri} ")
    return [x,y,z,yaw]


def calculate_total_yaw(x:float,y:float,curr_yaw:float) -> int:
    """
    2   |  1
        |
    ---------

    3   |  0
    """
    quadrant = 0
    radius = math.sqrt(x*x + y*y) # distance from origin
    if(x > 0 and y > 0):
        quadrant = 1
    elif(x < 0 and y > 0):
        quadrant = 2
    elif(x < 0 and y < 0):
        quadrant = 3
    
    local_yaw = (180/math.pi) * math.atan(abs(y)/abs(x) if quadrant % 2 != 0 else abs(x)/abs(y)) # yaw with quadrant as reference
    abs_yaw = (quadrant*90) + local_yaw # yaw relative to 0 yaw
    adjusted_yaw = abs_yaw - curr_yaw
    return adjusted_yaw