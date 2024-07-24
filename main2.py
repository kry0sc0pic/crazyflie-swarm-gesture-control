# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time

# Import Crazyflie Libraries
from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.crazyflie.swarm import Swarm, CachedCfFactory
from utils import wait_for_position_estimator

# crazyflie config
URIS = ["radio://0/90/2M/E7E7E7E7E5"]
DRY_RUN = True
# Init Drivers
init_drivers()

# Download the model from TF Hub.
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/3')
movenet = model.signatures['serving_default']

# Threshold for detection
threshold = .3

# Minimum height threshold for takeoff
MIN_HEIGHT_THRESHOLD = 0.55 # Adjust this value as needed

DEFAULT_HEIGHT = 0.5  # Default takeoff height

# Maximum height threshold for landing
MAX_HEIGHT_THRESHOLD = 0.45  # Adjust this value as needed

# Time threshold for takeoff and landing (in seconds)
TIME_THRESHOLD = 4

# Video source configuration
VIDEO_FILE = None  # Set this to the path of your video file, or None for webcam
if VIDEO_FILE:
    video_source = VIDEO_FILE
else:
    video_source = 0  # 0 is for main webcam

# Control mode configuration
AUTO_CONTROL = True  # Set to False for manual takeoff and no auto-land

# Loads video source
cap = cv2.VideoCapture(video_source)

# Checks errors while opening the Video Capture
if not cap.isOpened():
    print('Error loading video')
    quit()

success, img = cap.read()
curr_coords = {
    "l_hip": (0,0),
    "r_hip": (0,0),
    "r_wrist": (0,0),
    "l_wrist": (0,0)
}

if not success:
    print('Error reading frame')
    quit()

y, x, _ = img.shape

state = 'not_flying'
takeoff_start_time = None
landing_start_time = None

def diff_to_height(diff:float):
    if (0 < diff < 1):
        return (diff * 1.5) + 0.5
    return 0.4

if (not DRY_RUN):
    scf = SyncCrazyflie(URIS[0], cf=Crazyflie(rw_cache='./cache')) 
    scf.open_link()
    cmdr = PositionHlCommander(scf,
                            default_height=DEFAULT_HEIGHT,
                            x = 0,
                            y = 0,
                            z = 0,
                            controller=PositionHlCommander.CONTROLLER_PID)

    # Manual takeoff if AUTO_CONTROL is False
    if not AUTO_CONTROL:
        print(f"Taking off to default height of {DEFAULT_HEIGHT}m")
        cmdr.take_off(DEFAULT_HEIGHT, 2.0)  # 2.0 is the time for takeoff in seconds
        state = 'flying'

# Initialize variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

while success:
    # FPS calculation
    frame_count += 1
    if frame_count >= 10:  # Update FPS every 10 frames
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        frame_count = 0
        start_time = time.time()

    # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
    tf_img = cv2.resize(img, (192,192))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img,axis=0)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf_img, dtype=tf.int32)

    # Run model inference.
    outputs = movenet(image)

    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']
    kps = keypoints[0][0].numpy()
    r_hip = kps[12]
    l_hip = kps[11]
    r_wrist = kps[10]
    l_wrist = kps[9]
    if(r_hip[2] > threshold):
        curr_coords['r_hip'] = (r_hip[1],r_hip[0])
    
    if(l_hip[2] > threshold):
        curr_coords['l_hip'] = (l_hip[1],l_hip[0])
    
    if(r_wrist[2] > threshold):
        curr_coords['r_wrist'] = (r_wrist[1],r_wrist[0])
    
    if(l_wrist[2] > threshold):
        curr_coords['l_wrist'] = (l_wrist[1],l_wrist[0])
    
    l_diff = curr_coords["l_hip"][1] - curr_coords["l_wrist"][1]
    r_diff = curr_coords["r_hip"][1] - curr_coords["r_wrist"][1]
    avg_diff = (l_diff + r_diff)/2
    print(f"Difference: {avg_diff}")
    h_to_set = diff_to_height(avg_diff)
    print(f"Height: {h_to_set}")

    if AUTO_CONTROL:
        # Check for takeoff conditions
        if h_to_set > MIN_HEIGHT_THRESHOLD and state == 'not_flying':
            if takeoff_start_time is None:
                takeoff_start_time = time.time()
            elif time.time() - takeoff_start_time >= TIME_THRESHOLD:
                if not DRY_RUN:
                    print("Taking off!")
                    cmdr.take_off(h_to_set)
                    state = 'flying'
        else:
            takeoff_start_time = None

        # Check for landing conditions
        if h_to_set < MAX_HEIGHT_THRESHOLD and state == 'flying':
            if landing_start_time is None:
                landing_start_time = time.time()
            elif time.time() - landing_start_time >= TIME_THRESHOLD:
                if not DRY_RUN:
                    print("Landing!")
                    cmdr.land()
                    state = 'not_flying'
        else:
            landing_start_time = None

    if state == 'flying' and not DRY_RUN:
        cmdr.go_to(0, 0, h_to_set)

    # iterate through keypoints
    for i,k in enumerate(keypoints[0,0,:,:]):
        # Converts to numpy array
        k = k.numpy()

        # Checks confidence for keypoint
        if k[2] > threshold:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * y)
            xc = int(k[1] * x)

            # Draws a circle on the image for each keypoint
            img = cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)

    # Draw lines for height difference visualization
    if all(curr_coords[key] != (0,0) for key in ['l_hip', 'r_hip', 'l_wrist', 'r_wrist']):
        l_hip_y = int(curr_coords['l_hip'][1] * y)
        l_hip_x = int(curr_coords['l_hip'][0] * x)
        r_hip_y = int(curr_coords['r_hip'][1] * y)
        r_hip_x = int(curr_coords['r_hip'][0] * x)
        l_wrist_y = int(curr_coords['l_wrist'][1] * y)
        l_wrist_x = int(curr_coords['l_wrist'][0] * x)
        r_wrist_y = int(curr_coords['r_wrist'][1] * y)
        r_wrist_x = int(curr_coords['r_wrist'][0] * x)

        # Draw lines from hips to wrists
        cv2.line(img, (l_hip_x, l_hip_y), (l_wrist_x, l_wrist_y), (255, 0, 0), 2)
        cv2.line(img, (r_hip_x, r_hip_y), (r_wrist_x, r_wrist_y), (255, 0, 0), 2)

        # Draw a horizontal line at the average wrist height
        avg_wrist_y = int((l_wrist_y + r_wrist_y) / 2)
        cv2.line(img, (0, avg_wrist_y), (x, avg_wrist_y), (0, 255, 255), 2)

        # Add text to show current height
        cv2.putText(img, f"Height: {h_to_set:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display flying state and FPS in the top left corner
    cv2.putText(img, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img, f"Mode: {'Auto' if AUTO_CONTROL else 'Manual'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Shows image
    cv2.imshow('Movenet Results', img)
    # Waits for the next frame, checks if q was pressed to quit
    if cv2.waitKey(1) == ord("q"):
        break

    # Reads next frame
    success, img = cap.read()

    # If the video file ends, break the loop
    if VIDEO_FILE and not success:
        break

cap.release() 
if state == 'flying' and not DRY_RUN:
    cmdr.land()

if not DRY_RUN:
    scf.close_link()