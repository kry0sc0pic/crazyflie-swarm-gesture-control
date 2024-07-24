# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Import Crazyflie Libraries
from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.crazyflie.swarm import Swarm, CachedCfFactory
from utils import wait_for_position_estimator

# crazyflie config
URIS = ["radio://0/90/2M/E7E7E7E7E5"]
DRY_RUN = False
# Init Drivers
init_drivers()

# Download the model from TF Hub.
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/3')
movenet = model.signatures['serving_default']

# Threshold for detection
threshold = .3

# Loads video source (0 is for main webcam)
video_source = 0
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

def diff_to_height(diff:float):
    if (0 < diff < 1):
        return (diff * 1.5) + 0.5
    return 0.5

with SyncCrazyflie(URIS[0], cf=Crazyflie(rw_cache='./cache')) as scf:
    if(not DRY_RUN):
        cmdr = PositionHlCommander(scf,
                               default_height=0.5,
                               x = 0,
                               y = 0,
                               z = 0,
                               controller=PositionHlCommander.CONTROLLER_PID)
        cmdr.take_off(0.5)
    while success:
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
        if(not DRY_RUN): cmdr.go_to(0,0,h_to_set)


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
        
        # img = cv2.line(img,(),())
        
        # Shows image
        cv2.imshow('Movenet Results', img)
        # Waits for the next frame, checks if q was pressed to quit
        if cv2.waitKey(1) == ord("q"):
            break

        # Reads next frame
        success, img = cap.read()

    cap.release() 
    if(not DRY_RUN): cmdr.land()