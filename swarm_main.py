import time

# CFLIB Imports
from utils.manager import SwarmManager
from cflib.crtp import init_drivers

# Tensorflow Imports
import tensorflow as tf
import cv2
import numpy as np

# Utilities
from utils.kps import list_to_movenet_keypoints, KeypointMapper, MovenetKeypoints

# Pose Estimation Options
USE_TFLITE = False # Use the TFLite Lightning Model
TFLITE_MODEL = "models/singlepose-lightning/3.tflite" # Model Path
VIDEO_SOURCE = None # Set a path to use a video as input, leave it as `None` to use the webcam
PREDICTION_THRESHOLD = .3

# Crazyflie Options
URIS = [] # the addresses for the crazyflies
DRY_RUN = True # Set to `True` to not use the actual crazyflies


# Variables
calibrationComplete = False
rest_position_readings = []
rest_position_diff = None
readings_to_take = 10


# Setup Tensorflow
if not USE_TFLITE:
    import tensorflow_hub as hub
    model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/3')
    movenet = model.signatures['serving_default']

else:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()
    def movenet(input_image):
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.float32)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores



# Setup Video Capture
video_capture = cv2.VideoCapture(0 if VIDEO_SOURCE is None else VIDEO_SOURCE)
if not video_capture.isOpened():
    print("[VideoCapture] Error Loading Video Source")
    quit()

success, frame = video_capture.read()
if not success:
    print("[VideoCapture] Error reading frame")

dim_y , dim_x , _ = frame.shape

# Create Swarm Manger

manager = SwarmManager(URIS,dry_run=DRY_RUN)
print("Setting Up Swarm")
manager.setup()

keypointMapper = None

# debug data
fps = 0
frame_count = 0
capture_start_time = time.time()

while success:
    # Update FPS
    frame_count+=1
    if frame_count >= 10:
        capture_end_time = time.time()
        fps = frame_count / (capture_end_time - capture_start_time)
        frame_count = capture_end_time

    # Convert Image to Correct Format
    tf_img = cv2.resize(frame,(192,192))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img,axis=0)
    image = tf.cast(tf_img,dtype=tf.int32)


    # Inference
    output = movenet(image)

    keypoints = output
    points = keypoints[0][0]
    if not USE_TFLITE:
        keypoints = keypoints['output_0']
        points = points.numpy()
    
    mapped_points = list_to_movenet_keypoints(points)
    if keypointMapper is None:
        keypointMapper = KeypointMapper(mapped_points,threshold=PREDICTION_THRESHOLD)
    else:
        keypointMapper.update(mapped_points)
    
    des_height = keypointMapper.calculate_desired_height()
    if(not calibrationComplete and len(rest_position_readings) < readings_to_take):
        h = keypointMapper.calculate_raw_height()
        rest_position_readings.append(h)
        if(len(rest_position_readings) == readings_to_take):
            rest_position_diff = sum(rest_position_readings) / readings_to_take
            calibrationComplete = True

        
    
    #TODO: auto takeoff logic


    if calibrationComplete:
        manager.set_height(des_height)
    
 # iterate through keypoints
    for i,k in enumerate(keypoints[0,0,:,:]):
        # Converts to numpy array
        k = k if USE_TFLITE else k.numpy()

        # Checks confidence for keypoint
        if k[2] > PREDICTION_THRESHOLD:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * dim_y)
            xc = int(k[1] * dim_x)

            # Draws a circle on the image for each keypoint
            frame = cv2.circle(frame, (xc, yc), 2, (0, 255, 0), 5)
        
    if all(i != (0,0) for i in keypointMapper.tracked_coordinates_list):
        left_wrist = (keypointMapper.left_wrist[0]*dim_x,keypointMapper.left_wrist[1]*dim_y)
        right_wrist = (keypointMapper.right_wrist[0]*dim_x,keypointMapper.right_wrist[1]*dim_y)
        left_hip = (keypointMapper.left_hip[0]*dim_x,keypointMapper.left_hip[1]*dim_y)
        right_hip = (keypointMapper.right_hip[0]*dim_x,keypointMapper.right_hip[1]*dim_y)
        if(left_wrist[1] > left_hip[1]):
            left_line_color = (255,0,0) # red
        else:
            left_line_color = (0,255,0) # green
        
        if(right_wrist[1] > right_hip[1]):
            right_line_color = (255,0,0) # red
        else:
            right_line_color = (0,255,0) # green
        
        # wrist to hip level
        right_hip_level = (right_wrist[0],right_hip[1])
        left_hip_level = (left_wrist[0],left_hip[1])

        # lines
        cv2.line(frame, right_wrist, right_hip_level, right_line_color, 2)
        cv2.line(frame, left_wrist, left_hip_level, left_line_color, 2)

        # avg calc
        norm_h =  (right_wrist[0]  + left_wrist[0]) / 2
        img_h = norm_h * dim_y
        cv2.line(frame,(0,img_h),(dim_x,img_h),(0,0,255),1)
        
        # baseline
        if (calibrationComplete):
            cv2.line(frame,(0,rest_position_diff * dim_y),(dim_x,rest_position_diff*dim_y),(255,0,0),1)
        



    cv2.putText(frame,f"FPS: {fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.putText(frame,f"State: {manager.state}",(10,60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.putText(frame, f"Height: {des_height:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Pose Estimation",frame)
    
    if cv2.waitKey(1) == ord('q'):
        print("Exiting")
        manager.land()
        break

    success, frame = video_capture.read()

    if VIDEO_SOURCE and not success:
        break

video_capture.release()
manager.cleanup()