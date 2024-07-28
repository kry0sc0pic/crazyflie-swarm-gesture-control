from collections import namedtuple
import threading
MOVENET_POINTS = [
"nose",
"left_eye",
"right_eye",
"left_ear",
"right_ear",
"left_shoulder",
"right_shoulder",
"left_elbow",
"right_elbow",
"left_wrist",
"right_wrist",
"left_hip",
"right_hip",
"left_knee",
"right_knee",
"left_ankle",
"right_ankle",
]
MovenetKeypoints = namedtuple('MovenetKeypoints',MOVENET_POINTS)

BLAZEPOSE_POINTS = [
"nose",
"left_eye_inner",
"left_eye",
"left_eye_outer",
"right_eye_inner",
"right_eye",
"right_eye_outer",
"left_ear",
"right_ear",
"mouth_left",
"mouth_right",
"left_shoulder",
"right_shoulder",
"left_elbow",
"right_elbow",
"left_wrist",
"right_wrist",
"left_pinky",
"right_pinky",
"left_index",
"right_index",
"left_thumb",
"right_thumb",
"left_hip",
"right_hip",
"left_knee",
"right_knee",
"left_ankle",
"right_ankle",
"left_heel",
"right_heel",
"left_foot_index",
"right_foot_index",
"bodyCenter",
"forehead",
"leftThumb",
"leftHand",
"rightThumb",
"rightHand",
]
BlazeposeKeypoints = namedtuple('BlazeposeKeypoints',BLAZEPOSE_POINTS)

def list_to_movenet_keypoints(points) -> MovenetKeypoints:
    return MovenetKeypoints(*points)

def list_to_blazepose_points(points) -> BlazeposeKeypoints:
    return BlazeposeKeypoints(*points)

class KeypointMapper:
    def __init__(self,points: MovenetKeypoints | BlazeposeKeypoints,threshold: float = 0.3) -> None:
        self.points = points
        self.right_hip = (0,0)
        self.left_hip = (0,0)
        self.right_wrist = (0,0)
        self.left_wrist = (0,0) 
        self.nose = (0,0)
        self.tracked_coordinates_list = [self.nose,self.left_wrist,self.right_wrist,self.left_hip,self.right_hip]
        self.threshold = threshold
        if isinstance(points,MovenetKeypoints):
            self.type = 'movenet'
        elif isinstance(points,BlazeposeKeypoints):
            self.type = 'blazepose'
    
    def update(self,points: MovenetKeypoints | BlazeposeKeypoints):
        if self.type == 'movenet':
            if isinstance(points,MovenetKeypoints):
                if(points.right_hip[2] > self.threshold):
                    self.right_hip = (points.right_hip[0],points.right_hip[1])
                if(points.left_hip[2] > self.threshold):
                    self.left_hip = (points.left_hip[0],points.left_hip[1])
                if(points.right_wrist[2] > self.threshold):
                    self.right_wrist = (points.right_wrist[0],points.right_wrist[1])
                if(points.left_wrist[2] > self.threshold):
                    self.left_wrist = (points.left_wrist[0],points.left_wrist[1])
                if(points.nose[2] > self.threshold):
                    self.nose = (points.nose[0],points.nose[1])
                self.tracked_coordinates_list = [self.nose,self.left_wrist,self.right_wrist,self.left_hip,self.right_hip]
            else:
                raise Exception("Use MovenetKeypoints for movenet points")
    
        elif self.type == 'blazepose':
            if isinstance(points,BlazeposeKeypoints):
                pass
            else:
                raise Exception("Use BlazeposeKeypoints for blazepose points")\
        
    def calculate_raw_height(self) -> float:
        r_diff = self.right_hip[1] - self.right_wrist[1]
        l_diff = self.left_hip[1] - self.left_wrist[1]
        avg_diff = (r_diff+l_diff)/2
        return avg_diff       

    def calculate_desired_height(self) -> float:
        avg_diff = self.calculate_raw_difference()
        return avg_diff
    