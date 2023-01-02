import os
import cv2
import time
from display import Display 
from featureExtractor import FeatureExtractor 
from essentialMatrix import EssentialMatrix 
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import numpy as np 

from matplotlib import pyplot as plt 

from lib.visualization import plotting
from lib.visualization.video import play_trip
from tqdm import tqdm


W = 602*2 #screenwidth
H = 183*2 #screenheight
F = 707.09

# intrensics (calibration matrix)
K = np.array([[F ,0 , W/2],
              [0 ,F , H/2],
              [0 ,0 , 1  ]])
# intrensics (projection matrix)
P = np.array([[1 ,0 , 0  , 0],
              [0 ,1 , 0  , 0],
              [0 ,0 , 1/F, 0]])

disp = Display(W,H,"SLAM")
fe = FeatureExtractor(3000)
em = EssentialMatrix(K,P)

estimated_path = []
estimated_yaw = []
estimated_pitch = []
estimated_roll = []
estimated_frame_number = [0]
frameIndex = []

cur_pose = np.array([[1 ,0 ,0 ,0],
                   [0 ,1 ,0 ,0],
                   [0 ,0 ,1 ,0],
                   [0 ,0 ,0 ,1],])

def process_frame(img):
    global cur_pose
    img = cv2.resize(img, (W, H))
    fe.kpData.append(fe.computeKpData(img))
    if(len(fe.kpData)<3):return
    matchingInliers, src_pts_inliers, dst_pts_inliers = fe.ExtractMatchingInliers(fe.kpData[len(fe.kpData)-1],fe.kpData[len(fe.kpData)-3])
    
    transf = em.get_pose(src_pts_inliers,dst_pts_inliers)
    cur_pose = np.matmul(cur_pose, transf)
    estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    
    print ("\n Current pose:\n" + str(cur_pose))
    print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
    yaw, pitch, roll = em.getYawPitchRoll(transf)

    estimated_yaw.append(yaw)
    estimated_pitch.append(pitch)
    estimated_roll.append(roll)
    estimated_frame_number.append(estimated_frame_number[len(estimated_frame_number)-1]+1)
    print ("The current pose used x,y: \n",yaw, " : ",pitch, " : ", roll)

    for points in matchingInliers:
        u1,v1 = map(lambda x: int(round(x)),points[0])
        u2,v2 = map(lambda x: int(round(x)),points[1])
        cv2.circle(img,(u1,v1), color=(0,255,0), radius=2)
        cv2.circle(img,(u2,v2), color=(0,0,255), radius=2)
        cv2.line(img,(u1,v1),(u2,v2), color=(255,0,0))
    disp.draw(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test04.mp4")
    data_dir = 'SLAM'
    
    re_img_arr = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else: break

    estimated_frame_number.pop()
    plt.plot(estimated_frame_number, estimated_yaw,  label = "Yaw")
    plt.plot(estimated_frame_number, estimated_pitch,  label = "pitch")
    plt.plot(estimated_frame_number, estimated_roll,  label = "roll")
    plt.legend()
    plt.show()
    plotting.visualize_paths(estimated_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")
    
    
