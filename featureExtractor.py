import cv2
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import numpy as np 

class FeatureExtractor(object):
    def __init__(self, orbParam):
        self.kpData = []
        self.orb = cv2.ORB_create(orbParam)

    def computeKpData(self, img):
        kp, des = self.orb.detectAndCompute(img,None)
        return [kp,des]

    def getMatchingPoints(kpDataIdx01,kpDataIdx_02):
        return ExtractMatchingInliers(kpData[kpDataIdx01],kpData[kpDataIdx02])

    def ExtractMatchingInliers(self, srcImgKpData, dstImgKpData):
        #Matching
        prevImg = srcImgKpData
        curImg = dstImgKpData
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(curImg[1],prevImg[1], k=2)

        #Filtering
        # Lowe's Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        src_pts = np.float32([ prevImg[0][m.trainIdx].pt for m in good ]).reshape(-1, 2)
        dst_pts = np.float32([ curImg[0][m.queryIdx].pt for m in good ]).reshape(-1, 2)

        # Ransac
        model, inliers = ransac(
            (src_pts, dst_pts),
            AffineTransform, min_samples=4,
            residual_threshold=8, max_trials=100
        )

        #Format Output
        matchingInliers = []
        src_pts_inliers = []
        dst_pts_inliers = []
        index = 0
        for i in inliers:
            if(i):
                matchingInliers.append([src_pts[index],dst_pts[index]])
                src_pts_inliers.append(src_pts[index])
                dst_pts_inliers.append(dst_pts[index])
            index+=1        
        src_pts_inliers = np.array(src_pts_inliers) 
        dst_pts_inliers = np.array(dst_pts_inliers) 
        return matchingInliers,src_pts_inliers,dst_pts_inliers

    