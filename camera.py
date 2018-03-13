import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os


class camera:
    def __init__(self):
        self.objpoints = None
        self.imgpoints = None
        self.mtx = None
        self.dist = None
        
    def chessboard_calibrate(self,img_path,x_cor,y_cor):
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((y_cor*x_cor,3), np.float32)
        objp[:,:2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.
        img_size = None
        images = glob.glob(img_path) # Make a list of paths to calibration images
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Conver to grayscale
            ret, corners = cv2.findChessboardCorners(gray, (x_cor,y_cor), None) # Find the chessboard corners
            img_size = (img.shape[1], img.shape[0])
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size,None,None)
        
        
    def undistort(self,img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst
    
    
        