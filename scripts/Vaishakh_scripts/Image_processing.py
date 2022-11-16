#!/user/bin/env python

#Python libs
import numpy as np
import cv2 
import math
import sys
# acessing directories
import os

#ROS libs/packs
import rospy
import ros_numpy

#ROS message 
from sensor_msgs.msg import Image

PLAYCHESS_PKG_DIR = '/home/pal/tiago_public_ws/src/tiago_playchess'


class ImageProcessing(objects):
    def __init__(self):
        self.verbose = True
        

    #Image_prepration
    #Kernal size fo Gaussian filter =5
    #otsu T_val automatically calculated. set it to zero in function definition check later
    
    
    #Conversion to Gray scale
    def grayscale(self, image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #Show th original image and the grayscale image
        if self.verbose:
            cv2.imshow('Original image',image)
            cv2.imwrite(os.path.join(PLAYCHESS_PKG_DIR + '/Images/Segmentazione', '1-original.png'), image)
            cv2.imshow('Gray image', gray)
            cv2.imwrite(os.path.join(PLAYCHESS_PKG_DIR + '/Images/Segmentazione', '2-gray.png'), gray)
        return gray

    def gaussian_filter(self, image,kernel=5):
		#Filter the image with a Gaussian filter witht the specified kernel. The standard deviation in x an y directions is calculated starting from the kernel size (if not specifically given as inputs).
        gaussian = cv2.GaussianBlur(image,kernel, 0)
        if self.verbose:
            cv2.imshow('Gaussian filtered', gaussian)
            cv2.imwrite(os.path.join(PLAYCHESS_PKG_DIR + '/Images/Segmentazione', '3-gaussian.png'), gaussian)
        return gaussian

    def otsu_thresholding(self, image, T_val=0):
        ret, otsu = cv2.threshold(image, T_val, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if self.verbose:
            cv2.imshow('Otsu binarized', otsu)
            cv2.imwrite(os.path.join(PLAYCHESS_PKG_DIR + '/Images/Segmentazione', '4-otsu.png'), otsu)
        return otsu

    #Board detection
    #canny edge detection thresh1 and thresh2 definition
    
    def canny_edge(self, image, thresh1, thresh2):
        canny = cv2.Canny(image, thresh1, thresh2)
        if self.verbose:
            cv2.imshow('Edge detection', canny)
            cv2.imwrite(os.path.join(PLAYCHESS_PKG_DIR + '/Images/Segmentazione', '5-canny.png'), canny)
        return canny

    

    


