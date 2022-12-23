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

PLAYCHESS_PKG_DIR = '/home/vaishakh/scripts/Vaishakh_scripts/Static_images'


class image_processing():
    def __init__(self):
        self.debug = True
        
    # Image_preprocessing
    def processFile(self, img, debug=False):
        """
        Converts input image to grayscale & applies adaptive thresholding.
        """
        img = cv2.GaussianBlur(img,(5,5),0)
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # HSV Thresholding
        res,hsvThresh = cv2.threshold(hsv[:,:,0], 25, 250, cv2.THRESH_BINARY_INV)
        # Show adaptively thresholded image
        adaptiveThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        # Show both thresholded images
        # cv2.imshow("HSV Thresholded",hsvThresh)

        if debug:
            cv2.imshow("Adaptive Thresholding", adaptiveThresh)

        return img, adaptiveThresh



    def imageAnalysis(self, img, processedImage, debug=False):
        """
        Finds the contours in the chessboard, filters the largest one (the chessboard) and masks it.
        """

        ### CHESSBOARD EXTRACTION (Contours)

        # Find contours
        _, contours, hierarchy = cv2.findContours(processedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Create copy of original image
        imgContours = img.copy()

        for c in range(len(contours)):
            # Area
            area = cv2.contourArea(contours[c])
            # Perimenter
            perimeter = cv2.arcLength(contours[c], True)
            # Filtering the chessboard edge / Error handling as some contours are so small so as to give zero division
            #For test values are 70-40, for Board values are 80 - 75 - will need to recalibrate if change
            #the largest square is always the largest ratio
            if c ==0:
                Lratio = 0
            if perimeter > 0:
                ratio = area / perimeter
                if ratio > Lratio:
                    largest=contours[c]
                    Lratio = ratio
                    Lperimeter=perimeter
                    Larea = area
            else:
                pass

        # DEBUG statements
        color = (255,255,255)
        if debug:
            cv2.drawContours(imgContours, [largest], -1, color, 2)

        # Epsilon parameter needed to fit contour to polygon
        epsilon = 0.1 * Lperimeter
        # Approximates a polygon from chessboard edge
        chessboardEdge = cv2.approxPolyDP(largest, epsilon, True)


        # DEBUG
        if debug:
            cv2.drawContours(imgContours, [chessboardEdge], -1, color, 2)

        # Draw chessboard edges and assign to region of interest (ROI)
        roi = cv2.polylines(imgContours,[chessboardEdge],True,(0,255,255),thickness=3)

        # Show filtered contoured image

        #DEBUG
        if debug:
            #cv2.imshow("0 Filtered Contours", imgContours)
            cv2.imwrite("0FilteredContours.jpeg", imgContours)

        # Create new all black image
        mask = np.zeros((img.shape[0], img.shape[1]), 'uint8')*125
        # Copy the chessboard edges as a filled white polygon
        cv2.fillConvexPoly(mask, chessboardEdge, 255, 1)
        # Assign all pixels to out that are white (i.e the polygon, i.e. the chessboard)
        extracted = np.zeros_like(img)
        extracted[mask == 255] = img[mask == 255]
        # Make mask green in order to facilitate removal of the red strip around chessboard
        extracted[np.where((extracted == [125, 125, 125]).all(axis=2))] = [0, 0, 20]
        # Adds same coloured line to remove red strip based on chessboard edge
        cv2.polylines(extracted, [chessboardEdge], True, (0, 255, 0), thickness=5)

        if debug:
            #cv2.imshow("1 Masked", extracted)
            cv2.imwrite("1ExtractedMask.jpeg", extracted)

        return extracted

 
 #MAIN 

if __name__ == "__main__":
    
    img = cv2.imread('/home/vaishakh/scripts/vaishakh_scripts/Static_images/empty_chess_board.png')
    ip = image_processing()
    preprocessed_img = ip.processFile(img)
    analysed_img = ip.imageAnalysis(img,preprocessed_img)
    cv2.imshow('Finalimg',analysed_img)
    cv2.waitKey(0)