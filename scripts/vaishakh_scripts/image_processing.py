#!/user/bin/env python2

# Python libs
import numpy as np
import cv2
import math
import sys
import operator
# user defined classes
from perceptionLineClass import Line, filterClose
from perceptionsquareClass import Square
# acessing directories
import os

# ROS libs/packs
# import rospy
# import ros_numpy

# ROS message
from sensor_msgs.msg import Image

PLAYCHESS_PKG_DIR = '/home/vaishakh/playchess/scripts/vaishakh_scripts/Static_images'


class image_processing():
    def __init__(self):
        self.debug = 0

    # Canny edge function
    def cannyEdgeDetection(self, img):
        '''
        canny edge detection
        1. used in image_analysis befor findContour
        '''
        edges = cv2.Canny(img,
                          25,    # min_threshold: adjusting it helps to distinguish beteween brown boundary and black  square near to it
                          255
                          )

        # DEBUG
        if self.debug:
            cv2.imshow('Canny', edges)
            cv2.waitKey(0)

        return edges

    # Hough line
    def categoriseLines(self, lines):
        '''
        used to sort lines  in Haugh lines as horizontal and vertical. Then sort the lines based on their respective center (ascending)
        '''
        horizontal = []
        vertical = []
        for i in range(len(lines)):
            if lines[i].category == 'horizontal':
                horizontal.append(lines[i])
            else:
                vertical.append(lines[i])

        horizontal = sorted(horizontal, key=operator.attrgetter('centerH'))
        vertical = sorted(vertical, key=operator.attrgetter('centerV'))

        return horizontal, vertical

    def houghLines(self, edges, img, thresholdx=40, minLineLengthy=100, maxLineGapz=50):
        """
        Detects Hough lines
        edge- o/p of canny
        """

        # Detect hough lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=thresholdx,
                                minLineLength=minLineLengthy, maxLineGap=maxLineGapz)
        N = lines.shape[0]

        # Draw lines on image
        New = []
        for i in range(N):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]

            New.append([x1, y1, x2, y2])

        lines = [Line(x1=New[i][0], y1=New[i][1], x2=New[i][2],
                      y2=New[i][3]) for i in range(len(New))]

        # Categorise the lines into horizontal or vertical
        horizontal, vertical = self.categoriseLines(lines)
        # Filter out close lines based to achieve 9

        # STANDARD THRESHOLD SHOULD BE 20
        ver = filterClose(vertical, horizontal=False, threshold=20)
        hor = filterClose(horizontal, horizontal=True, threshold=20)

        '''
        Adjusting arguments of cv2.HoughLinesP
        '''

        # DEBUG TO SHOW LINES
        print(len(ver))
        print(len(hor))
        if self.debug:
            debugImg = img.copy()
            self.drawLines(debugImg, ver)
            self.drawLines(debugImg, hor)
            cv2.imshow(" Hough Lines Found", debugImg)
            cv2.waitKey(0)

        return hor, ver

    def drawLines(self, img, lines, color=(0, 0, 255), thickness=2):
        """
        Draws lines. This function was used to debug Hough Lines generation.
        """
        # print("Going to print: ", len(lines))
        for l in lines:
            l.draw(img, color, thickness)
            # DEBUG
            cv2.imshow('image', img)

    # Intersections
    def findIntersections(self, horizontals, verticals, image):
        '''
        Finds intersections between Hough lines and filters out close points.
        '''
        intersections = []

        # Find the intersection points
        for horizontal in horizontals:
            for vertical in verticals:
                d = horizontal.dy*vertical.dx-horizontal.dx*vertical.dy
                dx = horizontal.c*vertical.dx-horizontal.dx*vertical.c
                dy = horizontal.dy*vertical.c-horizontal.c*vertical.dy

                if d != 0:
                    x = abs(int(dx/d))
                    y = abs(int(dy/d))
                else:
                    return False

                intersections.append((x, y))
        if self.debug:
            print("")
            print("We have found: " + str(len(intersections)) + " intersections.")
            print("")
            debugImg = image.copy()

            for intersection in intersections:
                cv2.circle(debugImg, intersection, 10, 255, 1)

            cv2.imshow("Intersections Found", debugImg)
            cv2.waitKey(0)
        # FILTER

        # Filtering intersection points
        minDistance = 10
        '''
        lowering minDistance can help find corners far from tiago due to perspective distortion '''

        # Only works if you run it several times -- WHY? Very inefficient
        # Now also works if run only once so comment the loop out
        for i in range(4):
            for intersection in intersections:
                for neighbor in intersections:
                    distanceToNeighbour = np.sqrt(
                        (intersection[0] - neighbor[0]) ** 2 + (intersection[1] - neighbor[1]) ** 2)
                    # Check that it's not comparing the same ones
                    if distanceToNeighbour < minDistance and intersection != neighbor:
                        intersections.remove(neighbor)

        # We still have duplicates for some reason. We'll now remove these
        filteredIntersections = []
        # Duplicate removal
        seen = set()
        for intersection in intersections:
            # If value has not been encountered yet,
            # ... add it to both list and set.
            if intersection not in seen:
                filteredIntersections.append(intersection)
                seen.add(intersection)
        if self.debug:
            print("")
            print("We have filtered: " +
                  str(len(filteredIntersections)) + " intersections.")
            print("")

            debugImg = image.copy()

            for intersection in filteredIntersections:
                cv2.circle(debugImg, intersection, 10, (0, 0, 255), 1)
            cv2.imshow("filtered_intersections", debugImg)
            cv2.waitKey(0)

        return filteredIntersections

    # Assign Intersection
    def assignIntersections(self, image, intersections):
        """
        Takes the filtered intersections and assigns them to a list containing nine sorted lists, each one representing
        one row of sorted corners. The first list for instance contains the nine corners of the first row sorted
        in an ascending fashion. This function necessitates that the chessboard's horizontal lines are exactly
        horizontal on the camera image, for the purposes of row assignment.
        """

        # Corners array / Each list in list represents a row of corners
        corners = []

        '''
        key = lambda for sorting x wrt the second element here the secon element is y-co-ordinate of the points '''

        for row in corners:
            row.sort(key=lambda x: x[0])
        '''
        above code use ule if chess board parallel to camera
        '''
        # Sort auatomaticall in order left to right top to bot saved in a list
        print(intersections)
        print('end')

        print(intersections)

        del intersections[0:11]
        del intersections[-11:len(intersections)]
        del intersections[0:len(intersections):11]
        del intersections[9:len(intersections):10]

        print(type(intersections))

        # DEBUG
        if self.debug:

            print(intersections)

            cornerCounter = 0

            debugImg = image.copy()
            # for row in corners:
            for corner in intersections:
                # for corner in row:
                # cv2.circle(debugImg, corner, 10, (0,255,0), 1)
                # cornerCounter += 1
                cv2.circle(debugImg, corner, 10, (0, 255, 0), 1)
                cornerCounter += 1

            cv2.imshow("Final Corners", debugImg)

            print("")
            print("There are: " + str(cornerCounter) +
                  " corners that were found.")
            print("")

        return intersections, image

        """
    SQUARE INSTANTIATION
    """

    def makeSquares(self, corners, depthImage, image):
        """
        Instantiates the 64 squares given 81 corner points.
        """

        # List of Square objects
        squares = []
        coordinates = []
        # Lists containing positional and index information
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
        index = 0

        # print(corners)

        for i in range(8):
            for j in range(8):
                # Make the square - yay!
                position = letters[-i-1] + numbers[-j-1]
                c1 = corners[i][j]
                c2 = corners[i][j+1]
                c3 = corners[i+1][j+1]
                c4 = corners[i+1][j]
                square = Square(position, c1, c2, c3, c4, index, image)
                # print(c1, c2, c3, c4)
                squares.append(square)

                square.draw(image)

                index += 1
                # print(index)
                xyz = square.getDepth(depthImage)
                coordinates.append(xyz)

        cv2.imshow("Board Identified", image)

        if self.debug:
            cv2.imwrite("5SquaresIdentified.jpeg", image)

        # Get x,y,z coordinates from square centers & depth image
        # coordinates = self.getDepth(square.roi, depthImage)
        # DEBUG
        if self.debug:
            print("Number of Squares found: " + str(len(squares)))

        return squares

        # Trackbar for setting the Hough parameters
    def track_bar(self, img, canny_img, intersections):
        '''
        for mannually setting up values of Hough parameter to corretly detect the square corners
        '''
        def call_back(x):
            pass

        cv2.namedWindow('adjust_Hough_parameter', cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            'threshold', 'adjust_Hough_parameter', 40, 200, call_back)
        cv2.createTrackbar('min_line_length',
                           'adjust_Hough_parameter', 100, 200, call_back)
        cv2.createTrackbar(
            'max_line_gap', 'adjust_Hough_parameter', 50, 200, call_back)
        cv2.createTrackbar('off', 'adjust_Hough_parameter', 0, 1, call_back)
        switch = '0 : NOT_ACTIVE \n 1 : ACTIVE'
        cv2.createTrackbar(switch, 'adjust_Hough_parameter', 0, 1, call_back)
        off = 0
        while (off != 1):
            track_img = img.copy()
            cornerCounter = 0
            for corner in intersections:
                # for corner in row:
                # cv2.circle(debugImg, corner, 10, (0,255,0), 1)
                # cornerCounter += 1
                cv2.circle(track_img, corner, 10, (0, 255, 255), 1)
                cornerCounter += 1
            cv2.imshow('adjust_Hough_paramete', track_img)
            cv2.waitKey(30)
            s = cv2.getTrackbarPos(switch, 'adjust_Hough_parameter')

            if s == 0:
                pass
            else:
                off = cv2.getTrackbarPos('off', 'adjust_Hough_parameter')
                thresholdx = cv2.getTrackbarPos(
                    'threshold', 'adjust_Hough_parameter')
                minLineLengthy = cv2.getTrackbarPos(
                    'min_line_length', 'adjust_Hough_parameter')
                maxLineGapz = cv2.getTrackbarPos(
                    'max_line_gap', 'adjust_Hough_parameter')

                hor, ver = self.houghLines(
                    canny_img, img, thresholdx, minLineLengthy, maxLineGapz)
                intersections = self.findIntersections(hor, ver, img)
                corner, image = self.assignIntersections(img, intersections)

    # Image_preprocessing

    def preprocessing(self, img, kernelG=5):
        '''
        input image -> GaussianBlur -> Gray -> Adaptive Thrusholding
        '''
        img = cv2.GaussianBlur(img, (kernelG, kernelG), 0)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        adaptiveThresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

        # Show  images for debug
        if self.debug:
            cv2.imshow('Gray', gray)
            cv2.imshow("Adaptive_Thresholded", adaptiveThresh)
            cv2.waitKey(0)

        return adaptiveThresh

    # Image analysis
    def image_analysis(self, img, preprocessed_img):
        '''
        finding the contours in the chessboard, filter the largest one and masks it

        '''
        # Chessboard counter extration

        # # canny edge detection
        # preprocessed_img = self.cannyEdgeDetection(preprocessed_img)

        # Find countours

        imgC, contours, hierarchy = cv2.findContours(
            preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # copy image
        imgContours = img.copy()

        for i in range(len(contours)):
            # Area
            area = cv2.contourArea(contours[i])
            # Peremeter
            perimeter = cv2.arcLength(contours[i], True)
            ''' finding largest counter
                1. avoide zero error
                2. largest contour has largest ratio'''
            if i == 0:
                Lratio = 0
            if perimeter > 0:
                ratio = area/perimeter
                if ratio > Lratio:
                    largest = contours[i]
                    Lratio = ratio
                    Lperimeter = perimeter
                    Larea = area
                else:
                    pass
        # DEBUG
        color = (255, 0, 0)
        if self.debug:
            cv2.drawContours(imgContours, [largest], -1, color, 2)

        # approximation factor
        epsilon = 0.1 * Lperimeter

        # approximating a poligon for chess board edges
        chessboardEdge = cv2.approxPolyDP(largest, epsilon, True)

        # DEBUG
        if self.debug:
            cv2.drawContours(imgContours, [chessboardEdge], -1, color, 2)
            cv2.imshow('contours_in_image', imgContours)
            cv2.waitKey(0)

        # creating ROI
        roi = cv2.polylines(
            imgContours, [chessboardEdge], True, (0, 255, 255), thickness=3)
        # showing filtered contour image
        # DEBUG
        if self.debug:
            cv2.imshow('Filtered_contour', imgContours)
            cv2.waitKey(0)

        # creating a mask
        mask = np.zeros((img.shape[0], img.shape[1]), 'uint8') * 125

        # copy chess board edges as a filled white polygon
        cv2.fillConvexPoly(mask, chessboardEdge, 255, 1)
        # coping the chess board to mask
        extracted = np.zeros_like(img)
        extracted[mask == 255] = img[mask == 255]
        # making mask green check later
        # extracted[np.where((extracted == [125, 125, 125]).all(axis=2))]= [0, 0, 20]
        # # add same colored line to remove chess board edges
        cv2.polylines(extracted, [chessboardEdge],
                      True, (0, 255, 0), thickness=5)

        # chessboardEdge = chessboardEdge[0][0]
        # DEBUD
        if self.debug:
            print('chessBord edges')
            print(chessboardEdge)
            # cv2.imshow("1 Masked", extracted)
            cv2.waitKey(0)

        return extracted, chessboardEdge

 # MAIN


if __name__ == "__main__":

    img = cv2.imread(
        '/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/Static_images/optimum_lighting.png')
    ip = image_processing()
    preprocessed_img = ip.preprocessing(img)
    pre_canny = ip.cannyEdgeDetection(preprocessed_img)
    # Chessboard edges used to eliminate unwanted point in assign intersections function
    analysed_img, chessBoardEdges = ip.image_analysis(img, pre_canny)
    canny_img = ip.cannyEdgeDetection(analysed_img)
    hor, ver = ip.houghLines(canny_img, img)
    intersections = ip.findIntersections(hor, ver, img)
    corner, image = ip.assignIntersections(img, intersections)
    ip.track_bar(img, canny_img, corner)
    squares = ip.makeSquares(corner, img)

    # ip.makeSquares(corner,image)
    cv2.imshow('Finalimg', analysed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
