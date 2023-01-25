#!/usr/bin/env python3

"""Segment empty chessboard from image, and get centers of each cell.
This script is based on OpenCV and numpy (no ROS-related dependencies), and implements a class that segments a chessboard from an image (based on contours), and identifies the center of each cell in the chessboard (based on Hough transform). As a unique set of Hough transform parameters is hard to be identified experimentally, two tuning methods are implemented:
- manual, based on sliders in a graphical interface
- automatic, based on Bayesian Optimization
"""

import numpy as np
import operator
import yaml

# User defined classes
from useful_functions.homographic_transformation import HOMO_TRANSFOR as hf
from useful_functions.perceptionLineClass import Line, filterClose
# NOTE. if the Line and filterClose classes come from the Franka example, please put in the corresponding file some reference to that work (e.g. a comment on top stating `Taken from: <URL>` or `Adapted from: <URL>`). This is a way to acknowledge the authors, and also very good for traceback in case of malfunctioning. 

# Open CV2 for saving an image
import cv2
import os
from bayes_opt import BayesianOptimization


class ImageProcessing():
    """Class containing functions to process chessboard images and detect squares.
    """
    def __init__(self, debug = False):
        # Activate/Deactivate debug print and imshow.
        self.debug = debug

    ######################
    ### PRE-PROCESSING ###
    ######################
    def preprocessing(self, img, gaussian_kernel_size = 5):
        """Apply Gaussian filtering, greyscale conversion and adaptive thresholding to an image.

        Args:
            img (np.ndarray): The image to process.
            gaussian_kernel_size (int, optional): The side of the square Gaussian kernel used for smoothing. Defaults to 5.

        Returns:
            np.ndarray: The thresholded image.
        """
        # Apply Gaussian blurring for noise reduction
        img = cv2.GaussianBlur(img, (gaussian_kernel_size, gaussian_kernel_size), 0)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        img_thresholded = cv2.adaptiveThreshold(img_gray,   # img 
                                               255,         # max value
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 
                                               115,         # block size
                                               1
                                               )

        # Show  images for debug
        if self.debug:
            cv2.imshow('Preprocessing', np.concatenate((img_gray, img_thresholded), axis=1))
            cv2.waitKey(2000)
            cv2.destroyWindow('Preprocessing')

        return img_thresholded


    def canny_edge_detection(self, img):
        """Run Canny edge detection on the input image.
        NOTE. Canny edge detection is a common preprocessing step. At time of writing, it used: 
            - before dilation and contour extraction when segmenting the chessboard from the image,
            - before Hough line detection when identifying the centers of the cells. 
        Args:
            img (np.ndarray): The input image on which to perform edge detection.

        Returns:
            np.ndarray: A binary image with the extracted edges.
        """
        # NOTE. The threshold value has been experimentally defined.
        img_edges = cv2.Canny(img,
                          25,    # min_threshold: adjusting it helps to distinguish beteween brown boundary and black  square near to it
                          255
                          )
        
        # DEBUG
        if self.debug:
            cv2.imshow('Canny edge detection', img_edges)
            cv2.waitKey(2000)
            cv2.destroyWindow('Canny edge detection')

        return img_edges


    def dilation(self, img, kernel_size = (1, 1), iterations = 1):
        """Convolve a binary image with a rectangular kernel to expand (dilate) its borders. Used to reduce noise in extracted contours/lines and improve robustness against small light variation.

        Args:
            img (np.ndarray): The binary image to dilate.
            kernel_size (tuple, optional): Tuple of two ints, defining the shape of the rectangular kernel. Defaults to (1, 1).
            iterations (int, optional): Number of repetitions of the dilation operation. Defaults to 1.

        Returns:
            np.ndarray: Binary image after the dilation operation.
        """
        kernel = np.ones(kernel_size, np.uint8)
        img_dilated = cv2.dilate(img, kernel, iterations)
        if self.debug:
            cv2.imshow('Edge dilation', img_dilated)
            cv2.waitKey(2000)
            cv2.destroyWindow('Edge dilation')
        return img_dilated

    
    ###############################
    ### CHESSBOARD SEGMENTATION ###
    ###############################
    def chessboard_contour(self, img, preprocessed_img):
        '''
        finding the contours in the chessboard, filter the largest one and masks it
        '''
        # Find countours
        # NOTE. The if statement makes it (hopefully) compatible with all the OpenCV versions.
        # From: https://stackoverflow.com/questions/20851365/opencv-contours-need-more-than-2-values-to-unpack
        if cv2.getVersionMajor() in [2, 4]:
            # OpenCV 2, OpenCV 4 case
            contours, __ = cv2.findContours(preprocessed_img.copy(), 
                                                  cv2.RETR_TREE, 
                                                  cv2.CHAIN_APPROX_NONE
                                                  )
        else:
            # OpenCV 3 case
            __, contours, __ = cv2.findContours(preprocessed_img.copy(), 
                                                         cv2.RETR_TREE, 
                                                         cv2.CHAIN_APPROX_NONE
                                                        )

        # copy image
        img_contours = img.copy()

        for i, c in enumerate(contours):
            # Area
            area = cv2.contourArea(c)
            # Peremeter
            perimeter = cv2.arcLength(c, True)
            ''' 
            Finding largest contour
                1. avoid zero error
                2. largest contour has largest ratio
            '''
            if perimeter > 0:
                ratio = area/perimeter
                if i == 0:
                    largest = c
                    Lratio = ratio      # initialize the maximum to the first value
                    Lperimeter = perimeter
                    Larea = area
                elif ratio > Lratio:    # if larger value is found, update the maximum
                    largest = c
                    Lratio = ratio
                    Lperimeter = perimeter
                    Larea = area

        # Approximating a poligon for chess board edges
        epsilon = 0.1 * Lperimeter  # approximation factor
        chessboard_edge_poly = cv2.approxPolyDP(largest, epsilon, True)

        # DEBUG
        color = (255, 0, 0)
        if self.debug:
            cv2.drawContours(img_contours, [largest, chessboard_edge_poly], -1, color, 2)
            cv2.imshow('Filtered contour', img_contours)
            cv2.waitKey(2000)
            cv2.destroyWindow('Filtered contour')

        # Creating ROI corresponding to the region inside the polyline (i.e. the chessboard)
        roi = cv2.polylines(
            img_contours, [chessboard_edge_poly], True, (255, 0, 0), thickness = 2)

        # Initialize a mask
        mask = np.zeros((img.shape[0], img.shape[1]), 'uint8')
        # Copy chessboard edges as a filled white polygon
        cv2.fillConvexPoly(mask, chessboard_edge_poly, 255, 1)
        # Copy the chessboard to mask
        img_masked = np.zeros_like(img)  # blank image with same shape of the original one
        img_masked[mask == 255] = img[mask == 255]   # fill the ROI with values from original image
        # Add green border to the masked region
        cv2.polylines(img_masked, [chessboard_edge_poly],
                      True, (0, 255, 0), thickness = 5)
        
        # DEBUG
        if self.debug:
            cv2.imshow("Masked chessboard", img_masked)
            cv2.waitKey(2000)
            cv2.destroyWindow("Masked chessboard")

        return img_masked, chessboard_edge_poly, img_contours


    ######################################
    ### CORNERS/CENTERS IDENTIFICATION ###
    ######################################
    # NOTE. Some of the methods in this class are meant to be used somewhere else (i.e. they are meant to be public), such as `segmentation_sequence`. Some other are actually not used, but may be useful even outside the class (e.g. canny or dilation). Some more are just utility methods to be used inside the class (as the one below), thus it is a good habit to prepend a _ to the name.
    def _categorise_lines(self, lines):
        """Classify lines as horizontal or vertical, and then to sort them based on their center (ascending). Used to sort outcome of Hough lines detection.

        Args:
            lines ([perceptionLineClass.Line]): List of Line objects, storing information about the outcome of the Hough transform step.

        Returns:
            Tuple containing two lists of sorted Line objects, one with horizontal lines and one with vertical lines. Line objects are sorted based on their center coordinate in the relevant direction (ascending).
        """
        # Divide horizontal and vertical lines
        horizontal = []
        vertical = []
        for i in range(len(lines)):
            if lines[i].category == 'horizontal':
                horizontal.append(lines[i])
            else:
                vertical.append(lines[i])
        # Sort lines based on the relevant center coordinate
        horizontal = sorted(horizontal, key=operator.attrgetter('centerH'))
        vertical = sorted(vertical, key=operator.attrgetter('centerV'))

        return horizontal, vertical

    def hough_lines(self, edges, img, thresholdx=42, minLineLengthy=100, maxLineGapz=50):
        """Apply Hough transform to detect lines in an image.

        Args:
            edges (np.ndarray): Image on which to run the Hough lines detection. The method is tested to work with images that undergone Canny edge detection and dilation.
            img (np.ndarray): Image on which the outcome of Hough lines detection will be overlayed. Can be same as `edges`.
            thresholdx (int, optional): The threshold value of the Hough lines transform. Defaults to 42.
            minLineLengthy (int, optional): The min line length parameter of the Hough lines transform. Defaults to 100.
            maxLineGapz (int, optional): The max line gap parameter of the Hough lines transform. Defaults to 50.

        Returns:
            A tuple containing two lists of perceptionLineClass.Line objects, one for horizontal and one fo vertical lines, respectively.
        """
        # Detect Hough lines
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
        horizontal, vertical = self._categorise_lines(lines)
        
        # Filter out close lines based to (hopefully) achieve 9
        # NOTE. The threshold values has been experimentally tuned.
        ver = filterClose(vertical, horizontal = False, threshold = 20)
        hor = filterClose(horizontal, horizontal = True, threshold = 20)

        # DEBUG TO SHOW LINES
        if self.debug:
            img_hough = img.copy()
            img_hough = self._drawLines(img_hough, ver)
            img_hough = self._drawLines(img_hough, hor)
            cv2.imshow("Hough lines", img_hough)
            cv2.waitKey(2000)
            cv2.destroyWindow("Hough lines")

        return hor, ver

    def _drawLines(self, img, lines, color=(0, 0, 255), thickness=2):
        """Draws lines. This function was used to debug Hough lines generation.

        Args:
            img (np.ndarray): The image to overlay.
            lines ([perceptionLineClass.Line]): A list of Line object to draw on `img`.
            color (tuple, optional): The (BGR) color of the lines. Defaults to (0, 0, 255).
            thickness (int, optional): The thickness [px] of the lines. Defaults to 2.
        
        Returns:
            np.ndarray: The image with overlayed lines.
        """
        for l in lines:
            l.draw(img, color, thickness)
        
        return img

    # Intersections
    def findIntersections(self, horizontals, verticals, image):
        """Finds intersections between Hough lines and filters out close points.

        Args:
            horizontals ([perceptionLineClass.Line]): A list of Line object, obtained filtering horizontal lines from the output of the Hough transform.
            verticals ([perceptionLineClass.Line]): A list of Line object, obtained filtering vertical lines from the output of the Hough transform.
            image (np.ndarrya): The image on which to overlay the lines for debug prints.

        Returns:
            [tuple]: A list of tuples, each one representing the coordinates in pixel of an intersection point.
        """
        # Find the intersection points
        intersections = []
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
            print("\nWe have found: " + str(len(intersections)) + " intersections.\n")
            img_intersections = image.copy()

            for intersection in intersections:
                cv2.circle(img_intersections, intersection, 10, 255, 1)

            cv2.imshow("Intersections found", img_intersections)
            cv2.waitKey(2000)
            cv2.destroyWindow("Intersections found")
        

        # Filtering intersection points
        min_distance = 10
        # NOTE. The min_distance value has been experimentally tuned. Lowering it can help find corners far from camera, due to perspective distortion.
        
        # Only works if you run it several times -- WHY? Very inefficient
        # Now also works if run only once so comment the loop out
        #for i in range(4):
        for intersection in intersections:
            a = False
            for neighbor in intersections:
                distance_to_neighbour = np.sqrt(
                    (intersection[0] - neighbor[0]) ** 2 + (intersection[1] - neighbor[1]) ** 2)
                # Check that it's not comparing the same ones
                if distance_to_neighbour < min_distance and intersection != neighbor:
                    intersections.remove(neighbor)

        # We still have duplicates for some reason. We'll now remove these
        filtered_intersections = []
        seen = set()    # set of the already encountered points
        for intersection in intersections:
            # If value has not been encountered yet,
            # ... add it to both list and set.
            if intersection not in seen:
                filtered_intersections.append(intersection)
                seen.add(intersection)

        if self.debug:
            print("We have filtered: " + 
                  str(len(filtered_intersections)) + " intersections.")

            img_intersections = image.copy()

            for intersection in filtered_intersections:
                cv2.circle(img_intersections, intersection, 10, (0, 0, 255), 1)
            cv2.imshow("Filtered intersections", img_intersections)
            cv2.waitKey(2000)
            cv2.destroyWindow("Filtered intersections")

        return filtered_intersections

    # Assign Intersection
    def assignIntersections(self, image, intersections):
        """
        Takes the filtered intersections and assigns them to a list containing nine sorted lists, each one representing
        one row of sorted corners. The first list for instance contains the nine corners of the first row sorted
        in an ascending fashion. This function necessitates that the chessboard's horizontal lines are exactly
        horizontal on the camera image, for the purposes of row assignment.
        """
        # Exploit the fact that intersections are ordered according to chessboard pattern, to filter out 'false corners' resulting from the intersection of chessboard lines and the edge of the board frame.
        if len(intersections) == 121:
            del intersections[0:11]
            del intersections[-11:len(intersections)]
            del intersections[0:len(intersections):11]
            del intersections[9:len(intersections):10]
            del intersections[81:]
        elif len(intersections) == 81:
            pass
        else:
            raise ValueError('There are ' + str(len(intersections)) + ' intersections, but they are expected to be:\n * 81 (9x9) for a chessboard without frame\n* 121 (11x11) for a chessboard with frame')

        # DEBUG
        if self.debug:
            corner_cnt = 0
            debug_img = image.copy()
            # for row in corners:
            for corner in intersections:
                cv2.circle(debug_img, corner, 10, (0, 255, 0), 1)
                corner_cnt += 1

            cv2.imshow("Final Corners", debug_img)

            print("\nThere are: " + str(corner_cnt) +
                  " corners that were found.\n")

        # Sorting intersection to an 2D array for chess board pattern and easy square centoid calculation.
        sorted_intersections = [[], [], [], [], [], [], [], [], []]
        k = 0
        if len(intersections) == 81:
            for i in range(9):
                for j in range(9):
                    sorted_intersections[i].append(intersections[k])
                    k += 1
            
        return intersections, sorted_intersections, image
    

    def makeSquares(self, corners,  image, side=True):
        """
        Instantiates the 64 squares given 81 corner points.
        labelledsquare contains centroid of each suare with its x,y and square name
        side takes in False-> Black or True-> White according to TIAGo side
        """

        # List of Square objects
        squares = []
        coordinates = []
        # Lists containing positional and index information
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
        row = []
        count = 0

        for i in range(8):
            for j in range(8):
                # Make the square - yay!
                # position = letters[-i-1] + numbers[-j-1]
                c1 = corners[i][j]
                c2 = corners[i][j+1]
                c3 = corners[i+1][j]
                c4 = corners[i+1][j+1]
                '''check for type conversion'''
                centerx = int((c1[0]+c2[0]+c3[0]+c4[0])/4)
                centery = int((c1[1]+c2[1]+c3[1]+c4[1])/4)

                center = (centerx,
                          centery)
                # print(c1, c2, c3, c4, center)
                squares.append(center)
        # print(squares, len(squares))
        rows = [[], [], [], [], [], [], [], []]
        k = 0
        if len(squares) == 64:
            for i in range(8):
                for j in range(8):
                    rows[i].append(squares[k])
                    k += 1
            # print('sorted row')
            # print(rows,len(rows))

        # DEBUG

        squareCenters = 0
        debugImg = image.copy()
        # for row in corners:
        for center_ in squares:
            # for corner in row:
            # cv2.circle(debugImg, corner, 10, (0,255,0), 1)
            # cornerCounter += 1
            cv2.circle(debugImg, center_, 5, (0, 255, 0), -1)
            squareCenters += 1
        if self.debug:
            cv2.imshow("Final centers", debugImg)

            print("")
            print("There are: " + str(squareCenters) +
                  " centers that were found.")
            print("")

        # considering TIAGo playing as White
        labeledSquares = []
        alpha = 0
        num = 0

        # for TIAGo playing as WHITE
        if side:
            numbers.sort(reverse=True)
            for center_ in squares:

                labeledSquares.append((center_[0], center_[
                    1], letters[alpha]+numbers[num]))
                alpha += 1
                if alpha > 7:
                    alpha = 0
                    if num != 7:
                        num += 1

        # for Tiago playing as BLACK
        else:
            letters.sort(reverse=True)
            for center_ in squares:

                labeledSquares.append((center_[0], center_[
                    1], letters[alpha]+numbers[num]))
                alpha += 1
                if alpha > 7:
                    alpha = 0
                    if num != 7:
                        num += 1

        # DEbug
        if self.debug:
            debugImg1 = image.copy()
            squareCenters = 0
            # for row in corners:
            for center_ in labeledSquares:
                # for corner in row:
                # cv2.circle(debugImg, corner, 10, (0,255,0), 1)
                # cornerCounter += 1
                cv2.circle(debugImg1, (center_[0], center_[
                           1]), 5, (0, 255, 0), -1)
                cv2.putText(debugImg1, center_[2], (center_[0], center_[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1)
                squareCenters += 1
            cv2.imshow("Labeled centers", debugImg1)

            print(labeledSquares, len(labeledSquares))

        #         square.draw(image)

        #         index += 1
        #         # print(index)
        #         # xyz = square.getDepth(depthImage)
        #         # coordinates.append(xyz)

        # cv2.imshow("Board Identified", image)

        # if self.debug:
        #     cv2.imwrite("5SquaresIdentified.jpeg", image)

        # # Get x,y,z coordinates from square centers & depth image
        # # coordinates = self.getDepth(square.roi, depthImage)
        # # DEBUG
        # if self.debug:
        #     print("Number of Squares found: " + str(len(squares)))

        return squares, labeledSquares, rows, debugImg

    # Trackbar for setting the Hough parameters

    def track_bar(self, img, dilated_img, intersections):
        '''
        for mannually setting up values of Hough parameter to corretly detect 91  square corners
        '''
        def call_back(x):
            pass

        cv2.namedWindow('adjust_Hough_parameter', cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(
            'threshold', 'adjust_Hough_parameter', 42, 200, call_back)
        cv2.createTrackbar('min_line_length',
                           'adjust_Hough_parameter', 100, 200, call_back)
        cv2.createTrackbar(
            'max_line_gap', 'adjust_Hough_parameter', 50, 200, call_back)
        cv2.createTrackbar('off', 'adjust_Hough_parameter', 0, 1, call_back)
        switch = '0 : NOT_ACTIVE \n 1 : ACTIVE'
        cv2.createTrackbar(switch, 'adjust_Hough_parameter', 0, 1, call_back)
        off = 0
        while True:
            track_img = img.copy()
            cornerCounter = 0
            for corner in intersections:
                cv2.circle(track_img, corner, 10, (0, 255, 255), 1)
                cornerCounter += 1
            cv2.imshow('adjust_Hough_parameter', track_img)
            cv2.waitKey(30)
            s = cv2.getTrackbarPos(switch, 'adjust_Hough_parameter')

            if s == 0:
                pass
            else:
                off = cv2.getTrackbarPos('off', 'adjust_Hough_parameter')
                if off == 1:
                    break
                thresholdx = cv2.getTrackbarPos(
                    'threshold', 'adjust_Hough_parameter')
                minLineLengthy = cv2.getTrackbarPos(
                    'min_line_length', 'adjust_Hough_parameter')
                maxLineGapz = cv2.getTrackbarPos(
                    'max_line_gap', 'adjust_Hough_parameter')

                hor, ver = self.hough_lines(
                    dilated_img, img, thresholdx, minLineLengthy, maxLineGapz)
                intersections = self.findIntersections(hor, ver, img)
                corners, sorted_intersections, image = self.assignIntersections(
                    img, intersections)
        return corners, sorted_intersections
    
    def tune_hough_params(self, img, dilated_img):
        '''
        for automatically setting up values of Hough parameter to corretly detect 91  square corners
        '''
        def hough_lines_eval(threshold, min_line_length, max_line_gap):
            hor, ver = self.hough_lines(dilated_img, img, 
                                       int(threshold), 
                                       int(min_line_length), 
                                       int(max_line_gap)
                                       )
            intersections = self.findIntersections(hor, ver, img)
            # The evaluation function exploits the prior knowledge of the chessboard geometry, i.e. we want exactly 121 corners (81 of the actual chessboard + intersections with chessboard frame)
            return abs(len(intersections) - 121) * (-1)
            # NOTE. The above evaluation function makes the method appliable only to chessboard with an external frame. To extend it to any chessboard, try to include also the assignIntersections function and check that the output sorted_intersections is 9x9.

        # Setup the Bayesian Optimization process
        params = {'threshold':          (42, 200),
                  'min_line_length':    (100, 200),
                  'max_line_gap':       (50, 200)
                 }
        bo = BayesianOptimization(f = hough_lines_eval,
                                  pbounds = params,
                                  random_state = 1
                                  )
        
        # Mute the debug to avoid having too many prints during the learning process
        # NOTE. It could be a nice idea too show how the intersection detection changes throughout the optimization process. As all the tested parameter sets are contained in `bo`, this can be easily done offline.
        debug_tmp = self.debug
        self.debug = False

        # Run 10 iterations of Bayesian Optimization
        HOUGH_TUNING_ITERS_LIMIT = 100
        ITER_PER_BATCH = 10
        bo.maximize(init_points = 5,
                    n_iter = ITER_PER_BATCH,
                    )
        iter = ITER_PER_BATCH
        # If none of the stopping criteria (no. of iterations or convergence to optimum) is met, run 10 further iterations of Bayesian Optimization
        while not bo.max['target'] == 0:            # check convergence to optimum
            if iter >= HOUGH_TUNING_ITERS_LIMIT:    # check no. of iterations
                print('The limit of ' + str(HOUGH_TUNING_ITERS_LIMIT) + ' has been reached and the optimum has not been found...')
                break
            # Run 10 iterations of Bayesian Optimization
            bo.maximize(init_points = 0,
                        n_iter = ITER_PER_BATCH,
                        )
            iter += ITER_PER_BATCH  # update number of iterations done
        
        #print(bo.max)
        #print(len(bo.res))

        # Restore the previous debug value
        self.debug = debug_tmp  

        # Use the parameters found by the optimizer to run the chessboard segmentation.
        # TODO. What happens if the BO is not able to find an optimal set of params?
        hor, ver = self.hough_lines(dilated_img, img, 
                                   int(bo.max['params']['threshold']),
                                   int(bo.max['params']['min_line_length']),
                                   int(bo.max['params']['max_line_gap']) 
                                   )
        intersections = self.findIntersections(hor, ver, img)
        corners, sorted_intersections, image = self.assignIntersections(
                    img, intersections)
        
        return corners, sorted_intersections


    def segmentation_sequence(self, img):
        # Apply Gaussian blurring, convert the image to greyscale and apply adaptive thresholding
        img_thresholded = self.preprocessing(img)
        # Perform edge detection on the thresholded image
        edge_img = self.canny_edge_detection(img_thresholded)
        # Dilate the edges
        dilated_edge_img = self.dilation(edge_img, kernel_size=(5, 5), iterations=5)
        # Chessboard edges used to eliminate unwanted point in assign intersections function
        contoured_img, chessBoardEdges, annotated_image_vertices = self.chessboard_contour(
            img, dilated_edge_img)
        canny_img = self.canny_edge_detection(contoured_img)
        '''
        hor, ver = self.hough_lines(canny_img, img)
        intersections = self.findIntersections(hor, ver, img)
        corners, sorted_intersection, image = self.assignIntersections(
            img, intersections)
        corners, sorted_intersection = self.track_bar(img, canny_img, corners)
        '''
        corners, sorted_intersection = self.tune_hough_params(img, canny_img)
        squares, labelledSquares, rows, annotated_image = self.makeSquares(
            sorted_intersection, img, side=True)
        chessBoardEdgesS_with_out_borders = [
            corners[0], corners[8], corners[80], corners[72]]
                # save edges of chess board with out corners for detecting moves in chess_move_detection
        with open(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                  'yaml', 'store_chess_board_edges.yaml'), 
                  'w') as file:
            documents = yaml.dump(chessBoardEdgesS_with_out_borders, file)
        
        return rows, annotated_image, annotated_image_vertices, len(squares), chessBoardEdges, chessBoardEdgesS_with_out_borders


if __name__ == "__main__":
    print('This script implement the ImageProcessing class, which is meant to be as image processing utility in a ROS node. As the script has been executed as main, a demo of the funtioning on a test image is shown.')

    DEBUG = True    # Set it to True to display the outcome of each processing step.

    # Import the test image to run the demo processing.
    FILE = 'windows_open.png'   # the name of the file used for processing demo  
    image = cv2.imread(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'Static_images', FILE))
    image_processing = ImageProcessing(debug = DEBUG)
    # Run the chessboard segmentation and cell's centers identification
    __, img_out_seg, img_in_seg, __, __, __ = image_processing.segmentation_sequence(image)
    # Display the results
    cv2.imshow('Result', img_out_seg)
    cv2.waitKey(5000)   #[ms]
    cv2.destroyAllWindows()