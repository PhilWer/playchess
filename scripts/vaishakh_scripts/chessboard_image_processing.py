#!/usr/bin/env python3

# Rospy for the subscriber
import rospy
import numpy as np
import math
import sys
import operator
import yaml
# ROS Image messages
# from sensor_msgs.msg import CompressedImage, Image
# ROS Image message --> Open CV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# user defined classes
from useful_functions.homographic_transformation import HOMO_TRANSFOR as hf
from useful_functions.perceptionLineClass import Line, filterClose

# Open CV2 for saving an image
import cv2
import os
from bayes_opt import BayesianOptimization

bridge = CvBridge()
counter = 0


PLAYCHESS_PKG_DIR = '/home/luca/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/Static_images'


class ImageProcessing():
    # Class containing functions to process chessboard images and detect squares.
    # oder of execution of functions defined in segmentation oder function just above if __name__ line
    def __init__(self):
        self.debug = 0

    # Canny edge function
    def cannyEdgeDetection(self, img):
        '''
        canny edge detection
        1. used in image_analysis befor dilation and then findcountour
        1. also used before hough lines       '''
        edges = cv2.Canny(img,
                          25,    # min_threshold: adjusting it helps to distinguish beteween brown boundary and black  square near to it
                          255
                          )

        # DEBUG
        if self.debug:
            cv2.imshow('Canny', edges)
            cv2.waitKey(0)

        return edges

    def dilation(self, image, kernel_size=(1, 1), iterations=1):
        '''
        to improve against small light variation befor chess board countour detection
        '''
        kernel = np.ones(kernel_size, np.uint8)
        dilated = cv2.dilate(image, kernel, iterations)
        if self.debug:
            cv2.imshow('Edge dilation', dilated)
            cv2.waitKey(0)
        return dilated

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

    def houghLines(self, edges, img, thresholdx=42, minLineLengthy=100, maxLineGapz=50):
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
        # print(len(ver))
        # print(len(hor))
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
                a = False
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
        del intersections[0:11]
        del intersections[-11:len(intersections)]
        del intersections[0:len(intersections):11]
        del intersections[9:len(intersections):10]
        del intersections[81:]

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

        # sorting intersection to an 2D array for chess board pattern and easy square centoid calculation
        sorted_intersections = [[], [], [], [], [], [], [], [], []]
        k = 0
        if len(intersections) == 81:
            for i in range(9):
                for j in range(9):
                    sorted_intersections[i].append(intersections[k])
                    k += 1
            # print('sorted intersection')
            # print(sorted_intersections)
        return intersections, sorted_intersections, image

        """
    SQUARE INSTANTIATION
    """

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

                hor, ver = self.houghLines(
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
            hor, ver = self.houghLines(dilated_img, img, 
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

        # Use the parameters found by the optimizer to run the chessboard segmentation.
        # TODO. What happens if the BO is not able to find an optimal set of params?
        hor, ver = self.houghLines(dilated_img, img, 
                                   int(bo.max['params']['threshold']),
                                   int(bo.max['params']['min_line_length']),
                                   int(bo.max['params']['max_line_gap']) 
                                   )
        intersections = self.findIntersections(hor, ver, img)
        corners, sorted_intersections, image = self.assignIntersections(
                    img, intersections)
        
        return corners, sorted_intersections

    # Image_preprocessing

    def preprocessing(self, img, kernelG=5):
        '''
        preprocessing image for noise reduction and prepration for corner detection of chess board
        '''
        img = cv2.GaussianBlur(img, (kernelG, kernelG), 0)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        # NOTE. The if statement makes it (hopefully) compatible with all the OpenCV versions.
        # From: https://stackoverflow.com/questions/20851365/opencv-contours-need-more-than-2-values-to-unpack
        if cv2.getVersionMajor() in [2, 4]:
            # OpenCV 2, OpenCV 4 case
            contours, hierarchy = cv2.findContours(preprocessed_img.copy(), 
                                                  cv2.RETR_TREE, 
                                                  cv2.CHAIN_APPROX_NONE
                                                  )
        else:
            # OpenCV 3 case
            image, contours, hierarchy = cv2.findContours(preprocessed_img.copy(), 
                                                         cv2.RETR_TREE, 
                                                         cv2.CHAIN_APPROX_NONE
                                                        )

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
            imgContours, [chessboardEdge], True, (255, 0, 0), thickness=2)
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

        return extracted, chessboardEdge, imgContours

    def segmentation_sequence(self, img):
        preprocessed_img = self.preprocessing(img)
        pre_canny = self.cannyEdgeDetection(preprocessed_img)
        dilated = self.dilation(pre_canny, kernel_size=(5, 5), iterations=5)
        # Chessboard edges used to eliminate unwanted point in assign intersections function
        analysed_img, chessBoardEdges, annotated_image_vertices = self.image_analysis(
            img, dilated)
        canny_img = self.cannyEdgeDetection(analysed_img)
        '''
        hor, ver = self.houghLines(canny_img, img)
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
        with open(r'/home/luca/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/yaml/store_chess_board_edges.yaml', 'w') as file:
            documents = yaml.dump(chessBoardEdgesS_with_out_borders, file)
        # hft_img = transformed_chess_board.transform()
        return rows, annotated_image, annotated_image_vertices, len(squares), chessBoardEdges, chessBoardEdgesS_with_out_borders


def main():
    # rospy.init_node('image_processor')
    # /home/silvia/tiago_public_ws/src/tiago_playchess/Images_chessboard_empty/camera_image1.jpeg
    #image = cv2.imread(PLAYCHESS_PKG_DIR + '/test.png')
    # image = cv2.imread(
    #     '/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/Static_images/Image_move_detection/test.png')
    image = cv2.imread(
        '/home/luca/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/Static_images/windows_open.png')
    image_processing = ImageProcessing()
    __, img_out_seg, img_in_seg, __, __, __ = image_processing.segmentation_sequence(image)
    cv2.imshow('Result', img_out_seg)
    cv2.waitKey(5000)   #[ms]
    cv2.destroyAllWindows()


 # MAIN
if __name__ == "__main__":

    main()
    #############################################################################
    # transformed_chess_board = hf(img, chessBoardEdges, True)
    # hft_img = transformed_chess_board.transform()
    # canny_img = ip.cannyEdgeDetection(hft_img)
    # cv2.imshow('canny', canny_img)
    # hor, ver = ip.houghLines(canny_img, hft_img)
    # intersections = ip.findIntersections(hor, ver, hft_img)
    # corners, sorted_inersectioin, image = ip.assignIntersections(
    #     hft_img, intersections)
    # corners, sorted_inersectioin = ip.track_bar(hft_img, canny_img, corners)
    # squares = ip.makeSquares(sorted_inersectioin, hft_img, hft_img)

    #############################################################################