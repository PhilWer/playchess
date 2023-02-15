#!/usr/bin/env python3

"""As a unique set of Hough transform parameters is hard to be identified experimentally, two tuning methods are implemented:
- manual, based on sliders in a graphical interface
- automatic, based on Bayesian Optimization
"""

import os
import cv2
import numpy as np
import operator
import yaml
from bayes_opt import BayesianOptimization

# User defined classes
from useful_functions.perceptionLineClass import Line, filterClose
# NOTE. if the Line and filterClose classes come from the Franka example, please put in the corresponding file some reference to that work (e.g. a comment on top stating `Taken from: <URL>` or `Adapted from: <URL>`). This is a way to acknowledge the authors, and also very good for traceback in case of malfunctioning. 

import rospy
import ros_numpy
from playchess.srv import HoughTuning, HoughTuningResponse

class HoughTuning():
    """Class containing functions to process chessboard images and detect squares.
    """
    def __init__(self, hough_autotune = True):
        # Set the desired tuning modality
        self.hough_autotune = hough_autotune    # True:  Bayesian Optimization
                                                # False: Manual tuning

        if hough_autotune:
            s = rospy.Service('playchess/hough_tuning', HoughTuning, self.track_bar)
        else:
            s = rospy.Service('playchess/hough_tuning', HoughTuning, self.tune_hough_params)

    ######################################
    ### CORNERS/CENTERS IDENTIFICATION ###
    ######################################
    def hough_lines(self, edges, img, threshold = 42, min_line_length = 100, max_line_gap = 50):
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
        lines = cv2.HoughLinesP(edges, rho = 1, theta = 1 * np.pi / 180, threshold = threshold,
                                minLineLength = min_line_length, maxLineGap = max_line_gap)
        if lines is None:   # i.e. if there are no lines
            return [], []    
        else:
            N = lines.shape[0]

        # Draw lines on image
        new = []
        for i in range(N):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]

            new.append([x1, y1, x2, y2])

        lines = [Line(x1=new[i][0], y1=new[i][1], x2=new[i][2],
                      y2=new[i][3]) for i in range(len(new))]

        # Categorise the lines into horizontal or vertical
        horizontal, vertical = self._categorise_lines(lines)
        
        # Filter out close lines based to (hopefully) achieve 9
        # NOTE. The threshold values has been experimentally tuned.
        ver = filterClose(vertical, horizontal = False, threshold = 20)
        hor = filterClose(horizontal, horizontal = True, threshold = 20)

        return hor, ver
    

    ### Intersections utils
    def find_intersections(self, horizontals, verticals):
        """Finds intersections between Hough lines and filters out close points.
        Args:
            horizontals ([perceptionLineClass.Line]): A list of Line object, obtained filtering horizontal lines from the output of the Hough transform.
            verticals ([perceptionLineClass.Line]): A list of Line object, obtained filtering vertical lines from the output of the Hough transform.
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

        # Filtering intersection points
        min_distance = 10
        # NOTE. The min_distance value has been experimentally tuned. Lowering it can help find corners far from camera, due to perspective distortion.
        
        # Only works if you run it several times -- WHY? Very inefficient
        # Now also works if run only once so comment the loop out
        for intersection in intersections:
            a = False
            for neighbor in intersections:
                distance_to_neighbour = np.sqrt(
                    (intersection[0] - neighbor[0]) ** 2 + (intersection[1] - neighbor[1]) ** 2)
                # Check that it's not comparing the same ones
                if distance_to_neighbour < min_distance and intersection != neighbor:
                    intersections.remove(neighbor)

        # We still have duplicates for some reason. We'll now remove these
        intersections_filtered = []
        seen = set()    # set of the already encountered points
        for intersection in intersections:
            # If value has not been encountered yet,
            # ... add it to both list and set.
            if intersection not in seen:
                intersections_filtered.append(intersection)
                seen.add(intersection)

        return intersections_filtered

    '''
    def assign_intersections(self, intersections):
        """Takes the filtered intersections and assigns them to a list containing nine sorted lists, each one representing one row of sorted corners. The first list for instance contains the nine corners of the first row sorted in an ascending fashion.
        Args:
            intersections [tuple]: A list of tuples, each one representing the coordinates in pixel of an intersection point.
        Raises:
            ValueError: If the length of `intersections` is neither 121 (11x11, framed chessboard) nor 81 (9x9 chessboard without frame).
        Returns:
            A tuple containing the same `intersections` given as input, excluding the intersections with the edge of the frame (if any); and the intersections ordered according to the chessboard pattern (i.e. list of 9 lists, each one containing 9 tuples with intersections coordinates).
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
            raise ValueError('There are ' + str(len(intersections)) + ' intersections, but they are expected to be:\n* 81 (9x9) for a chessboard without frame\n* 121 (11x11) for a chessboard with frame')

        # Sorting intersection to an 2D array for chess board pattern and easy square centoid calculation.
        sorted_intersections = [[], [], [], [], [], [], [], [], []]
        k = 0
        if len(intersections) == 81:
            for i in range(9):
                for j in range(9):
                    sorted_intersections[i].append(intersections[k])
                    k += 1
            
        return intersections, sorted_intersections
    '''

    ### Hough lines parameters tuning
    def track_bar(self, msg):
        """Manually tune the Hough transform parameters using sliders on a graphical interface.

        Args:
            msg (playchess/HoughTuningRequest): ...

        Returns:
            A tuple containing all the corners detected with the optimal set of parameters, excluding the intersections with the edge of the frame (if any); and the intersections ordered according to the chessboard pattern (i.e. list of 9 lists, each one containing 9 tuples with intersections coordinates).
        """
        def callback(x):
            pass
        
        # Get the image(s) to process
        img, img_edge = self._unpack_msg(msg)

        #           name:               (default, max)
        params = {'threshold':          (40, 200),
                  'min_line_length':    (100, 200),
                  'max_line_gap':       (50, 200)
                 }

        # Create the window displaying the image and the track bars to tune the parameters
        window_name = 'Hough parameters tuning'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        for param_name in params:
            cv2.createTrackbar(param_name, window_name, *params[param_name], callback)
            # NOTE. Depending on the OpenCV version, the above line may throw a warning.
        
        # Add one trackbar that allows to exit the loop
        switch = '0 : change params and update image \n 1 : close window and confirm params'
        cv2.createTrackbar(switch, window_name, 0, 1, callback)
        while True:
            track_img = img.copy()
            # Find the intersections with the current set of parameters
            threshold, min_line_length, max_line_gap = [cv2.getTrackbarPos(param_name, 
                                                                            window_name)
                                                        for param_name in params
                                                        ]
            hor, ver = self.hough_lines(img_edge, img, 
                                        threshold, min_line_length, max_line_gap
                                        )
            intersections = self.find_intersections(hor, ver, img)
            #corners, intersections_sorted = self.assign_intersections(img, 
            #                                                          intersections
            #                                                         )
            for corner in intersections:
                cv2.circle(track_img, corner, 10, (0, 255, 255), 1)
            cv2.imshow(window_name, track_img)
            cv2.waitKey(30) # refresh rate in [ms]

            off = cv2.getTrackbarPos(switch, window_name)
            if off == 1:
                cv2.destroyWindow(window_name)  # close the param tuning window
                break

        return HoughTuningResponse([int(threshold), 
                                    int(min_line_length), 
                                    int(max_line_gap)
                                    ])
    
    def tune_hough_params(self, msg):
        """Automatically tune the Hough transform parameters exploting a Bayesian Optimization algorithm.

        Args:
            TODO.

        Returns:
            A tuple containing all the corners detected with the optimal set of parameters, excluding the intersections with the edge of the frame (if any); and the intersections ordered according to the chessboard pattern (i.e. list of 9 lists, each one containing 9 tuples with intersections coordinates).
        """
        def hough_lines_eval(threshold, min_line_length, max_line_gap):
            hor, ver = self.hough_lines(img_edge, img, 
                                       int(threshold), 
                                       int(min_line_length), 
                                       int(max_line_gap)
                                       )
            intersections = self.find_intersections(hor, ver, img)
            # The evaluation function exploits the prior knowledge of the chessboard geometry, i.e. we want exactly 121 corners (81 of the actual chessboard + intersections with chessboard frame)
            return abs(len(intersections) - 121) * (-1)
            # NOTE. The above evaluation function makes the method appliable only to chessboard with an external frame. To extend it to any chessboard, try to include also the assign_intersections function and check that the output sorted_intersections is 9x9.

        # Get the image(s) to process
        img, img_edge = self._unpack_msg(msg)

        # Setup the Bayesian Optimization process
        #           name:               (min, max)
        params = {'threshold':          (0, 200),
                  'min_line_length':    (0, 200),
                  'max_line_gap':       (0, 200)
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
                #break
                return -1
            # Run 10 iterations of Bayesian Optimization
            bo.maximize(init_points = 0,
                        n_iter = ITER_PER_BATCH,
                        )
            iter += ITER_PER_BATCH  # update number of iterations done
        
        return HoughTuningResponse([int(bo.max['params']['threshold']), 
                                    int(bo.max['params']['min_line_length']),
                                    int(bo.max['params']['max_line_gap'])
                                    ]) 
    
    
    #############
    ### UTILS ###
    #############
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
    

    def _unpack_msg(self, msg):
        img      = ros_numpy.numpify(msg.img) 
        img_edge = ros_numpy.numpify(msg.img_edge)
        return img, img_edge

if __name__ == "__main__":
    print('This script implement the ImageProcessing class, which is meant to be as image processing utility in a ROS node. As the script has been executed as main, a demo of the funtioning on a test image is shown.')

    rospy.init_node('hough_tuning')
    ht = HoughTuning(hough_autotune = True)

    try:
        while not rospy.is_shutdown():
            pass
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down the node for tuning of Hough lines transform params.')
