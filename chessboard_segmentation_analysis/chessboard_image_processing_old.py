#!/usr/bin/env python

# Rospy for the subscriber
# import rospy
import numpy as np
import math
import sys
# ROS Image messages
# from sensor_msgs.msg import CompressedImage, Image
# ROS Image message --> Open CV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# Open CV2 for saving an image
import cv2
import os
import yaml
# import imutils


bridge = CvBridge()
counter = 0


class ImageProcessing:
    # Class containing functions to process chessboard images and detect squares.
    def __init__(self):
        self.verbose = False

    def grayscale(self, image):
        # Conversion to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Show th original image and the grayscale image
        if self.verbose:
            cv2.imshow('Original image', image)
            cv2.imwrite(os.path.join(
                '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '1-original.png'), image)
            cv2.imshow('Gray image', gray)
            cv2.imwrite(os.path.join(
                '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '2-gray.png'), gray)
        return gray

    def gaussian_filter(self, image, kernel):
        # Filter the image with a Gaussian filter witht he specified kernel. The standard deviation in x an y directions is calculated starting from the kernel size (if not specifically given as inputs).
        gaussian = cv2.GaussianBlur(image, kernel, 0)
        if self.verbose:
            cv2.imshow('Gaussian filtered', gaussian)
            cv2.imwrite(os.path.join(
                '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '3-gaussian.png'), gaussian)
        return gaussian

    def otsu_thresholding(self, image, T):
        ret, otsu = cv2.threshold(
            image, T, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if self.verbose:
            cv2.imshow('Otsu binarized', otsu)
            cv2.imwrite(os.path.join(
                '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '4-otsu.png'), otsu)
        return otsu

    def canny_edge(self, image, thresh1, thresh2):
        canny = cv2.Canny(image, thresh1, thresh2)
        if self.verbose:
            cv2.imshow('Edge detection', canny)
            cv2.imwrite(os.path.join(
                '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '5-canny.png'), canny)
        return canny

    def dilation(self, image, kernel_size):
        kernel = np.ones(kernel_size, np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        if self.verbose:
            cv2.imshow('Edge dilation', dilated)
            cv2.imwrite(os.path.join(
                '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '6-dilation.png'), dilated)
        return dilated

    def squares_contouring(self, image, min_square_len, max_square_len, grayscale, chessboard):
        # Function to find and draw contours of objects in a scene.
        # chessboard is True if i am segmenting only the chessboard, false if I am segmenting the chessboard squares
        count_squares = 0
        count = 0
        found = 0  # Number of squared selected and drawn
        # _ , contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
        # REMEMBER: if not using TIAGo add "__," before "contours"
        contours, hierarchy = cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        M = np.zeros(len(contours))
        cX = np.zeros(len(contours), dtype=int)
        cY = np.zeros(len(contours), dtype=int)
        dist = np.zeros((len(contours), len(contours)))
        squares_centers = []
        row = []  # A list to momentaneously contain the squares from the same row
        # A list to contain the squares of the same row in the same element of the list (array)
        rows = []
        last_row = []
        x_distance = []
        x_distance_new = []
        x_distances = []
        x_distances_new = []
        number_squares = []
        right_square = [0, 0]
        left_square = [1000, 1000]
        more_centers = []
        all_new_centers = []
        all_new_centers_border = []
        new_coord = []
        final_squares_number = 0
        # Sort the found contours and put the one with the bigger area the beginning of the array
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Computation of the centroids of the identified contours
        for c in contours:
            M = cv2.moments(c)
            cX[count] = int(M["m10"] / M["m00"])
            cY[count] = int(M["m01"] / M["m00"])
            count = count + 1

        for b in range(0, len(cX)):
            for a in range(0, len(cX)):
                # Matrix of distances between each centroid
                dist[b][a] = math.sqrt((cX[b] - cX[a])**2 + (cY[b] - cY[a])**2)
        squares = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

        # print('FIRST')
        # print(len(contours))

        if chessboard:  # If I need to isolate only the chessboard
            # Centroid computation
            cX1 = cX[0]
            cY1 = cY[0]
            squares1 = cv2.drawContours(squares, contours, 0, (0, 0, 255), 2)
            if self.verbose:
                cv2.imshow('Chessboard border', squares1)
                cv2.imwrite(os.path.join(
                    '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '7-chessboardBorder.png'), squares1)

            chessboard_contour = contours[0]
            peri = cv2.arcLength(chessboard_contour, True)
            self.approx = cv2.approxPolyDP(
                chessboard_contour, 0.1 * peri, True)
            self.approx_square = cv2.drawContours(
                squares, self.approx, 0, (0, 0, 255), 2)
            for o in range(0, len(self.approx)):
                self.approx_square = cv2.circle(
                    self.approx_square, (self.approx[o][0][0], self.approx[o][0][1]), 5, (255, 0, 0), -1)
            if not self.verbose:
                # print('Corners of the approximated chessboard: ' + str(self.approx))
                # Draw the new centers in green
                cv2.imshow('Approx chessboard', self.approx_square)
                cv2.imwrite(os.path.join('/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione',
                            '8-approxChessboard.png'), self.approx_square)

        if not chessboard:  # If I need to compute the squares centroids
            for c in contours:
                if (cv2.contourArea(c)) < 3000 and (cv2.contourArea(c)) > 100:
                    # Find the convex hull of the foud contours, to exclude all the wrong contours
                    hull = cv2.convexHull(c)
                    if cv2.contourArea(hull) > 3000:
                        contours.remove(c)
                    else:
                        index = count_squares
                        squares = cv2.drawContours(
                            squares, contours, index, (0, 0, 255), 1)
                        # squares = cv2.circle(squares, (cX[index], cY[index]), 5, (0, 0, 255), -1)
                        squares_centers.append([cX[index], cY[index]])
                        found = found + 1
                count_squares = count_squares + 1

            # print('SECOND')
            # print(len(contours))

            # Complete the computation of the centers of all the squares of the chessboard
            # Sort the list of centers based on the y coordinates
            squares_centers.sort(key=self.take_second)
            # Delete the squares that are repeated
            save_index = []
            continue_search = True
            happened = False
            while continue_search:
                for i in range(0, len(squares_centers)):
                    for j in range(0, len(squares_centers)):
                        if i != j:
                            if abs(squares_centers[i][0]-squares_centers[j][0]) < 30 and abs(squares_centers[i][1]-squares_centers[j][1]) < 10:
                                happened = True
                                squares_centers[i][0] = (
                                    squares_centers[i][0]+squares_centers[j][0])//2
                                save_index.append(squares_centers[j])
                            if abs(squares_centers[i][0]-squares_centers[j][0]) == 0 and abs(squares_centers[i][1]-squares_centers[j][1]) == 0:
                                happened = True
                                squares_centers[j][0] = squares_centers[j][0] + 1000
                                save_index.append(squares_centers[j])
                for elem in squares_centers:
                    if elem in save_index:
                        squares_centers.remove(elem)
                if not happened:
                    continue_search = False
                happened = False
                save_index = []

            # print('THIRD')
            # print(squares_centers)

            # Create the list of squares on the same row
            for i in range(1, len(squares_centers)):
                # and (abs(squares_centers[i][0] - squares_centers[i-1][0]) < 100): #It means that the 2 points are centroids of squares of the same row.  and (abs(squares_centers[i][0] - squares_centers[i-1][0]) > 40)
                if ((squares_centers[i][1] - squares_centers[i-1][1]) < 10):
                    if not row:
                        row.append(squares_centers[i-1])
                        row.append(squares_centers[i])
                    else:
                        row.append(squares_centers[i])
                        last_row = row
                else:  # If the next centroid is not in the same row anymore, empty the row list
                    if row:
                        rows.append(row)
                    row = []
            rows.append(last_row)

            # print('ROWS')
            # print(rows)

            # Compute the number of squares found for each row and the distances in the x direction between the found squares, so that I can reconstruct the missing centers.
            # FORSE DOVRO' SCRIVERE UN CONTROLLO SUL NUMERO DI RIGHE TROVATE FINORA
            save_index = []
            for i in range(0, len(rows)):
                # List of numbers of squares found for each row
                number_squares.append(len(rows[i]))
                # Order the centers of the row from smallest x to biggest
                rows[i].sort(key=self.take_first)
                for j in range(1, len(rows[i])):
                    # List of distances on the x direction
                    x_distance.append(rows[i][j][0]-rows[i][j-1][0])
                x_distances.append(x_distance)
                x_distance = []
            if self.verbose:
                print('Number squares found at the beginning:')
                print(number_squares)

            # Draw the centers in red
            for o in range(0, len(rows)):
                for d in range(0, len(rows[o])):
                    squares = cv2.circle(
                        squares, (rows[o][d][0], rows[o][d][1]), 5, (0, 0, 255), -1)
            squares = cv2.circle(
                squares, (cX[index], cY[index]), 5, (0, 0, 255), -1)

            # Save the squares with the further right and further left x coordinates
            for t in range(0, len(squares_centers)):
                if squares_centers[t][0] > right_square[0]:
                    right_square = squares_centers[t]
                if squares_centers[t][0] < left_square[0]:
                    left_square = squares_centers[t]

            number_with_new_squares = number_squares
            for r in range(0, len(x_distances)):
                for q in range(0, len(x_distances[r])):
                    # There's a square that has not been identified
                    if x_distances[r][q] > 55 and x_distances[r][q] < 110:
                        coord1 = rows[r][q]
                        coord2 = rows[r][q+1]
                        new_coord = [(coord1[0] + coord2[0])//2,
                                     (coord1[1] + coord2[1])//2]
                        more_centers.append(new_coord)
                        number_with_new_squares[r] = number_with_new_squares[r] + 1
                all_new_centers.append(more_centers)
                more_centers = []
            if self.verbose:
                print('Squares number found after the first reconstruction:')
                print(number_with_new_squares)

            # Draw the new centers in green
            for o in range(0, len(all_new_centers)):
                for d in range(0, len(all_new_centers[o])):
                    squares = cv2.circle(
                        squares, (all_new_centers[o][d][0], all_new_centers[o][d][1]), 5, (0, 255, 0), -1)

            # Add the new centers  to the totality of the squares and compute the new distances
            for r in range(0, len(all_new_centers)):
                if all_new_centers[r]:  # If new centers were found, compute new distances
                    for g in range(0, len(all_new_centers[r])):
                        rows[r].append(all_new_centers[r][g])
                rows[r].sort(key=self.take_first)  # Sort the centers
                for j in range(1, len(rows[r])):
                    if (rows[r][j][0]-rows[r][j-1][0]) != 0:
                        # List of distances on the x direction
                        x_distance_new.append(rows[r][j][0]-rows[r][j-1][0])
                x_distances_new.append(x_distance_new)
                x_distance_new = []

            all_new_centers_border = []
            not_yet = True
            conta = 0
            ind = []
            flag_toomany = False
            while not_yet:
                for y in range(0, len(number_with_new_squares)):
                    # If I still havent't found 8 squares for the row
                    if number_with_new_squares[y] < 8:
                        if y < 3:  # If I'm considering the last 4 rows of the chessboard
                            # In this case we're missing the left square of the row
                            if (rows[y][0][0] - left_square[0] > 55):
                                new_coord = [
                                    rows[y][0][0] - x_distances_new[y][0], rows[y][0][1]]
                                more_centers.append(new_coord)
                                rows[y].append(new_coord)
                                # Sort the centers
                                rows[y].sort(key=self.take_first)
                                number_with_new_squares[y] = number_with_new_squares[y] + 1
                            # In this case we're missing the left square of the row
                            if (abs(rows[y][-1][0] - right_square[0]) > 55) and number_with_new_squares[y] < 8:
                                new_coord = [
                                    rows[y][-1][0] + x_distances_new[y][-1], rows[y][-1][1]]
                                more_centers.append(new_coord)
                                rows[y].append(new_coord)
                                # Sort the centers
                                rows[y].sort(key=self.take_first)
                                number_with_new_squares[y] = number_with_new_squares[y] + 1
                        else:  # If I'm considering the rows of the chessboard closest to TIAGo
                            # In this case we're missing the left square of the row
                            if (rows[y][0][0] - left_square[0] > 25):
                                new_coord = [
                                    rows[y][0][0] - x_distances_new[y][0], rows[y][0][1]]
                                more_centers.append(new_coord)
                                rows[y].append(new_coord)
                                # Sort the centers
                                rows[y].sort(key=self.take_first)
                                number_with_new_squares[y] = number_with_new_squares[y] + 1
                            # In this case we're missing the left square of the row
                            if (abs(rows[y][-1][0] - right_square[0]) > 25) and number_with_new_squares[y] < 8:
                                new_coord = [
                                    rows[y][-1][0] + x_distances_new[y][-1], rows[y][-1][1]]
                                more_centers.append(new_coord)
                                rows[y].append(new_coord)
                                # Sort the centers
                                rows[y].sort(key=self.take_first)
                                number_with_new_squares[y] = number_with_new_squares[y] + 1
                        if not more_centers:
                            for i in number_with_new_squares:
                                if i < 8:
                                    ind.append(i)
                            for indice in ind:
                                left_dist = abs(rows[y][0][0] - left_square[0])
                                right_dist = abs(
                                    rows[y][-1][0] - right_square[0])
                                if left_dist >= right_dist:
                                    new_coord = [
                                        rows[y][0][0] - x_distances_new[y][0], rows[y][0][1]]
                                    more_centers.append(new_coord)
                                    rows[y].append(new_coord)
                                    # Sort the centers
                                    rows[y].sort(key=self.take_first)
                                    number_with_new_squares[y] = number_with_new_squares[y] + 1
                                else:
                                    new_coord = [
                                        rows[y][-1][0] + x_distances_new[y][-1], rows[y][-1][1]]
                                    more_centers.append(new_coord)
                                    rows[y].append(new_coord)
                                    # Sort the centers
                                    rows[y].sort(key=self.take_first)
                                    number_with_new_squares[y] = number_with_new_squares[y] + 1

                    else:
                        not_yet = False
                    all_new_centers_border.append(more_centers)
                    more_centers = []

                for number in number_with_new_squares:
                    if number < 8:
                        not_yet = True
                    elif number > 8:
                        not_yet = False
                        flag_toomany = True
                    elif number == 8:
                        conta = conta + 1
                if conta < 8 and not flag_toomany:
                    not_yet = True
                    conta = 0
                elif flag_toomany:
                    not_yet = False
                elif conta == 8 and not flag_toomany:
                    not_yet = False

            # Check the number of squares that have been found
            for i in number_with_new_squares:
                if i > 8:
                    pass
                    # rospy.logerr('Too many squares found')
                    # sys.exit('Try segmenting the squares again')

            # Add the new centers  to the totality of the squares and compute the new distances
            x_distances_new = []

            for r in range(0, len(rows)):
                '''
                if all_new_centers_border[r]: #If new centers were found, compute new distances
                        for g in range(0, len(all_new_centers_border[r])):
                                rows[r].append(all_new_centers_border[r][g])
                rows[r].sort(key=self.take_first) #Sort the centers
                '''
                for j in range(1, len(rows[r])):
                    if (rows[r][j][0]-rows[r][j-1][0]) != 0:
                        # List of distances on the x direction
                        x_distance_new.append(rows[r][j][0]-rows[r][j-1][0])
                x_distances_new.append(x_distance_new)
                x_distance_new = []

            # Check that all the found centers are inside the contour of the chessboard
            # for i in range(0, len(rows)):
            #     for point in rows[i]:
            #         result = cv2.pointPolygonTest(
            #             contours[0], (point[0], point[1]), False)
            #         if result == -1 or result == 0:  # If the center is outside of the chessboard border or exactely on it
            #             # rospy.logerr('One of the computed points is outside of the chessboard boundaries!')
            #             # sys.exit('Try segmenting the squares again')
            #             pass

            # Check the stds of the distances. If the stds are > than 1, it means that the identified squares are wrong.
            stds = []
            for i in range(0, len(x_distances_new)):
                stds.append(np.std(x_distances_new[i]))
                if np.std(x_distances_new[i]) > 3:
                    pass
                    # rospy.logerr('One of the computed points is outside of the chessboard boundaries!')
                    # sys.exit('Try segmenting the squares again')
            if self.verbose:
                print('STD:')
                print(stds)

            # Draw the new centers in blue
            for o in range(0, len(all_new_centers_border)):
                for d in range(0, len(all_new_centers_border[o])):
                    squares = cv2.circle(
                        squares, (all_new_centers_border[o][d][0], all_new_centers_border[o][d][1]), 5, (255, 0, 0), -1)

            # Count the squares
            final_squares_number = sum(number_with_new_squares)

            # print(rows)
            # print(final_squares_number)

            if self.verbose:
                print('Centers')
                print(rows)
                print('All new centers border')
                print(all_new_centers_border)
                print('Number of found squares for each row')
                print(number_with_new_squares)
                print('Number of found squares')
                print(final_squares_number)
                print('Distances between rows')
                print(x_distances_new)
                print('Squares')
                print(right_square)
                print(left_square)
        if not self.verbose:
            cv2.imshow('Squares segmented', squares)
            cv2.imwrite(os.path.join(
                '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '9-squaresSegmented.png'), squares)
        return rows, contours, squares, self.approx_square, final_squares_number, self.approx

    def take_first_elem(self, elem):
        return elem[0][0]

    def take_second(self, elem):
        return elem[1]

    def take_first(self, elem):
        return elem[0]

    def hough_lines(self, image, grayscale):
        lines = cv2.HoughLines(image, 1, np.pi/180, 200)
        print(lines)
        lines = sorted(lines, key=self.take_first_elem, reverse=True)
        print(lines)
        lines_number = len(lines)
        rho = lines[0][0][0]
        hough = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        count = 0
        count_drawn_lines = 0
        # Draw the detected lines on the grayscale image
        while count < lines_number:
            old_rho = rho
            rho = lines[count][0][0]
            theta = lines[count][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            print('ciao')
            # print(theta)
            # print(rho)
            # print(y1)
            # print(x2)
            # print(y2)
            color = (0, 0, 255)
            print(rho - old_rho)
            '''
			hough = cv2.line(hough, (x1,y1), (x2,y2), color, 2)
			'''
            if (theta < 0.1 or (theta > 1.2 and theta < 1.6) or (theta > 2.5 and theta < 3)) and (abs(rho - old_rho) > 10 or (rho - old_rho) == 0):
                hough = cv2.line(hough, (x1, y1), (x2, y2), color, 2)
                count_drawn_lines = count_drawn_lines + 1

            count = count + 1
        # and ((rho - old_rho) < -200 or (rho - old_rho) > 200 or (rho - old_rho) == 0)
        # (rho > 100 or rho < -100)  and (rho > 100 or rho < -200)
        cv2.imshow('Straight lines', hough)
        cv2.imwrite(os.path.join(
            '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '10-hough.png'), hough)
        return hough

    def mask(self, image, contours):
        # image is the grayscale image of the chessboard. the mask is applied so that the next operations are done only on the previously segmented chessboard
        mask = np.zeros(image.shape, dtype=np.uint8)
        # mask = cv2.drawContours(mask, contours, 0, (0, 0, 255), 2)
        # Fill the contoured area of the chessboard
        mask = cv2.fillPoly(mask, pts=[contours[0]], color=(255, 255, 255))
        # Mask input image with binary image
        result = cv2.bitwise_and(image, mask)
        # Color background white
        result[mask == 0] = 255  # Optional
        if self.verbose:
            cv2.imshow('Result', result)
            cv2.imwrite(os.path.join(
                '/root/tiago_public_ws/src/tiago_playchess/Images/Segmentazione', '11-result.png'), result)
        return result

    def segmentation_sequence(self, image):
        grayscale = self.grayscale(image)  # Convert the image to grayscale
        # Filter the image with a Gaussian kernel
        gaussian = self.gaussian_filter(grayscale, (5, 5))
        # Binarize the image with an Otsu threshold
        otsu = self.otsu_thresholding(gaussian, 0)
        # Detection of the edges using Canny method
        canny = self.canny_edge(otsu, 100, 200)
        dilated = self.dilation(canny, (3, 3))  # Edge dilation
        __, contours, __, annotated_image_vertices, __, vertices = self.squares_contouring(
            dilated, 60, 300, grayscale, True)  # Returns the vertices of the chessboard.
        # hough = self.hough_lines(dilated, grayscale) #Detection of straight lines on the image using the Hough transform
        # Mask the original grayscale image of the chessboard considering only the segmented chessboard
        masked = self.mask(grayscale, contours)
        # Filter the image with a Gaussian kernel
        gaussian_masked = self.gaussian_filter(masked, (3, 3))
        # Binarize the image with an Otsu threshold
        otsu_masked = self.otsu_thresholding(gaussian_masked, 0)
        # Detection of the edges using Canny method
        canny_masked = self.canny_edge(otsu_masked, 100, 200)
        # Detection of the edges using Canny method
        dilated_masked = self.dilation(canny_masked, (3, 3))
        rows_main, __, annotated_image, __, final_squares_number, __ = self.squares_contouring(
            dilated_masked, 10, 60, masked, False)  # It returns a list of the squares' centers, divided by rows
        # hough_masked = self.hough_lines(dilated_masked, masked) #Detection of straight lines on the image using the Hough transform

        chessboard = []
        rows = ['8', '7', '6', '5', '4', '3', '2', '1']
        cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        x = rows_main
        x_flat = []
        for sublist in x:
            for elem in sublist:
                x_flat.append(elem)
        x = x_flat

        chess_squares = ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',                 'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',                 'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',                 'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
                         'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',                 'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',                 'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',                 'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']

        result = []

        for i in range(0, 8):
            for j in range(0, 8):
                square = chess_squares[i*8 + j] if i * \
                    8+j < len(chess_squares) else [0, 0]
                values = x[i*8 + j] if i*8+j < len(x) else [0, 0]
                result.append([0, 0, 0, 0, values, square])

        

        
        tmp_yaml_path = os.path.realpath(os.path.dirname(__file__))

        with open(os.path.join(tmp_yaml_path, 'corners_names_empty_square_old.yaml'), 'w') as file:
            documents = yaml.dump(result, file)

        return rows_main, annotated_image, annotated_image_vertices, final_squares_number, vertices


def main():
    # rospy.init_node('image_processor')
    # /home/silvia/tiago_public_ws/src/tiago_playchess/Images_chessboard_empty/camera_image1.jpeg
    image = cv2.imread('/home/vaishakh/tiago_public_ws/src/playchess/config/tmp/rgb/segmentation_square_center_identification/after_noon_artifical lighting/rgb_image_for_plane_model2023-03-09 14:02:50.878361.png')
    image_processing = ImageProcessing()
    image_processing.segmentation_sequence(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
