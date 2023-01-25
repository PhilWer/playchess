#!/usr/bin/env python3
'''
class for homographic transformation of perspective chess board image
'''
import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np


class HOMO_TRANSFOR():
    def __init__(self, im_src, chess_board_edges, debug=False):
        self.im_src = im_src
        self.im_dst = np.zeros((480, 480), 'uint8') * 125
        self.debug = debug
        self.chess_board_edges = chess_board_edges

    def transform(self):
        # rearanging the chess board corners in order i.e from top left CW order
        # only when accepting corner values from imageprocessing.py
        chess_corner = self.chess_board_edges
        # for i in range(4):
        #     for j in range(2):
        #         chess_corner.append(self.chess_board_edges[i,0,j])
        # Four corners of the book in source image
        pts_src = np.array([[chess_corner[0][0], chess_corner[0][1]], [chess_corner[1][0], chess_corner[1][1]], [
                           chess_corner[2][0], chess_corner[2][1]], [chess_corner[3][0], chess_corner[3][1]]])

        # Four corners of the chess_board in destination image.
        pts_dst = np.array([[0, 0], [480, 0], [480, 480], [0, 480]])

        # Calculate Homography
        h, status = cv2.findHomography(pts_src, pts_dst)

        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(
            self.im_src, h, (self.im_dst.shape[1], self.im_dst.shape[0]))

        # Display images
        if self.debug:
            print('chess board corners in cw order from top left', chess_corner)
            # cv2.imshow("Source Image", self.im_src)
            # for i in range(64):
            # cv2.rectangle(im_out, (360, 360), (420, 420), (0, 255, 0), 3)
            cv2.imshow("Warped Source Image", im_out)

            cv2.waitKey(0)

        return im_out
