#!/user/bin/env python2

import cv2
import pickle
import os
from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import compare_mse
'''change it for p2.7 or older version of skimages  '''
# user defined module
from useful_functions.homographic_transformation import HOMO_TRANSFOR as ht
import numpy as np


PLAYCHESS_PKG_DIR = '/home/vaishakh/tiago_public_ws/src/playchess'


class MoveDetection:
    """_summary_: user definded class to determine difference between image befor opponent move and after opponent move 
        a detect_square_change function used
    """

    def __init__(self):
        self.debug = True

    def detect_square_change(self, previous, current,tiago_color, debug=False):
        """_summary_ compare two images based on Structural Similarity Index (SSIM)

        Args:
            previous (image): camera image before opponent move
            current (image): camera image after oppponent move
            tiago_colour (str): "black" if TIAGo plays as black. Defaults to "white".
            debug (bool, optional): True for debuging. Defaults to False.

        Returns:
            none
            it stores an pickle file with a list contaning detected squares with name squares_changed
        """
        # Convert the images to grayscale
        grayA = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

        # Computes the Structural Similarity Index (SSIM) between previous and current
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        # Threshold the difference image
        thresh = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        img_copy = img.copy()

        # conforming naming of chess board based on starting position of white pieces
        if tiago_color == 'white':
            a1_pos = 1
        else:
            a1_pos = 0
        # Initialize list to store square information
        squares = []
        # Draw squares on image
        for i in range(8):
            for j in range(8):
                # Determine square name
                if a1_pos:
                    square_name = chr(ord('a') + i) + str(8-j)
                else:
                    square_name = chr(ord('a') + 7-i) + str(j+1)
                # Coordinates of square corners
                x1, y1 = i*60, j*60
                x2, y2 = (i+1)*60, (j+1)*60
                # Coordinates of square center
                x_center, y_center = (x1+x2)//2, (y1+y2)//2
                radius = 30
                # # Get sub-image of current square , can be used if circular masking fails
                square = thresh[y1:y2, x1:x2]
                # get window size to get total pixel in one square cell
                height1, width1 = square.shape[:2]
                # Get image dimensions
                height, width = img.shape[:2]
                # Create blank image for mask
                mask = np.zeros((height, width), np.uint8)
                # Draw circle on mask
                cv2.circle(mask, (x_center, y_center),
                           radius, (255, 255, 255), -1)
                # Apply mask to image
                masked_img = cv2.bitwise_and(img, img, mask=mask)
                # converting to single channel
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("Image", square)
                # cv2.waitKey(0)
                # Count non-zero pixels in masked region
                non_zero_pixels = cv2.countNonZero(masked_img)
                non_zero_pixels = cv2.countNonZero(square)
                # Calculate percentage of non-zero pixels
                percent_non_zero = (non_zero_pixels * 100 / (height1 * width1))
                # Check if square contains white pixels
                if percent_non_zero > 8:
                    # Draw square and write square name
                    if self.debug:
                        cv2.rectangle(img_copy, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_copy, square_name, (x1+20, y1+40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Append square information to list
                    squares.append({"name": square_name, "center": (
                        x_center, y_center), "corners": ((x1, y1), (x2, y2))})

        # Save above list to pickle file which contains details of cell having changs
        # with open(os.path.join(PLAYCHESS_PKG_DIR + '/scripts/vaishakh_scripts/image_processing/pickle', 'squares_changed.pickle'), "wb") as f:
        #     pickle.dump(squares, f)

        # DEBUG
        if debug:
            print("detected squares having changes", squares)
            # cv2.imshow("image after SSIM", img)
            cv2.imshow(
                "difference b/w previous and current state of board", img_copy)
            cv2.waitKey(0)

        return squares

    def possible_squares_start_end(self, chessboardedges_with_out_borders, state1_image, state2_image, tiago_color,debug=True):
        """_summary_

        Args:
            chessboardedges_with_out_borders (list):A list of four corners obtained from a YAML file created through the processing of a chessboard_image_processing.py  using the CV2 library.
            state1_image (image):An image acquired from the Tiago camera following the completion of Tiago's rotation
            state2_image (image): An image acquired from Tiago camera following the completion of opponent player's rotation.
            tiago_color (string): whats the  color of the chess piece for Tiago : "white" or "black."
        """
        m_d=MoveDetection()
        transformed_chess_board = ht(
            state1_image, chessboardedges_with_out_borders)
        state1_image = transformed_chess_board.transform()

        transformed_chess_board = ht(
            state2_image, chessboardedges_with_out_borders)
        state2_image = transformed_chess_board.transform()
        thresh = m_d.detect_square_change(state2_image, state1_image, tiago_color,debug)
        
        return thresh


if __name__ == '__main__':

    m_d = MoveDetection()

    with open(r'/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/pickle/store_chess_board_edges.pickle', 'rb') as file:
        chessBoardEdgesS_with_out_borders = pickle.load(file)
        # print(chessBoardEdgesS_with_out_borders)
    # for i in range(25):
    #     img_path = '/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/Static_images/Image_move_detection/test{}.png'.format(i)
    #     img_previous = cv2.imread(img_path)

    #     img_path = '/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/Static_images/Image_move_detection/test{}.png'.format(i+1)
    #     img_current = cv2.imread(img_path)

        img_previous = cv2.imread(
            '/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/Static_images/Image_move_detection/test22.png')
        # print(chessBoardEdgesS_with_out_borders)

        img_current = cv2.imread(
            '/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/Static_images/Image_move_detection/test23.png')

        T = m_d.possible_squares_start_end(chessBoardEdgesS_with_out_borders, img_previous, img_current, tiago_color='white')
        print(T)
