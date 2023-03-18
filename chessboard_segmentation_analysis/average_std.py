
import os
import cv2
import sys
import yaml
import numpy as np


class BirdEyeView():
    def __init__(self, debug):
        self.debug = debug

    def bird_eye_view(self, img, chessboard_corners):
        im_dst = np.zeros((100, 100), 'uint8') * 125 # (61, 61)
        # for i in range(4):
        #     for j in range(2):
        #         chess_corner.append(self.chess_board_edges[i,0,j])
        # Four corners of the book in source image
        pts_src = np.array([chessboard_corners[0], chessboard_corners[1],
                           chessboard_corners[2], chessboard_corners[3]])

        # Four corners of the chess_board in destination image.
        pts_dst = np.array([[0, 0], [60, 0], [60, 60], [0, 60]])
        # Calculate Homography
        h, status = cv2.findHomography(pts_src, pts_dst)

        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(
            img, h, (im_dst.shape[1], im_dst.shape[0]))

        # identified center  point
        point = np.array([chessboard_corners[4]], dtype=np.float32)

        # Transform the point
        transformed_point = cv2.perspectiveTransform(point.reshape(1, 1, 2), h)

        # Define the expected value
        expected_center_point = np.array([30, 30])

        # Calculate the % error
        error = (np.linalg.norm(expected_center_point - transformed_point) /
                 np.linalg.norm(expected_center_point)) * 100

        # Display images
        if self.debug:
            print("Error:", error)
            print('Original Center Point:', point)
            print('Transformed Center Point:', transformed_point)
            print('chess board corners in cw order from top left',
                  chessboard_corners)
            # cv2.imshow("Source Image", self.im_src)
            # for i in range(64):
            # cv2.rectangle(im_out, (360, 360), (420, 420), (0, 255, 0), 3)
            cv2.imshow("Warped Source Image", im_out)

            cv2.waitKey(0)

        return h, im_out, error


if __name__ == '__main__':

    # read the input image
    img = cv2.imread(
        '/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/Static_images/empty_chess_board.png')

   # read the chessboard square  corners center, name and

    with open(r'/home/vaishakh/tiago_public_ws/src/playchess/config/tmp/yaml/corners_names_empty_square.yaml', 'r') as file:
        chessBoardEdgesS_with_out_borders = yaml.load(file, Loader=yaml.Loader)
    # Initialize an empty list to store the errors
    errors = []
    # iterate through all square centers identified
    for i in chessBoardEdgesS_with_out_borders:
        chess_square_corners = i
        tf = BirdEyeView(debug=0)
        transform_matrix, bird_eye_view, error = tf.bird_eye_view(
            img, chess_square_corners)

        # add errors to the listTrue
        errors.append(error)

    # Calculate the average and standard deviation of the errors
        avg_error = np.mean(errors)
        std_error = np.std(errors)

    # Calculate the accuracy using a threshold value of 5% error
    threshold = 2.0
    accuracy = np.mean(np.array(errors) < threshold) * 100

    # Calculate the average and standard deviation of errors
    avg_error = np.mean(errors)
    std_error = np.std(errors)

    # Print the results
    print("Average error:", avg_error)
    print("Standard deviation of error:", std_error)
    print("Accuracy (threshold={}%): {:.2f}%".format(threshold, accuracy))
    '''
     The threshold value is set to 5% and is used to calculate the accuracy. 
     The accuracy is calculated as the percentage of errors that are less 
     than the threshold value.
     '''
