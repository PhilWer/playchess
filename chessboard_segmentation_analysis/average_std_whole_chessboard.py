#!/user/bin/env python2
import sys
import os
# root=os.path.normpath(os.path.join(os.path.dirname(__file__),'..'))
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),'..','scripts')))
import cv2
import yaml
from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import compare_mse
'''change it for p2.7 or older version of skimages  '''
# user defined module
import numpy as np

# adding path to useful functions for imports
sys.path.append(os.path.join(os.path.dirname(__file__),'..','scripts'))

from chessboard_image_processing_old import ImageProcessing as IMOld
from chessboard_image_processing import ImageProcessing as IMNew


PLAYCHESS_PKG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__),'..'))


class AvgStsAccu:
    """_summary_: user definded class to determine difference between image befor opponent move and after opponent move 
        a detect_square_change function used
    """

    def __init__(self, im_src, chess_board_edges, debug=False):
        self.im_src = im_src
        self.im_dst = np.zeros((480, 480), 'uint8') * 125
        self.debug = debug
        self.chess_board_edges = chess_board_edges

    def homo_transform(self):
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
        #  if self.debug:
        #      print('chess board corners in cw order from top left', chess_corner)
        #      # cv2.imshow("Source Image", self.im_src)
        #      # for i in range(64):
        #      # cv2.rectangle(im_out, (360, 360), (420, 420), (0, 255, 0), 3)
        #      cv2.imshow("Warped Source Image", im_out)  
        #      cv2.waitKey(0) 
         return im_out,h
    
    def make_square_save_center_val(self,img,tiago_color):
        img_copy = img.copy()
        img_copy_1 =img.copy()
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
                if True:
                    # Draw square and write square name
        
                    cv2.rectangle(img_copy, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_copy, '{},{}'.format(x_center,y_center), (x_center,y_center),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255),1)
                    cv2.putText(img_copy, square_name, (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    cv2.circle(img_copy,(x_center,y_center),2,(255,0,0),-1)
                    cv2.circle(img_copy_1,(x_center,y_center),5,(0,255,0),-1)
                    # Append square information to list
                    squares.append({"name": square_name, "center": (
                        x_center, y_center), "corners": ((x1, y1), (x2, y2))})

        # Save above list to pickle file which contains details of cell having changs
        # with open(os.path.join(PLAYCHESS_PKG_DIR + '/scripts/vaishakh_scripts/image_processing/pickle', 'squares_changed.pickle'), "wb") as f:
        #     pickle.dump(squares, f)

        # DEBUG
        if self.debug:
            #print("detected squares having changes", squares)
            # cv2.imshow("image after SSIM", img)
            cv2.imshow('Image wit squares',img_copy)
            

        return squares, img_copy_1
    
    def accuracy(self,t_mat,chessboard_corners,tf_square,threshold,img,pipeline):
        #errors
        errors = []
        # identified center  point
        for i in chessboard_corners:
            chessboard_corners = i
            # identified center  point
            point = np.array([chessboard_corners[4]], dtype=np.float32)
            name = chessboard_corners[5]
            # Transform the point
            transformed_point = cv2.perspectiveTransform(point.reshape(1, 1, 2), t_mat)

            # Define the expected value
            for i in tf_square:
                if i['name'] == name:
                    x,y = i['center'][0],i['center'][1]
                    expected_center_point = np.array([x, y])
            cv2.circle(img,(x,y),2,(0,0,255),-1)
            #print('tf and expected',transformed_point,expected_center_point)

            # Calculate the % error
            error = (np.linalg.norm(expected_center_point - transformed_point) /
                     np.linalg.norm(expected_center_point)) * 100
            
            
            
            # add errors to the listTrue
            errors.append(error)
        

        # Calculate the average and standard deviation of the errors
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        # Calculate the accuracy using a threshold value of 5% error
        accuracy = np.mean(np.array(errors) < threshold) * 100
        # Calculate the average and standard deviation of errors
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        # Print the results
        print(pipeline)
        print("Average error:", avg_error)
        print("Standard deviation of error:", std_error)
        print("Accuracy (threshold={}%): {:.2f}%".format(threshold, accuracy))

        # if self.debug:
        #     cv2.imshow('green:expected  red:transformed',img)
        #     cv2.waitKey(0)

        return accuracy
        


if __name__ == '__main__':
   
   #initialize befor running
    threshold = 2
    '''
    The threshold value is set to 5% and is used to calculate the accuracy. 
    The accuracy is calculated as the percentage of errors that are less 
    than the threshold value.
    '''
    new = True# if u run chessboard image processing from scripts and false if u run sivia's code from chessboard_segment analysis
    
    debug = True
    tiago_color= 'white'
    img = '/home/vaishakh/tiago_public_ws/src/playchess/config/tmp/rgb/segmentation_square_center_identification/after_noon_artifical lighting/rgb_image_for_plane_model2023-03-09 14:02:50.878361.png'
    img = cv2.imread(img)

    #getting data playchess
    # old pipe line
    ip = IMOld()
    ip.segmentation_sequence(img)

    # new pipe lineOld
    ip = IMNew()
    ip.segmentation_sequence(img)

    #load yaml fime save during image processing

    with open ( os.path.join(PLAYCHESS_PKG_DIR,'chessboard_segmentation_analysis/store_chess_board_edges.yaml'),'r') as file:
     chessboard_edges = yaml.load(file.read(), Loader=yaml.Loader)  

    
    with open ( os.path.join(PLAYCHESS_PKG_DIR,'chessboard_segmentation_analysis/corners_names_empty_square.yaml'),'r') as file:
        chess_square_name_center_new = yaml.load(file.read(), yaml.Loader)  

    
    with open ( os.path.join(PLAYCHESS_PKG_DIR,'chessboard_segmentation_analysis/corners_names_empty_square_old.yaml'),'r') as file:
     chess_square_name_center_old = yaml.load(file.read(), yaml.Loader) 
     

# # new pipeline accuracy
    accu_new = AvgStsAccu(img,chessboard_edges,debug)
    tf_img,t_mat = accu_new.homo_transform()
    tf_square,img = accu_new.make_square_save_center_val(tf_img,tiago_color)
    accuracy=accu_new.accuracy(t_mat,chess_square_name_center_new,tf_square,threshold,img,pipeline='new') 
    

    #old pipeline accuracy
    accu = AvgStsAccu(img,chessboard_edges,debug)
    tf_img,t_mat = accu.homo_transform()
    tf_square,img = accu.make_square_save_center_val(tf_img,tiago_color)
    accuracy=accu.accuracy(t_mat,chess_square_name_center_old,tf_square,threshold,img,pipeline='old') 

    
