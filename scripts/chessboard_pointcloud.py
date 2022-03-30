#!/usr/bin/env python
#This script defines a subscriber node that reads the info published on 
#/output/segmentation and returns the pointcloud of every segmented object.

#Python libs
import numpy as np
import cv2
import os
import yaml
import pickle
import time
#ROS libs
import rospy
import tf
import message_filters
import moveit_commander

#ROS messages
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
from std_msgs.msg import Bool, Int16
from geometry_msgs.msg import Quaternion, PointStamped, Pose, Point, PoseStamped, WrenchStamped
from sensor_msgs import point_cloud2

#My scripts
from chessboard_image_processing import ImageProcessing
import config as cfg


PLAYCHESS_PKG_DIR = '/home/luca/tiago_public_ws/src/tiago_playchess'
GUI_PKG_DIR       = '/home/luca/tiago_public_ws/src/chess_gui'
imported_chessboard_squares_pickle = PLAYCHESS_PKG_DIR + '/config/simul_config_not_transformed.pickle'
imported_chessboard_vertices_pickle = PLAYCHESS_PKG_DIR + '/config/vertices_not_transformed.pickle'

offset = 0.01 #Offset of height of the cylinders representing pieces in the planning scene (1cm)


class PlanningScene:
    def __init__(self):
        self.img_processing = ImageProcessing()

        #Publishers initialization
        self.cloud_publisher = rospy.Publisher('/segmentation_result/cloud', PointCloud2, queue_size = 10)
        self.annotated_img_pub = rospy.Publisher('/segmentation_result/annotated_img/compressed', CompressedImage, queue_size = 10)
        self.done_publisher = rospy.Publisher('/segmentation_done', Bool, queue_size = 10)
        self.transformation_done_publisher = rospy.Publisher('/transformation_done', Bool, queue_size = 10)
        self.state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)

        #Initialize a subscriber that receives the images from the depth camera
        self.depth_subscriber = rospy.Subscriber("/xtion/depth_registered/points", PointCloud2, self.pointcloud_chessboard)
        self.done_subscribe = rospy.Subscriber("/segmentation_done", Bool, self.transform_coordinates)
        self.transformation_done_subscriber = rospy.Subscriber("transformation_done", Bool, self.shutdownNode)

        #Transform listener
        self.listener = tf.TransformListener()

        #Initialize variables
        self.done = False #Flag to understand if the segmentation of the chessboard has been done correctly

        self.simul_config = rospy.get_param('/tiago_playchess/simul_config')
        self.simul_config_not_transformed = rospy.get_param('/tiago_playchess/simul_config_not_transformed')
        self.chessboard_vertices_file = rospy.get_param('/tiago_playchess/chessboard_vertices')
        self.chessboard_vertices_not_transformed_file = rospy.get_param('/tiago_playchess/chessboard_vertices_not_transformed')
        self.z_coord_chessboard_mean = rospy.get_param('/tiago_playchess/z_coord_chessboard_mean')

        #Define TIAGo's interface
        self.scene = moveit_commander.PlanningSceneInterface() #The interface with the world surrounding the robot
        self.pieces_coordinates = cfg.pieces_coordinates
        with open(PLAYCHESS_PKG_DIR + '/scripts/config/standard_config.yaml') as t_p:
                params = yaml.load(t_p)
        if params['color'] == 'white':
            self.squares_to_index = cfg.squares_to_index_white
        else:
            self.squares_to_index = cfg.squares_to_index_black

        #Import pieces characteristics
        self.pawn = cfg.pawn
        self.rook = cfg.rook
        self.bishop = cfg.bishop
        self.king = cfg.king
        self.queen = cfg.queen
        self.knight = cfg.knight

    def pointcloud_chessboard(self, pointcloud2):
        #Interrupt function that is called whenever a new message from the depth camera arrives.
        if not self.done:
            self.centroids = [] #Centroids will contain arrays of the 3 coordinates in the space of the chessboard centers
            self.vertices = []

            #Wait for a message from the RGB camera
            rgb_image = rospy.wait_for_message('/xtion/rgb/image_rect_color/compressed', CompressedImage, timeout = 30)

            #Conversion to cv2 image
            np_arr = np.fromstring(rgb_image.data, np.uint8)
            img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            #Segment the image
            self.rows, annotated_img_np, annotated_img_np_vertices, squares_number, self.chessboard_vertices = self.img_processing.segmentation_sequence(img_np)
            if len(self.chessboard_vertices) < 4:
                rospy.logwarn('Not all 4 vertices have been identified. Try segmenting again')
            #t = time.time()
            #If the squares have been segmented correctly (hopefully)
            if squares_number == 64:
                #Publish the annotated image
                annotated_img_msg = CompressedImage()
                #Header and format
                annotated_img_msg.header.stamp = rospy.Time.now()
                annotated_img_msg.format = "jpeg"
                #Actual image
                immagine = np.array(cv2.imencode('.jpg', annotated_img_np)[1]).tostring()
                annotated_img_msg.data = immagine
                self.annotated_img_pub.publish(annotated_img_msg)
                #print("Publishing the annotated image took " + str(time.time() - t))   # it takes up to 0.01s
                                                                                        # think about making this part optional, even at runtime through a param
                
                cv2.imwrite(os.path.join(GUI_PKG_DIR + '/images', 'segmentation_example.png'), annotated_img_np)
                rospy.loginfo('Segmentation result image saved in the chess_gui/images folder')
                '''
                #Publish the annotated image of the veritces
                annotated_img_msg_vertices = CompressedImage()
                #Header and format
                annotated_img_msg_vertices.header.stamp = rospy.Time.now()
                annotated_img_msg_vertices.format = "jpeg"
                #Actual image
                annotated_img_msg_vertices.data = np.array(cv2.imencode('.jpg', annotated_img_np_vertices)[1]).tostring()
                cv2.imwrite(os.path.join(GUI_PKG_DIR + '/images', 'vertices.png'), annotated_img_np_vertices)
                rospy.loginfo('Vertices image saved in the chess_gui/images folder')
                '''

                #Get an iterable object from the pointcloud and use it to fill a numpy array of points (faster than point_cloud2.read_points_list)
                depth_points_iterable = point_cloud2.read_points(pointcloud2, skip_nans = False, field_names = ("x", "y", "z"))
                self.depth_points_array = np.reshape(np.array([point for point in depth_points_iterable]), newshape = (pointcloud2.height, pointcloud2.width, 3))

                count = 0
                for row in range(0, len(self.rows)):
                    for square in self.rows[row]:
                        #print(depth_points_array[square[1], square[0], :])
                        self.centroids.append(self.depth_points_array[square[1], square[0], :])
                        count += 1

                for vertices in self.chessboard_vertices:
                    self.vertices.append(self.depth_points_array[vertices[0][1], vertices[0][0], :])

                if count == 64:
                    self.done = True
                    rospy.loginfo('Chessboard segmentation ended successfully')
                    self.done_publisher.publish(True)

            else:
                rospy.logwarn('The squares have not been correctly identified. Try segmenting the chessboard again')


    def transform_coordinates(self, data):
        #Function to transform the coordinates of the chesboard squares fron the vision reference frame to the base_footprint reference frame.
        if data:
            #Empty the lists of the points corresponding to the centers of the squares in the optical frame and in the base_footprint frame.
            self.optical_points = []
            self.base_points = []
            self.optical_points_vertices = []
            self.base_points_vertices = []
            
            #Transform the coordinates of the centers of the squares in PointStamped messages
            for coord in self.centroids:
                point = PointStamped()
                point.header.frame_id = 'xtion_rgb_optical_frame'
                point.header.stamp = rospy.Time.now()
                point.point = Point(coord[0], coord[1], coord[2])
                self.optical_points.append(point) #List containing all the PointStamped messages

            for vertices in self.vertices:
                point = PointStamped()
                point.header.frame_id = 'xtion_rgb_optical_frame'
                point.header.stamp = rospy.Time.now()
                point.point = Point(vertices[0], vertices[1], vertices[2])
                self.optical_points_vertices.append(point) #List containing all the PointStamped messages

            #Transform the coordinates in PointStamped messages in the base_footprint frame
            for opt_point in self.optical_points:
                now = rospy.Time.now()
                self.listener.waitForTransform('/xtion_rgb_optical_frame', '/base_footprint', now, rospy.Duration(4))
                new_point = self.listener.transformPoint('/base_footprint', opt_point)
                self.base_points.append(new_point)

            for opt_vert in self.optical_points_vertices:
                now = rospy.Time.now()
                self.listener.waitForTransform('/xtion_rgb_optical_frame', '/base_footprint', now, rospy.Duration(4))
                new_point = self.listener.transformPoint('/base_footprint', opt_vert)
                self.base_points_vertices.append(new_point)  

            sum = 0
            for coord in self.base_points:
                z_coord = coord.point.z 
                sum += z_coord
            self.z_coord_chessboard_mean_number = sum/len(self.base_points) #Save the mean z coordinate of the chessboard squares.
            with open(self.z_coord_chessboard_mean, "w") as t_coord:
                yaml.dump(self.z_coord_chessboard_mean_number, t_coord) #Save the mean z coordinate of the chessboard in the yaml file

            #Save the transformed centroids in a yaml file
            with open(self.simul_config, "w") as t_p:
                    yaml.dump(self.base_points, t_p)

            #Save the transformed vertices in a yaml file
            with open(self.chessboard_vertices_file, "w") as t_p:
                    yaml.dump(self.base_points_vertices, t_p)

            #Save the centroids in the eyes reference frame in a yaml file
            with open(self.simul_config_not_transformed, "w") as t_p:
                    yaml.dump(self.optical_points, t_p)

            with open(imported_chessboard_squares_pickle, "wb") as fout:
                    pickle.dump(self.optical_points, fout)

            #Save the vertices in the eyes reference frame in a yaml file
            with open(self.chessboard_vertices_not_transformed_file, "w") as t_p:
                    yaml.dump(self.optical_points_vertices, t_p)

            with open(imported_chessboard_vertices_pickle, "wb") as fout:
                    pickle.dump(self.optical_points_vertices, fout)

            self.transformation_done_publisher.publish(True)

            #return self.base_points #The function returns a list of the coordinates (PointStamped type) of the squares' centers in the base_footprint reference frame.

    def populate_pieces_starting_position(self, base_centers):
        #Function to populate the planning scene with the pieces in their starting position. It receives as input the coordinates of the squares' centeres in the base_footprint reference frame.
        #In case there is something in the world form previous executions (maybe the robot was closer or further from the chessboard)
        for name in self.pieces_coordinates:
            self.scene.remove_world_object(name)

        #Add cylinders corresponding to the pieces to the planning scene.
        for name in self.pieces_coordinates:
            pose = PoseStamped()
            pose.header.frame_id = 'base_footprint'
            pose.header.stamp = rospy.Time.now()
            pose.pose = Pose(Point(base_centers[self.squares_to_index[self.pieces_coordinates[name][0]]].point.x, base_centers[self.squares_to_index[self.pieces_coordinates[name][0]]].point.y, base_centers[self.squares_to_index[self.pieces_coordinates[name][0]]].point.z + self.pieces_coordinates[name][1]['height']/2 + offset), Quaternion(0, 0, 0, 1))
            self.scene.add_cylinder(name, pose, height = self.pieces_coordinates[name][1]['height'], radius = self.pieces_coordinates[name][1]['diameter']/2)

    def populate_pieces(self, base_centers, live_chessboard_situation):
        #Function to populate the planning scene with the pieces in their current position.
        #It receives as input the coordinates of the squares' centeres in the base_footprint reference frame and live_chessboard_situation that keeps track of the pieces positions thoughout the game.
        #This function needs to be used whenever TIAGo sees that a piece has been moved from the opponent.

        #In case there is something in the world form previous executions (maybe the robot was closer or further from the chessboard)
        for name in self.pieces_coordinates:
            self.scene.remove_world_object(name)

        key_list = list(live_chessboard_situation.keys())
        val_list = list(live_chessboard_situation.values())

        #Add cylinders corresponding to the pieces to the planning scene.
        for name in self.pieces_coordinates:
            position = val_list.index(name) #Get the position (index) of the current piece in the chessboard
            square = key_list[position] #Get the square where the current piece is.
            pose = PoseStamped()
            pose.header.frame_id = 'base_footprint'
            pose.header.stamp = rospy.Time.now()
            pose.pose = Pose(Point(base_centers[self.squares_to_index[square]].point.x, base_centers[self.squares_to_index[square]].point.y, base_centers[self.squares_to_index[square]].point.z + self.pieces_coordinates[name][1]['height']/2 + offset), Quaternion(0, 0, 0, 1))
            self.scene.add_cylinder(name, pose, height = self.pieces_coordinates[name][1]['height'], radius = self.pieces_coordinates[name][1]['diameter']/2)

    def populate_table(self, z_mean):
        #Function to populate the planning scene with a box representing the table.
        #It receives as input the mean z coordinate of the squares centers in the base_footprint reference frame.
        self.scene.remove_world_object('table')

        #Add the box.
        pose = PoseStamped()
        pose.header.frame_id = 'base_footprint'
        pose.header.stamp = rospy.Time.now()
        pose.pose = Pose(Point(0.5 + 0.30, 0, z_mean / 2), Quaternion(0, 0, 0, 1)) ####DA VERIFICARE IL + 0.30
        self.scene.add_box('table', pose, size = (1, 1, z_mean))

    def shutdownNode(self, data):
        if data:
            self.populate_pieces_starting_position(self.base_points) #Add the pieces to the planning scene for collision management.
            self.populate_table(self.z_coord_chessboard_mean_number)
            time.sleep(2)

            self.state_publisher.publish(30) #Pass to the state of enabling the GUI pushbutton to confirm segmentation.
        
		
if __name__ == '__main__':
    rospy.init_node('chessboard_segmenter', anonymous = True)
    rospy.loginfo('Chessboard segmentation in execution...')
    segmenter = PlanningScene()
    try:
        rospy.spin()    
    except KeyboardInterrupt:
        print("Shutting down the chessboard segmenter module.")
