import os
import numpy as np
import open3d as o3d
import pickle
import cv2
import matplotlib.path as mpath
import copy
import csv
import yaml


class DepthImageProcessing:
    def __init__(self):
        self.debug = False

    def depth_to_point_cloud(self, depth_array_from_csv, rgb_image, filter_points):
        self.depth_image = depth_array_from_csv

        if self.debug:
            cv2.imshow('depth_image', self.depth_image)
            print('depth image min value:{} /n max value:{}'.format(
                np.min(self.depth_image), np.max(self.depth_image)))
            cv2.waitKey(100)

        # depth camera parameters:
        self.FX_DEPTH = 523.2994491601762/1
        self.FY_DEPTH = 524.1979376240457/1
        self.CX_DEPTH = 312.0471722095908/1
        self.CY_DEPTH = 249.9013550067579/1

        # Create a path object from the points
        path = mpath.Path(filter_points)

        # Create arrays to store the row and column indices of the pixels inside the quadrilateral
        ii = []
        jj = []

        # Loop over all pixels in the depth image
        height, width = self.depth_image.shape
        for i in range(height):
            for j in range(width):
                # Check if the pixel is inside the quadrilateral
                if path.contains_point((j, i)):
                    # Store the row and column indices of the pixel
                    ii.append(i)
                    jj.append(j)
                # ii.append(i)
                # jj.append(j)

        # Convert the row and column indices to numpy arrays
        ii = np.array(ii)
        jj = np.array(jj)

        # Extract the depth values for the pixels inside the quadrilateral
        z = self.depth_image[ii, jj]
        # Compute constants:
        xx = (jj - self.CX_DEPTH) / self.FX_DEPTH
        yy = (ii - self.CY_DEPTH) / self.FY_DEPTH
        pcd = np.dstack((xx * z, yy * z, z)).reshape((-1, 3))

        ###################################################################
        # RGB camera parameters:
        self.FX_RGB = 521.8336396330244/1
        self.FY_RGB = 523.0267908518745/1
        self.CX_RGB = 316.5304178834524/1
        self.CY_RGB = 250.880673751086/1

        # Convert the RGB image to the correct format
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Get the color values for the pixels inside the quadrilateral
        color = rgb_image[ii, jj, :]

        # Reshape the color values to have the same number of rows as the point cloud
        color = color.reshape((-1, 3))

        pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        # Add the color values to the point cloud object
        pcd_o3d.colors = o3d.utility.Vector3dVector(color / 255.0)

        if  self.debug:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=50)
            o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])
        return pcd_o3d

    def voxel_down_and_outlier_remove(self, pcd_o3d):
        voxel_down_pcd = pcd_o3d.voxel_down_sample(voxel_size=0.02)
        cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=2.0)
        pcd_o3d = cl

        if self.debug:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=50)
            o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])

        return pcd_o3d

    def transforming_optical_to_base_frame(self, pcd_o3d, quat=[-0.673, 0.679, -0.202, 0.214], trans=[0.232, 0.016, 1.318]):
        quat[0], quat[-1] = quat[-1], quat[0]
        quat[1], quat[-1] = quat[-1], quat[1]
        # where qw, qx, qy, qz are the values of the quaternion in ros order is qx,qy,qz , hence the swap
        quat[2], quat[-1] = quat[-1], quat[2]
       # where tx, ty, tz are the values of the translation
        trans = [i*1000 for i in trans]
        # print('quat:',quat)
        # print('trans:',trans)
        # # Convert the quaternion to a rotation matrix
        '''to base frame use base to optic transformation on points value in optic frame'''
        rot_mat = o3d.geometry.get_rotation_matrix_from_quaternion(quat)

        # Combine the rotation and translation into a 4x4 transformation matrix
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = trans

        # Transform the point cloud
        pcd_o3d.transform(trans_mat)
        self.inv_trans_mat = np.linalg.inv(trans_mat)

        if  self.debug:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=50)
            o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])

        return pcd_o3d
        ##################################################

    def plane_model(self):
        # rgb_image path
        root = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..', '..'))
        # Load image from csv file
        depth_array = np.genfromtxt(os.path.join(root, 'config', 'tmp', 'depth',
                                    'depth_array_for_plane_model.csv'),
                                    delimiter=","
                                    )
        # Load image from png
        rgb_img = cv2.imread(os.path.join(root, 'config', 'tmp', 'rgb',
                              'rgb_image_for_plane_model.png'))
        # Uncomment to run the script alone
        #rgb_img = cv2.imread('/home/vaishakh/tiago_public_ws/src/playchess__/playchess/scripts/vaishakh_scripts/image_processing/move_detection/csv/Special_moves_test_opponent_black/empty_boARD/empty_rgb_image.png')
        #depth_array = np.genfromtxt('/home/vaishakh/tiago_public_ws/src/playchess__/playchess/scripts/vaishakh_scripts/image_processing/move_detection/csv/Special_moves_test_opponent_black/empty_boARD/empty_depth_image.csv', delimiter=',')
        with open(os.path.join(root, 'config', 'tmp', 'yaml', 'store_chess_board_edges.yaml'), 'r') as file:
            filter_points = yaml.load(file, Loader=yaml.Loader)
            print(filter_points)
        pcl = self.depth_to_point_cloud(depth_array, rgb_img, filter_points)
        down_sampled_pcl = self.voxel_down_and_outlier_remove(pcl)
        transformed_pcl = self.transforming_optical_to_base_frame(
            down_sampled_pcl, quat=[-0.673, 0.679, -0.202, 0.214], trans=[0.232, 0.016, 1.318])
        # creating the plane using RANSAC
        distance_threshold = 0.1
        plane_model, inliers = transformed_pcl.segment_plane(distance_threshold=distance_threshold,
                                                             ransac_n=3,
                                                             num_iterations=1000)

        return plane_model

    def neglect_pcd_above_and_below_certain_height(self, pcd_o3d):

        # Get the plane equation coefficients
        a, b, c, d = self.plane_model()
        print(a, b, c, d)
        # assigning point to be filtered
        points_for_filtering = np.asarray(pcd_o3d.points)
        colors = np.asarray(pcd_o3d.colors)

        # Calculate the distance of each point from the plane
        distances = a * points_for_filtering[:, 0] + b * \
            points_for_filtering[:, 1] + c * points_for_filtering[:, 2] + d

        # Remove points above 40 and below -10
        points = points_for_filtering[(distances >= 5) & (distances <= 50)]
        pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(points)

        ######################################################
        if len(np.asarray(pcd_o3d.points)) < 1000:
            # Assign colors to the filtered points
            filtered_colors = np.zeros((len(points), 3), dtype=np.float32)
            for i, point in enumerate(points):
                index = np.where(
                    (points_for_filtering == point).all(axis=1))[0][0]
                filtered_colors[i] = colors[index]
            pcd_o3d.colors = o3d.utility.Vector3dVector(filtered_colors)
            mean_rgb_value = np.mean(filtered_colors, axis=0)
            occupancy = False
            if len(np.asarray(pcd_o3d.points)) >= 10:
                occupancy = True
        else:
            mean_rgb_value = 0
        aabb = pcd_o3d.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        vertices = (np.asarray(aabb.get_box_points()))
        pcl_count = len(np.asarray(pcd_o3d.points))
        # print(vertices[3, 0])
        # Visualize:
        if not self.debug:

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=10, origin=[vertices[0, 0],  vertices[0, 1], vertices[0, 2]])
            o3d.visualization.draw_geometries([pcd_o3d, aabb, mesh_frame])
        return pcl_count, mean_rgb_value, occupancy

    def depth_pcl_counting(self, depth_image, rgb_image, filter_points, quat=[-0.673, 0.679, -0.202, 0.214], trans=[0.232, 0.016, 1.318],debug=False):
        pcl = self.depth_to_point_cloud(depth_image, rgb_image, filter_points)
        down_sampled_pcl = self.voxel_down_and_outlier_remove(pcl)
        transformed_pcl = self.transforming_optical_to_base_frame(
            down_sampled_pcl, quat=[-0.673, 0.679, -0.202, 0.214], trans=[0.232, 0.016, 1.318])
        pcl_count, mean_rgb_value, occupancy = self.neglect_pcd_above_and_below_certain_height(
            transformed_pcl)

        return pcl_count, mean_rgb_value, occupancy


if __name__ == '__main__':
    dip = DepthImageProcessing()
    root = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..','..'))
    tmp_yaml_path = os.path.join(root, 'config', 'tmp', 'yaml')
    with open(os.path.join(tmp_yaml_path,'store_chess_board_edges.yaml' ),'r') as file:
        chessboard_corners = yaml.load(file, Loader=yaml.Loader)
    print(chessboard_corners)
    with open(os.path.join(tmp_yaml_path,'corners_names_empty_square.yaml' ),'r') as file:
        chessboard_square_corners = yaml.load(file, Loader=yaml.Loader)
        for i in chessboard_square_corners:
            if i[5] == 'c4':
                print(i[5])
                a = i
                filter_points = np.array([a[0], a[1], a[2], a[3]])
                # for full chess_board
                # filter_points = np.array(
                #     [chessboard_corners[0], chessboard_corners[1], chessboard_corners[2], chessboard_corners[3]])
                depth_image = np.genfromtxt(
                    '/home/pal/tiago_public_ws/src/playchess/data/moves/move_2/depth_after.csv', delimiter=',')
                rgb_image = cv2.imread(
                    '/home/pal/tiago_public_ws/src/playchess/data/moves/move_2/rgb_after.png')
                pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                    depth_image, rgb_image, filter_points, quat=[-0.673, 0.679, -0.202, 0.214], trans=[0.232, 0.016, 1.318])
                print('mean_rgb:', mean_rgb_value)
                print('plc_count:',pcl_count)
                print('occupancy:', occupancy)

    # filter_points = np.array([(182, 126), (463, 116), (547, 387), (143, 408)]) #for full chess_board

    # depth_image = np.genfromtxt('/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/move_detection/csv/empty_chess_board_new/empty_depth_image.csv', delimiter=',')
    # rgb_image=cv2.imread('/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/move_detection/csv/empty_chess_board_new/empty_rgb_image.png')
    # # ((272, 205), (312, 203), (272, 237), (314, 235), (292, 220), 'd5') structure in chessboard_square_corners
    # pcl_count,mean_rgb_value,occupancy=dip.depth_pcl_counting(depth_image,rgb_image,filter_points)
    # print(pcl_count,mean_rgb_value)
    # print('occupancy:',occupancy)
