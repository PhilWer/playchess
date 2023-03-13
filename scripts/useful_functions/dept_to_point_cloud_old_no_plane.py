import numpy as np
import open3d as o3d
import pickle
import cv2
import matplotlib.path as mpath
import copy
import csv
import yaml
import gc


class DepthImageProcessing:
    def __init__(self):
        self.debug = 0

    def depth_to_point_cloud(self, depth_array_from_csv,rgb_image, filter_points):
        self.depth_image = depth_array_from_csv
        
        if self.debug:
            cv2.imshow('depth_image', self.depth_image)
            print('depth image min value:{} /n max value:{}'.format(
                np.min(self.depth_image), np.max(self.depth_image)))
            cv2.waitKey(100)

        # depth camera parameters: # TODO. Receive from /xtion/camera_info topic
        FX_DEPTH = 523.2994491601762/1
        FY_DEPTH = 524.1979376240457/1
        CX_DEPTH = 312.0471722095908/1
        CY_DEPTH = 249.9013550067579/1

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
        xx = (jj -CX_DEPTH) / FX_DEPTH
        yy = (ii -CY_DEPTH) / FY_DEPTH
        pcd = np.dstack((xx * z, yy * z, z)).reshape((-1, 3))

        ###################################################################
    

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

        if  self.debug:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=50)
            o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])

        return pcd_o3d

    def transforming_optical_to_base_frame(self, pcd_o3d, quat, trans):
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
        #inv_trans_mat = np.linalg.inv(trans_mat)
        

        if   self.debug:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=50)
            o3d.visualization.draw_geometries([pcd_o3d,mesh_frame])

        return pcd_o3d
        ##################################################

    def neglect_pcd_above_and_below_certain_height(self, h, pcd_o3d):
        aabb = pcd_o3d.get_axis_aligned_bounding_box()
        vertices = (np.asarray(aabb.get_box_points()))
        colors = np.asarray(pcd_o3d.colors)
        # print(vertices)
        points_for_filtering = np.asarray(pcd_o3d.points)

        points = points_for_filtering[points_for_filtering[:, 2]<= vertices[0, 2]+h]
        points = points_for_filtering
        points_for_filtering = points
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        aabb = pcd_o3d.get_axis_aligned_bounding_box()
        vertices = (np.asarray(aabb.get_box_points()))
        # print(base_points)
        # Remove all points above a height of h
        points_for_filtering = np.asarray(pcd_o3d.points)

        h=(vertices[3, 2]-vertices[0,2])
        
        # if h>=10:
        #     h=h/2
        # else:
        #     h=0
        # points = points_for_filtering[points_for_filtering[:, 2]
        #                               >= vertices[3, 2]-h]
        
        pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        if len(np.asarray(pcd_o3d.points))<1000:
             # Assign colors to the filtered points
            filtered_colors = np.zeros((len(points), 3), dtype=np.float32)
            for i, point in enumerate(points):
                index = np.where((points_for_filtering == point).all(axis=1))[0][0]
                filtered_colors[i] = colors[index]
            pcd_o3d.colors = o3d.utility.Vector3dVector(filtered_colors)
            mean_rgb_value = np.mean(filtered_colors, axis=0)
            occupancy=False
            if len(np.asarray(pcd_o3d.points))>=20:
                occupancy=True
        else:
            mean_rgb_value= 0
        aabb = pcd_o3d.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        vertices = (np.asarray(aabb.get_box_points()))
        occupancy=False
        # Visualize:
        if not self.debug:

            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=10, origin=[vertices[3, 0],  vertices[3, 1], vertices[3, 2]-h])
            o3d.visualization.draw_geometries([pcd_o3d, aabb,mesh_frame])
        return len(np.asarray(pcd_o3d.points)),mean_rgb_value,occupancy

    def depth_pcl_counting(self, depth_image,rgb_image, filter_points, quat=[-0.673, 0.679, -0.202, 0.214], trans=[0.232, 0.016, 1.318], h=40):
        pcl = self.depth_to_point_cloud(depth_image,rgb_image, filter_points)
        pcl_1 = self.voxel_down_and_outlier_remove(pcl)
        pcl_2= self.transforming_optical_to_base_frame(pcl_1, quat, trans)
        pcl_3,mean_rgb_value,occupancy = self.neglect_pcd_above_and_below_certain_height(h, pcl_2)
       
        return pcl_3, mean_rgb_value,occupancy


if __name__ == '__main__':
    dip = DepthImageProcessing()
    
    with open(r'/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/yaml/corners_names_empty_square.yaml', 'r') as file:
            chessboard_square_corners = yaml.load(file,Loader=yaml.Loader)
            for i in chessboard_square_corners:
                #if i[5]=='c8':
                    a=i
                    print(a[5])
                    filter_points =np.array([a[0],a[1],a[2],a[3]])
                    #filter_points = np.array([[165, 127],[446, 116],[524, 388],[120, 410]]) #for full chess_board

                    depth_image = np.genfromtxt('/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/move_detection/csv/initial_board_state/empty_depth_image.csv', delimiter=',')
                    rgb_image=cv2.imread('/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/move_detection/csv/initial_board_state/empty_rgb_image.png')
                    # ((272, 205), (312, 203), (272, 237), (314, 235), (292, 220), 'd5') structure in chessboard_square_corners
                    pcl_count,mean_rgb_value,occupancy=dip.depth_pcl_counting(depth_image,rgb_image,filter_points,quat=[-0.673, 0.679, -0.202, 0.214],trans=[0.232, 0.016, 1.318])
                    gc.collect()
                    print(pcl_count,mean_rgb_value)
                    print('occupancy:',occupancy)