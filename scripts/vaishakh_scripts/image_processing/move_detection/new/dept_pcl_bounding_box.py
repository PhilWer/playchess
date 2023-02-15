import numpy as np
import open3d as o3d
import yaml
import copy
import cv2


class DepthProcessing():
    def __init__(self):
        self.verbose = True
        self.K = np.array([523.2994491601762, 0.0, 312.0471722095908, 0.0,
                          524.1979376240457, 249.9013550067579, 0.0, 0.0, 1.0])  # Intrinsic matrix
        self.width = 640
        self.height = 480

    def downsample(self, pcl, voxel_size):
        downsample = pcl.voxel_down_sample(voxel_size=0.01)
        if self.verbose:
            o3d.vizualization.draw_geometries([downsample])

    def visualize_points(self, pcl, points):
        pcd_red = o3d.geometry.PointCloud()
        xyz = np.asarray(points)
        pcd_red.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcl, pcd_red])
        return pcd_red

    def normals(self, pcl):
        pcl.estimate_normals(
            serch_param=o3d.geometry.KDTreeSerchParamHybrid(radius=0.1, max_nn=30))
        if self.verbose:
            o3d.visualization.draw_geometries([pcl], point_show_normal=True)
            return pcl.normals

    def create_pointcloud_from_depth(self):
        # Callback function to recreate the desired pointcloud starting from the acquired rgb and depth images.
        self.acquire = True
        if self.acquire:

            # Import the rgb image to save colors of the points
            # cv2.imread(imported_rgb_image) #DA VERIFICARE SE FUNZIONA, SE NO DEVO SALVARE LE IMMAGINI ACQUISITE DA TIAGO
            self.rgb_im = cv2.imread(
                '/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/move_detection/csv/random_ordered_board/rgb_image.png')
            # Convert the image from BGR to RGB
            self.rgb_im = cv2.cvtColor(self.rgb_im, cv2.COLOR_BGR2RGB) / 255.0
            self.depth_image = np.genfromtxt(
                "/home/vaishakh/tiago_public_ws/src/playchess/scripts/vaishakh_scripts/image_processing/move_detection/csv/random_ordered_board/depth_image.csv", delimiter=",")

            ii = []
            jj = []
            # depth camera parameters:
            FX_DEPTH = 523.2994491601762/1
            FY_DEPTH = 524.1979376240457/1
            CX_DEPTH = 312.0471722095908/1
            CY_DEPTH = 249.9013550067579/1

            # Loop over all pixels in the depth image
            height, width = self.depth_image.shape
            for i in range(height):
                for j in range(width):
                    # Check if the pixel is inside the quadrilateral
                    ii.append(i)
                    jj.append(j)

            # Convert the row and column indices to numpy arrays
            ii = np.array(ii)
            jj = np.array(jj)

            # Extract the depth values for the pixels inside the quadrilateral
            z = self.depth_image[ii, jj]
            # Compute constants:
            xx = (jj - CX_DEPTH) / FX_DEPTH
            yy = (ii - CY_DEPTH) / FY_DEPTH
            pcd = np.dstack((xx * z, yy * z, z)).reshape((-1, 3))
            # changing reference frame
            # pcd = np.dot(np.linalg.inv(R), pcd.T).T - np.dot(np.linalg.inv(R), T)
            # Convert to Open3D.PointCLoud:
            pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(pcd)
           # self.pcd.colors = o3d.utility.Vector3dVector(self.pts_color)

            if self.verbose:
                o3d.visualization.draw_geometries([self.pcd])
        return self.pcd

    def create_pointcloud(self, points, color):
        # Function that creates a PointCloud starting from a matrix of xyz points [nx3 matrix].
        # color: array that tells the desired color for the PointCloud (Red: [1,0,0], Green: [0, 1, 0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)  # Paint the points i red
        if self.verbose:
            o3d.visualization.draw_geometries([pcd])
        return pcd

    def uvz_to_xyz(self, uv, z, K):
        # From: https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
        # uv: [list] pixel coordinates of the point in the image
        # z: [float] depth value in the same measurement unit of the output
        # K: [list[list]] intrinsic camera matrix (3x3)
        # Format the arrays
        z = z[:, np.newaxis]  # (N, 1)
        ones = np.ones((z.shape))  # (N, 1)
        uv = np.hstack((uv, ones, np.reciprocal(z)))  # (N, 4)
        # Attach a dummy dimension so that matmul sees it as a stack of (4, 1) vectors
        uv = np.expand_dims(uv, axis=2)  # (N, 4, 1)
        # Invert the intrinsic matrix
        fx, S, cx, fy, cy = K[0], K[1], K[2], K[4], K[5]
        K_inv = [[1/fx, -S/(fx*fy), (S*cy-cx*fy)/(fx*fy), 0],
                 [0, 1/fy, -cy/fy, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]
                 ]
        # Compute the spacial 3D coordinates for the points
        xyz = z[:, np.newaxis] * np.matmul(K_inv, uv)   # (N, 4, 1)
        return xyz[:, :3].reshape(-1, 3)

    def create_bounding_box(self, pcl, length, width, R, centro, reference_normal, box_extent, margin, translation):
        # Function to create a bounding box given information about the PointCloud and the location of the desired box.
        # length: extension of the chessboard in the y direction.
        # width: extension of the chessboard in the x direction.
        # R: rotation matrix computed looking at the normals of the chessboard squares.
        # centro: center of the chessboard.
        # reference normal: dimensions of the normal to the chessboard in the xyz directions.
        # box_extent: extension in the z direction of the bounding box with which I want to crop the PointCloud
        # margin: external margin to make sure to take all the chessboard.
        # translation:

        old_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        old_frame.translate(centro)
        self_created_frame = copy.deepcopy(old_frame)
        self_created_frame.rotate(R, center=centro)

        # Creation of the bounding box
        centro[2] = centro[2] - translation
        bounding_box = o3d.geometry.OrientedBoundingBox(
            centro, R, [length + margin, width + margin, box_extent])

        # Visualize the bounding box over the original PointCloud
        if self.verbose:
            # pass
            o3d.visualization.draw_geometries(
                [pcl, bounding_box, self_created_frame])
        return bounding_box, self_created_frame

    def get_rot_matrix(self, pcl, normals, centro):
        # Function to compute the rotation matrix to rotate the reference frame of the bounding box that I want to create and allign it to the ref frame of the pointcloud seen by TIAGo.

        # Choose the reference normal to keep to compute rotation
        for n in normals:
            if n[0] > 0:
                reference_normal = n
                break
        if self.verbose:
            print('NORMAL: ' + str(reference_normal))

        R = pcl.get_rotation_matrix_from_xyz((- reference_normal[1], 0, 0))
        pcl_r = copy.deepcopy(pcl)
        pcl_r.rotate(R, center=(centro[0], centro[1], centro[2]))
        if self.verbose:
            o3d.visualization.draw_geometries([pcl, pcl_r])
        return R, reference_normal


if __name__ == '__main__':
    id = DepthProcessing()
    pcd = id.create_pointcloud_from_depth()
    normals = id.normals(pcd)
    R, reference_normal = id.get_rot_matrix(pcd, normals, [10, 10, 10])
    id.create_bounding_box(pcd, 100, 100, R,[10, 10, 10], reference_normal, 100, 100, 0)
