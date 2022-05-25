import open3d as o3d
import numpy as np
import cv2


class GraspCandidate:
    def __init__(self, file=None):
        self.pointcloud = o3d.cuda.pybind.geometry.PointCloud()
        if file is not None:
            self.pointcloud = o3d.io.read_point_cloud(file)
        self.bbox = None

    def set_point_cloud_from_aligned_frames(self, frame, depth_frame, cam_intrinsics):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_depth = o3d.geometry.Image(depth_frame)
        img_color = o3d.geometry.Image(frame)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(cam_intrinsics.width, cam_intrinsics.height, cam_intrinsics.fx, cam_intrinsics.fy, cam_intrinsics.ppx, cam_intrinsics.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        pcd = pcd.voxel_down_sample(0.03)
        self.pointcloud.points = pcd.points
        self.pointcloud.colors = pcd.colors

    def set_point_cloud_from_aligned_masked_frames(self, frame, depth_frame, cam_intrinsics):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_depth = o3d.geometry.Image(depth_frame)
        img_color = o3d.geometry.Image(frame)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(cam_intrinsics.width, cam_intrinsics.height, cam_intrinsics.fx, cam_intrinsics.fy, cam_intrinsics.ppx, cam_intrinsics.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        # ROI selection

        # Get points and colors
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Filter out points where the color is exactly black - this will filter out all points that were masked before
        rows, _ = np.where(colors != [0,0,0])
        res_points = points[rows]
        res_colors = colors[rows]
        # Set the points and color of the point cloud to the masked point
        pcd.points = o3d.utility.Vector3dVector(res_points)
        pcd.colors = o3d.utility.Vector3dVector(res_colors)

        # Downsampling to reduce computation time later on
        pcd = pcd.voxel_down_sample(0.01)

        # Update point cloud with calculated points
        self.pointcloud.points = pcd.points
        self.pointcloud.colors = pcd.colors

    def save_pcd(self, file):
        o3d.io.write_point_cloud(file, self.pointcloud)

    def add_points_and_color_to_pcd(self, points, color):
        curr_points = np.asarray(self.pointcloud.points)
        curr_colors = np.asarray(self.pointcloud.colors)

        new_points = np.concatenate((curr_points, points), axis=0)
        new_colors = np.concatenate((curr_colors, [color for _ in range(len(points))]), axis=0)

        self.pointcloud.points = new_points
        self.pointcloud.colors = new_colors

    def visualise_pcd(self, file=None):
        if file is None:
            pcd = self.pointcloud
        else: 
            pcd = o3d.io.read_point_cloud(file)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window('PCD', width=1280, height=720)
        
        # Add default coordinate frame
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # vis.add_geometry(mesh)

        vis.add_geometry(pcd)
        while True:
            try:
                vis.poll_events()
                vis.update_renderer()
            except KeyboardInterrupt:
                vis.destroy_window()
                del vis

    def visualize_geometries(self, geometries):
        # Pass in a list of geometries to visualize
        pass

    def find_centroid(self, add_to_pcd=False):
        mean, cov = self.pointcloud.compute_mean_and_covariance()
        
        if add_to_pcd: 
            # Add to pointcloud for visualisation
            points = np.asarray(self.pointcloud.points)
            colors = np.asarray(self.pointcloud.colors)
            new_points = np.concatenate((points, [mean, ]), axis=0)
            new_colors = np.concatenate((colors, [[255, 0, 0],]), axis=0)

            self.pointcloud.points = o3d.utility.Vector3dVector(new_points)
            self.pointcloud.colors = o3d.utility.Vector3dVector(new_colors)
        
        return mean

    def find_largest_axis(self, add_to_pcd=False):
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(self.pointcloud.points, robust=True)
        self.bbox = bbox
        box_points = np.asarray(bbox.get_box_points())
        center = bbox.center
        extents = bbox.extent

        T = np.eye(4, 4)
        T[:3, :3] = bbox.R
        T[:3, 3] = center
        
        # Get index of maximum in array to determine longest axis and create a unit vector in the direction of the longest axis
        max_idx = np.argmax(extents)
        unit_vec = np.zeros(3)
        unit_vec[max_idx] = 1


        # These are the other unit vectors perpendicular to the one above
        # We will use these to create our grasping plane
        idx_1 = (max_idx + 1) % 3
        idx_2 = (max_idx + 2) % 3
        
        unit_vec_plane_1 = np.zeros(3)
        unit_vec_plane_2 = np.zeros(3)
        unit_vec_plane_1[idx_1] = 1
        unit_vec_plane_2[idx_2] = 1
        
        # Now, apply the transformation to transform the vector to the bounding box rotation
        # This gives the vector that defines the direction of the axis of the object
        main_axis = np.dot(bbox.R, unit_vec)
        axis_plane_1 = np.dot(bbox.R, unit_vec_plane_1)
        axis_plane_2 = np.dot(bbox.R, unit_vec_plane_2)
            
        if add_to_pcd: 
            # Add to pointcloud for visualisation
            points = np.asarray(self.pointcloud.points)
            colors = np.asarray(self.pointcloud.colors)

            # Add a few points visualising the main axis
            axis_points = [np.add(center, np.dot(main_axis, 0.1*k)) for k in range(8)]
            
            new_points = np.concatenate((points, axis_points), axis=0)
            new_colors = np.concatenate((colors, [[255, 0, 0] for _ in range(len(axis_points))]), axis=0)

            # Add corner points of box
            new_points = np.concatenate((new_points, box_points), axis=0)
            new_colors = np.concatenate((new_colors, [[255, 0, 0] for _ in range(len(box_points))]), axis=0)

            self.pointcloud.points = o3d.utility.Vector3dVector(new_points)
            self.pointcloud.colors = o3d.utility.Vector3dVector(new_colors)

        return main_axis, axis_plane_1, axis_plane_2


    def find_all_grasping_candidates(self):
        main_axis, axis_plane_1, axis_plane_2 = self.find_largest_axis()
        max_idx = np.argmax(self.bbox.extent)
        if max_idx == 0:
            max_idx = 1
        elif max_idx == 1:
            max_idx = 0
        

        print(self.bbox.extent)
        centroid = self.find_centroid()

        points = np.asarray(self.pointcloud.points)
        rotated_points = np.matmul(points, self.bbox.R)
        colors = np.asarray(self.pointcloud.colors)
        interest = abs(points[:, max_idx] - centroid[max_idx])
        print(np.max(interest))
        print(np.min(interest))
        print(interest)
        
        rows = np.where(abs(points[:, max_idx] - centroid[max_idx]) < 0.0125)
        print(rows)
        colors[rows] = (255,0,0)
        # self.pointcloud.points = o3d.utility.Vector3dVector(points[rows])
        # self.pointcloud.colors = o3d.utility.Vector3dVector(colors[rows])

        # self.pointcloud.points = o3d.utility.Vector3dVector(points[rows])
        self.pointcloud.colors = o3d.utility.Vector3dVector(colors)
        

        pass
    
    def find_opposing_point(self):
        
        pass

    def find_grasping_points(self):
        pass


if __name__=='__main__':
    grasp = GraspCandidate('pcd/pointcloud_bottle_36.pcd')
    grasp.find_all_grasping_candidates()
    grasp.visualise_pcd()

        
