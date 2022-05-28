import open3d as o3d
import numpy as np
import cv2
import time


class GraspCandidate:
    def __init__(self, file=None):
        self.pointcloud = o3d.geometry.PointCloud()
        if file is not None:
            self.pointcloud = o3d.io.read_point_cloud(file)
        self.bbox = None
        self.main_axis = None

    def set_point_cloud_from_aligned_frames(self, frame, depth_frame, cam_intrinsics):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_depth = o3d.geometry.Image(depth_frame)
        img_color = o3d.geometry.Image(frame)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(cam_intrinsics.width, cam_intrinsics.height, cam_intrinsics.fx, cam_intrinsics.fy, cam_intrinsics.ppx, cam_intrinsics.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
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
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        
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

        self.pointcloud.points = o3d.utility.Vector3dVector(new_points)
        self.pointcloud.colors = o3d.utility.Vector3dVector(new_colors)

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
        vis = o3d.visualization.Visualizer()
        vis.create_window('PCD', width=1280, height=720)
        
        # Add default coordinate frame
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # vis.add_geometry(mesh)
        for g in geometries:
            vis.add_geometry(g)
        while True:
            try:
                vis.poll_events()
                vis.update_renderer()
            except KeyboardInterrupt:
                vis.destroy_window()
                del vis
        pass

    def find_centroid(self, add_to_pcd=False):
        mean, _ = self.pointcloud.compute_mean_and_covariance()
        return mean

    def find_largest_axis(self):
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(self.pointcloud.points, robust=True)
        self.bbox = bbox
        box_points = np.asarray(bbox.get_box_points())
        center = bbox.center
        extents = bbox.extent
        
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

        # Use for visualisation          
        # axis_points = [np.add(center, np.dot(main_axis, 0.1*k)) for k in range(8)]
        self.main_axis = main_axis
        return main_axis, axis_plane_1, axis_plane_2


    def find_all_grasping_candidates(self):
        main_axis, axis_plane_1, axis_plane_2 = self.find_largest_axis()
        max_idx = np.argmax(self.bbox.extent)
        if max_idx == 0:
            max_idx = 1
        elif max_idx == 1:
            max_idx = 0
        
        centroid = self.find_centroid()

        points = np.asarray(self.pointcloud.points)
        colors = np.asarray(self.pointcloud.colors)
        
        rows = np.where(abs(points[:, max_idx] - centroid[max_idx]) < 0.008)
        colors[rows] = (255,0,0)
        self.pointcloud.colors = o3d.utility.Vector3dVector(colors)
        return rows
    


    def find_grasping_points(self):
        rows = self.find_all_grasping_candidates()
        self.pointcloud.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        points = np.asarray(self.pointcloud.points)
        colors = np.asarray(self.pointcloud.colors)
        points = points[rows]
        colors = colors[rows]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd.estimate_normals()
        pcd = pcd.uniform_down_sample(2)
        

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals = np.asarray(pcd.normals)

        curr = 0
        last = len(points) - 1
        weights = np.asarray([2, -0.25, 2.5, -5, 100, -100])
        best_partners = np.full((len(points), 3), -1.0)

        while curr < last:
            if best_partners[curr, 2] == 1:
                curr += 1
                continue
        
            point = points[curr]
            normal = normals[curr]
            max_score = 0
            max_idx = curr
            
            curr += 1
            for i in range(curr, last):
                p = points[i]
                n = normals[i]
                # Points should have an opposing normal, can't use signs because the estimation is quick to flip a normal
                k1 = weights[0] * abs(np.dot(normal, n))

                # If the point is not orthogonal to the main axis, we want to punish it
                k2 = weights[1] * abs(np.dot(normal, self.main_axis))
                
                # The normals of the point should ideally align with the difference vector of the points
                d = np.subtract(point, p)
                k3 = weights[2] * (abs(np.dot(d, normal)) + abs(np.dot(d, n))) 

                # The difference vector should be normal to the main axis
                k4 = weights[3] * abs(np.dot(d, self.main_axis))
                
                # Reward points that are far apart
                k5 = weights[4] * np.linalg.norm(d)
                
                # Punish differences in the z component
                k6 = weights[5] * abs(point[2] - p[2])
                # Weighted sum is score
                score = k1 + k2 + k3 + k4 + k5
            
                if score > max_score:
                    # print(f'updated score to {score}')
                    max_score = score
                    max_idx = i 
                    best_partners[curr, 0] = max_idx
                    best_partners[curr, 1] = score
            
            # Set indices to indicate the best pairing
            best_partners[curr, 0] = max_idx
            best_partners[max_idx, 0] = curr
            
            # Set score
            best_partners[curr, 1] = score
            best_partners[max_idx, 1] = score
            
            # Set mark to done so that we can save some computation time 
            # If a point is done, we won't consider possible pairings again
            # We can do this because our evaluation function is symmetrical
            best_partners[curr, 2] = 1
            best_partners[max_idx, 2] = 1     

                           
            
        # Find best scoring point and its partner
        max_idx = np.argmax(best_partners[:, 1])
        max_idx_partner = int(best_partners[max_idx, 0])

        # Visualize with color
        colors[max_idx] = [0,255,0]
        colors[max_idx_partner] = [0,255,0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.points = o3d.utility.Vector3dVector(points)

        # New point cloud since visualization is flaky
        grasp_points = [points[max_idx], points[max_idx_partner]]
        print(grasp_points)
        grasp_colors = [[0,255,0] for _ in range(len(grasp_points))]
        grasp_pcd = o3d.geometry.PointCloud()
        grasp_pcd.points = o3d.utility.Vector3dVector(grasp_points)
        grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_colors)
        
        self.visualize_geometries([self.pointcloud, grasp_pcd])



if __name__=='__main__':
    grasp = GraspCandidate('pcd/pointcloud_bottle_36.pcd')
    grasp.find_grasping_points()
    # grasp.visualise_pcd()

        
