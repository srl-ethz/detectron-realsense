import open3d as o3d
import numpy as np
import cv2
import pyrealsense2 as rs2

class GraspCandidate:
    def __init__(self, file=None):
        self.pointcloud = o3d.cuda.pybind.geometry.PointCloud()
        if file is not None:
            self.pointcloud = o3d.io.read_point_cloud(file)

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

    def visualise_pcd(self, file=None):
        if file is None:
            pcd = self.pointcloud
        else: 
            pcd = o3d.io.read_point_cloud(file)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window('PCD', width=1280, height=720)
        
        vis.add_geometry(pcd)
        while True:
            try:
                vis.poll_events()
                vis.update_renderer()
            except KeyboardInterrupt:
                vis.destroy_window()
                del vis

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
        

    def find_largest_axis(self):
        pass

    def find_opposing_point(self):
        pass

    def find_grasping_points(self):
        pass


if __name__=='__main__':
    grasp = GraspCandidate('pcd/pointcloud_bottle_36.pcd')
    # grasp.visualise_pcd('pcd/pointcloud_bottle_36.pcd')
    grasp.find_centroid()
    grasp.visualise_pcd()

        
