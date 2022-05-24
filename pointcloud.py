import open3d as o3d
import numpy as np
import cv2
import pyrealsense2 as rs2

class GraspCandidate:
    def __init__(self):
        self.pointcloud = o3d.cuda.pybind.geometry.PointCloud()

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

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print(points.shape)

        # for point, color in zip(points, colors):
        #     pass

        rows, cols = np.where(colors != [0,0,0])
        res_points = points[rows]
        res_colors = colors[rows]
        print(res_points.shape)
        pcd.points = o3d.utility.Vector3dVector(res_points)
        pcd.colors = o3d.utility.Vector3dVector(res_colors)


        # pcd = pcd.voxel_down_sample(0.03)
        self.pointcloud.points = pcd.points
        self.pointcloud.colors = pcd.colors

    def save_pcd(self, file):
        o3d.io.write_point_cloud(file, self.pointcloud)


    def visualise_pcd(self, file):
        vis = o3d.visualization.Visualizer()
        vis.create_window('PCD', width=1280, height=720)
        pcd = o3d.io.read_point_cloud(file)
        vis.add_geometry(pcd)
        while True:
            try:
                vis.poll_events()
                vis.update_renderer()
            except KeyboardInterrupt:
                vis.destroy_window()
                del vis


if __name__=='__main__':
    grasp = GraspCandidate()
    grasp.visualise_pcd('pcd/pointcloud_bottle_36.pcd')
        
