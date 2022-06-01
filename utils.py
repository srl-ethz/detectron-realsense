import math
import pyrealsense2 as rs

def get_record_counter(file):
        # Determine the record counter
        with open(file, 'r+', encoding='utf8') as f:
            # Process lines and get first line, there should only be one line
            lines = (line.strip() for line in f if line)
            x = [int(float(line.replace('\x00', ''))) for line in lines]
            ret = x[0]

            # Delete all file contents
            f.truncate(0)

            # Write back to file beginning
            f.seek(0)
            f.write(str(ret + 1))

        return ret

def truncate(number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper


RECORD_COUNTER = get_record_counter('counter')
LOG_NAME = 'test'
LOG_FILE = f'logs/{LOG_NAME}_{RECORD_COUNTER}.csv'
VIDEO_FILE = f'videos/{LOG_NAME}_{RECORD_COUNTER}.avi'
VIDEO_RAW_FILE = f'videos/{LOG_NAME}_{RECORD_COUNTER}_raw.avi'
VIDEO_DEPTH_FILE = f'videos/{LOG_NAME}_{RECORD_COUNTER}_depth.avi'
VIDEO_GRASP_FILE = f'videos/{LOG_NAME}_{RECORD_COUNTER}_grasp.avi'

class RSCameraMockup():
    def __init__(self):
        self.width = 640
        self.height = 480
        self.intrinsics = rs.intrinsics()
        self.intrinsics.coeffs = [-0.053769, 0.0673601, -0.000281265, 0.000637035, -0.0215778]
        self.intrinsics.ppx = 323.432 
        self.intrinsics.ppy = 242.815
        self.intrinsics.fx = 384.879
        self.intrinsics.fy = 384.372
        self.intrinsics.width = 640
        self.intrinsics.height = 480
        self.intrinsics.model = rs.distortion.inverse_brown_conrady
        self.depth_scale = 0.0010000000474974513

    def deproject(self, intrinsics, x, y, depth):
        return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)

    def project(self, intrinsics, point):
        return rs.rs2_project_point_to_pixel(intrinsics, point)

    def release(self):
        print('Release called')