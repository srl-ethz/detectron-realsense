import numpy as np
from math import cos, sin, pi




def transform_frame_EulerXYZ(euler_angles, translation, point, degrees=True):
    '''
    We translate from frame A to frame B using homogenous coordinates.
    euler_angles describe the rotation of frame A relative to frame B.
    translation describes the offset of frame A relative to frame B. 
    point is the point in frame A that we want to convert to frame B.
    If degrees is set to true, the angles will be passed in as degrees
    '''

    R = get_rotation_matrix_EulerXYZ(euler_angles, degrees)

    # Total transformation matrix which includes the transformation matrix and the translation in homogenous coordinates
    T = np.eye(4, 4)
    T[:3, :3] = R
    T[:3, 3] = translation

    
    
    # Apply transformation on target point we want to transform into different coordinate system
    return np.dot(T, point)


def get_rotation_matrix_EulerXYZ(euler_angles, degrees=True):
    if degrees:
        alpha = euler_angles[0] * pi / 180
        beta = euler_angles[1] * pi / 180
        gamma = euler_angles[2] * pi / 180
    else:
        alpha = euler_angles[0]
        beta = euler_angles[1]
        gamma = euler_angles[2]
    
    # Rotation matrix around x-axis - alpha is roll
    rx = [[1, 0, 0],
        [0, cos(alpha), sin(alpha)],
        [0, -sin(alpha), cos(alpha)]]

    # Rotation matrix around y-axis - beta is pitch
    ry = [[cos(beta), 0, -sin(beta)],
        [0, 1, 0], 
        [sin(beta), 0, cos(beta)]]

    # Rotation matrix around z-axis - gamma is yaw
    rz = [[cos(gamma), sin(gamma), 0],
        [-sin(gamma), cos(gamma), 0],
        [0, 0, 1]]

    # Multiplications such that the total rotation matrix R is R = rz * ry * rx
    R1 = np.matmul(ry, rx)
    R = np.matmul(rz, R1)
    return R
