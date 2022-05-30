import socket 
import time
import cv2
import numpy as np
from realsense import RSCamera
import imagezmq

class VideoSender:
    def __init__(self, addr) -> None:
        # self.sender_color = imagezmq.ImageSender(
        #     connect_to='tcp://10.10.10.232:5555')
        self.sender_color = imagezmq.ImageSender(
            connect_to=addr)

        self.sender_depth = imagezmq.ImageSender(
            connect_to=addr)
        self.hostname = socket.gethostname()
        self.jpeg_quality = 95

    def send_frames(self, color, depth):
        ret, jpg_frame = cv2.imencode(
            '.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        ret, jpg_frame_depth = cv2.imencode(
            '.jpg', depth, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        self.sender_color.send_jpg(self.hostname, jpg_frame)
        self.sender_depth.send_jpg(self.hostname + '_depth', jpg_frame_depth)
        print('Sent frames')