import socket 
import time
import cv2
import numpy as np
from realsense import RSCamera
import imagezmq
from utils import truncate

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
        
        start = time.time()
        ret, frame_depth = cv2.imencode(
            '.png', depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
        print(f'PNG time (ms): {(time.time() - start) * 1000}')
        
        start = time.time()
        ret, jpg_frame_depth = cv2.imencode(
            '.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        # print(f'JPG time (ms): {(time.time() - start) * 1000}')
        # print(f'Original length: {depth.nbytes}')
        # print(f'PNG compression ratio: {truncate(len(frame_depth)/depth.nbytes, 3)}')
        # print(f'JPG compression ratio: {truncate(len(jpg_frame_depth)/depth.nbytes, 3)}')

        # print(f'PNG: {len(frame_depth)}')
        # print(f'JPG: {len(jpg_frame_depth)}')
        # print(f'Ratio: {truncate(len(jpg_frame_depth)/len(frame_depth), 3)}')

        self.sender_color.send_jpg(self.hostname, jpg_frame)
        self.sender_depth.send_jpg(self.hostname + '_depth', frame_depth)
        # print('Sent frames')

    # def send_frames(self, color, depth):
    #     ret, jpg_frame = cv2.imencode(
    #         '.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
    #     self.sender_color.send_jpg(self.hostname, jpg_frame)
    #     self.sender_depth.send_image(self.hostname + '_depth', depth)
    #     print('Sent frames')

if __name__=='__main__':
    sender = VideoSender('tcp://10.10.10.122:5555')
    cam = RSCamera()
    while True:
        color, depth = cam.get_raw_color_aligned_frames()
        sender.send_frames(color, depth)
        print('sent frames')