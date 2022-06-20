from audioop import mul
import socket 
import time
import cv2
import numpy as np
from realsense import RSCamera
import imagezmq
from utils import truncate
from threading import Thread
import multiprocessing

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
        manager = multiprocessing.Manager()
        self.ret_dict = manager.dict()

    def encode_color(self, color, ret_dict):
        ret, jpg_frame = cv2.imencode(
            '.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        ret_dict['color'] = jpg_frame 

    def encode_depth(self, depth, ret_dict):
        ret, frame_depth = cv2.imencode(
            '.png', depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
        ret_dict['depth'] = frame_depth

    def send_frames(self, color, depth):
        start = time.time()
        t1 = Thread(target=self.encode_color, args=(color, self.ret_dict))
        t2 = Thread(target=self.encode_depth, args=(depth, self.ret_dict))
        t1.start()
        t2.start()

        t1.join()
        t2.join()
        jpg_color = self.ret_dict['color']
        jpg_depth = self.ret_dict['depth']

        print(f'threaded finished after {(threaded_time := time.time() - start) * 1000}')

        start = time.time()
        ret, frame_depth = cv2.imencode(
            '.png', depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
        # jpg_depth = frame_depth if ret else None
   

        ret, jpg_frame = cv2.imencode(
            '.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        # jpg_color = jpg_frame if ret else None

        print(f'non-threaded finished after {(non_threaded_time := time.time() - start) * 1000}')

        print(f'Ratio: {truncate(threaded_time/non_threaded_time, 3)}')

        self.sender_color.send_jpg(self.hostname, jpg_color)
        self.sender_depth.send_jpg(self.hostname + '_depth', jpg_depth)
    

if __name__=='__main__':
    sender = VideoSender('tcp://10.39.60.6:5555')
    cam = RSCamera()
    while True:
        color, depth = cam.get_raw_color_aligned_frames()
        sender.send_frames(color, depth)
        print('sent frames')