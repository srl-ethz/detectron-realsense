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
    '''
    This class serves to send a pair of image frames from an Intel RealSense series camera 
    over the local network using imagezmq. We are compressing the frames before sending them.
    '''
    def __init__(self, addr) -> None:
        
        # One sender for color images, one for depth images
        self.sender_color = imagezmq.ImageSender(
            connect_to=addr)
        self.sender_depth = imagezmq.ImageSender(
            connect_to=addr)
        
        self.hostname = socket.gethostname()
        
        # Quality of JPEG and PNG compression
        self.jpeg_quality = 95
        self.png_quality = 2

        # Handle thread-safe access of compression results
        manager = multiprocessing.Manager()
        self.ret_dict = manager.dict()

    
    def encode_color(self, color, ret_dict):
        '''
        Compress an OpenCV image frame with JPEG compression. Returns a bytestring in the given 
        return dictionary. 
        '''

        ret, jpg_frame = cv2.imencode(
            '.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        ret_dict['color'] = jpg_frame 

    def encode_depth(self, depth, ret_dict):
        '''
        Compress an OpenCV image frame with PNG compression. Returns a bytestring in the given 
        return dictionary. 
        '''

        ret, frame_depth = cv2.imencode(
            '.png', depth, [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_quality])
        ret_dict['depth'] = frame_depth

    def send_frames(self, color, depth):
        '''
        Send the compressed frames using imagezmg and Python threads. 
        '''
        # Threads are used here instead of multiprocessing since they provide quick setup and teardown
        # While there is still the global interpreter lock at play and we do not have multiple processes
        # we are creating and destroying a lot of threads, so the increased overhead of multiprocesssing
        # is likely to slow things down instead of accelerate them compared to normal threads

        t1 = Thread(target=self.encode_color, args=(color, self.ret_dict))
        t2 = Thread(target=self.encode_depth, args=(depth, self.ret_dict))
        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Get results from threads
        jpg_color = self.ret_dict['color']
        png_depth = self.ret_dict['depth']
       
        self.sender_color.send_jpg(self.hostname, jpg_color)
        self.sender_depth.send_jpg(self.hostname + '_depth', png_depth)
    
# This loop will run if this file is invoked directly, which should be the normal use case
if __name__=='__main__':
    sender = VideoSender('tcp://10.31.62.7:5555')
    cam = RSCamera()
    while True:
        start = time.time()
        color, depth = cam.get_raw_color_aligned_frames()
        sender.send_frames(color, depth)
        end = time.time() - start
        print(f'total loop took: (ms) {end}')
        # print('sent frames')