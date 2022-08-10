import sys

import numpy as np
import cv2
import imagezmq
import time
from logger import Logger
from utils import RECORD_COUNTER

from realsense import RSCamera

# image_hub = imagezmq.ImageHub()
# while True:  # show streamed images until Ctrl-C
#     rpi_name, jpg_buffer = image_hub.recv_jpg()
#     image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
#     # see opencv docs for info on -1 parameter
#     cv2.imshow(rpi_name, image)  # 1 window for each RPi
#     cv2.waitKey(1)
#     image_hub.send_reply(b'OK')


class VideoReceiver:
    def __init__(self) -> None:
        self.image_hub = imagezmq.ImageHub()
        
        self.delays = []
        self.logger = Logger()

    def recv_frames(self):
        start = time.time()
        color_header, color_jpg_buffer = self.image_hub.recv_jpg()
        color = cv2.imdecode(np.frombuffer(color_jpg_buffer, dtype='uint8'), -1)
        self.image_hub.send_reply(b'OK')
        
        depth_header, depth_buffer = self.image_hub.recv_jpg()
        
        delay = (time.time() - float(depth_header))
        print(f'images arrived with delay of {delay}')
        self.delays.append(delay)
        # print(f'mean delay: {np.mean(self.delays)}')
        # print(f'std dev: {np.std(self.delays)}')
        self.image_hub.send_reply(b'OK')

        depth = cv2.imdecode(np.frombuffer(depth_buffer, dtype='uint8'), -1)
        # self.image_hub.send_reply(b'OK')
        # print(f'receiving images took {(time.time() - start)*1000}')
        # cv2.imshow(color_header, color)
        # cv2.imshow(depth_header, depth)
        # cv2.waitKey(1)
        return color, depth


    def close(self):
        self.logger.records = np.asarray(self.delays)
        self.logger.export_single_col_csv(f'logs/cam_delays_{RECORD_COUNTER}.csv')
        print('delays exported')


if __name__=='__main__':
    receiver = VideoReceiver()
    while True:
        # try:
        receiver.recv_frames()
            # print('received frames')
        # except KeyboardInterrupt as e:
        #     receiver.close()
        #     break