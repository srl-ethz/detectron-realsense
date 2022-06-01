import sys

import numpy as np
import cv2
import imagezmq

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

    def recv_frames(self):
        color_header, color_jpg_buffer = self.image_hub.recv_jpg()
        color = cv2.imdecode(np.frombuffer(color_jpg_buffer, dtype='uint8'), -1)
        self.image_hub.send_reply(b'OK')
        
        depth_header, depth_jpg_buffer = self.image_hub.recv_jpg()
        depth = cv2.imdecode(np.frombuffer(depth_jpg_buffer, dtype='uint8'), -1)
        self.image_hub.send_reply(b'OK')

        cv2.imshow(color_header, color)
        cv2.imshow(depth_header, depth)
        cv2.waitKey(1)


if __name__=='__main__':
    receiver = VideoReceiver()
    while True:
        receiver.recv_frames()
        print('received frames')