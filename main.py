from tracemalloc import start
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import time

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from realsense import RSCamera
import zmq
import utils
from logger import Logger
from frame_transformations import transform_frame_EulerXYZ
import detection_msg_pb2

SHOW_WINDOW_VIS = True
SEND_OUTPUT = True
TARGET_OBJECT = 'person'

cam = RSCamera()

output = cv2.VideoWriter(utils.VIDEO_FILE, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (cam.width, cam.height))

output_depth = cv2.VideoWriter(utils.VIDEO_DEPTH_FILE, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (cam.width, cam.height))

output_raw = cv2.VideoWriter(utils.VIDEO_RAW_FILE, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (cam.width, cam.height))

logger = Logger()
records = np.empty((0, logger.cols))


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
v = VideoVisualizer(metadata)
class_catalog = metadata.thing_classes

context = zmq.Context()
socket = context.socket(zmq.REP)

if SEND_OUTPUT:
    socket.connect('tcp://localhost:5555')

starting_time = time.time()
frame_counter = 0
elapsed_time = 0
serial_msg = None
quad_pose = None

while True:
    try:
        if SEND_OUTPUT:
            quad_pose_serial = socket.recv()
            quad_pose = detection_msg_pb2.Detection()
            quad_pose.ParseFromString(quad_pose_serial)
            print(quad_pose)

        frame, depth_frame = cam.get_rs_color_aligned_frames()
        cam_intrinsics = frame.profile.as_video_stream_profile().intrinsics
        
        depth_colormap = cam.colorize_frame(depth_frame)

        frame = np.asarray(frame.get_data())
        depth_frame = np.asanyarray(depth_frame.get_data())

        output_depth.write(depth_colormap)
        output_raw.write(frame)

        outputs = predictor(frame)
        detected_class_idxs = outputs['instances'].pred_classes
        pred_boxes = outputs['instances'].pred_boxes
        
        out = v.draw_instance_predictions(frame, outputs['instances'].to('cpu'))

        # v = Visualizer(frame[:, :, ::-1], metadata=metadata, scale=1.2)
        # out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        
        vis_frame = np.asarray(out.get_image()[:, :, ::-1])

        mask_array = outputs['instances'].pred_masks.to('cpu').numpy()
        num_instances = mask_array.shape[0]

        mask_array = np.moveaxis(mask_array, 0, -1)
        mask_array_instance = []
        
        for i in range(num_instances):
            class_idx = detected_class_idxs[i]
            bbox = pred_boxes[i].to('cpu')
            class_name = class_catalog[class_idx]
            [(xmin, ymin, xmax, ymax)] = bbox
            # cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

            center_x = int((xmax - xmin)/2 + xmin)
            center_y = int((ymax - ymin)/2 + ymin)

            if center_y < cam.height and center_x < cam.width:
                        depth = depth_frame[int(center_y), int(
                            center_x)].astype(float)
                        distance = depth * cam.depth_scale
            else:
                # No valid distance found
                distance = 0.0


            # Get translation vector relative to the camera frame
            tvec = cam.deproject(cam_intrinsics, center_x, center_y, distance)

            mask_array_instance.append(mask_array[:, :, i:(i+1)])
            obj_mask = np.zeros_like(frame)
            obj_mask = np.where(mask_array_instance[i] == True, 255, obj_mask)
            cv2.imwrite(f'pictures/mask_{class_name}.png',obj_mask)

            
            if class_name == TARGET_OBJECT:
                msg = detection_msg_pb2.Detection()
                # Align camera frame with standard motion capture frame
                if SEND_OUTPUT:
                    camera_point = [tvec[2], tvec[0], tvec[1]]
                    translation = [quad_pose.x(), quad_pose.y(), quad_pose.z()]
                    rotation = [quad_pose.roll(), quad_pose.pitch(), quad_pose.yaw()]
                    tvec = transform_frame_EulerXYZ(rotation, translation, camera_point) 
                msg.x = tvec[0]
                msg.y = tvec[1]
                msg.z = tvec[2]
                msg.label = class_name
                msg.confidence = 0
                serial_msg = msg.SerializeToString()

                # Log values
                elapsed_time = time.time() - starting_time
                logger.record_value([np.array(
                        [tvec[0], tvec[1], tvec[2], elapsed_time, 0, class_name]), ])
                
        if SHOW_WINDOW_VIS:
            cv2.imshow('output', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if SEND_OUTPUT:
            if serial_msg is not None:
                socket.send(serial_msg)
            else:
                msg = detection_msg_pb2.Detection()
                msg.x = 0.0
                msg.y = 0.0
                msg.z = 0.0
                msg.label = 'Nothing'
                msg.confidence = 0.0
                serial_msg = msg.SerializeToString()
                socket.send(serial_msg)
        
        output.write(vis_frame)
        frame_counter += 1


    except KeyboardInterrupt as e:
        output.release()
        output_depth.release()
        output_raw.release()
        cam.release()
        cv2.destroyAllWindows()
        socket.close()
        logger.export_to_csv(utils.LOG_FILE)

