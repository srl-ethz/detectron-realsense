import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from realsense import RSCamera

cam = RSCamera()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
v = VideoVisualizer(metadata)
class_catalog = metadata.thing_classes

while True:
    frame, depth_frame = cam.get_rs_color_aligned_frames()
    frame = np.asarray(frame.get_data())

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

        mask_array_instance.append(mask_array[:, :, i:(i+1)])
        obj_mask = np.zeros_like(frame)
        obj_mask = np.where(mask_array_instance[i] == True, 255, obj_mask)
        cv2.imwrite(f'mask_{class_name}.jpg',obj_mask)
        

    cv2.imshow('output', vis_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()