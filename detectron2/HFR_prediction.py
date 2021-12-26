from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
import os

import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode, Visualizer
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

def get_HFradar_dicts(directory):
    classes = ['Left Sea Clutter', 'Right Sea Clutter'] # ['left first order region', 'right first order region'] 
    dataset_dicts = []

    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["height"] = img_anns['imageHeight'] #480
        record["width"] = img_anns['imageWidth'] #640
        
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    print('get HFradar dics end============')
    return dataset_dicts


for d in ["train", "test"]:
    DatasetCatalog.register("HFradar_" + d, lambda d=d: get_HFradar_dicts('/home/set-spica/Desktop/test_jh/jihye/HFradar Segmentation/New_sample/' + d))
    MetadataCatalog.get("HFradar_" + d).set(thing_classes=['Left Sea Clutter', 'Right Sea Clutter']) #['left first order region', 'right first order region']) #

HFradar_metadata = MetadataCatalog.get("HFradar_train")

#print(HFradar_metadata.thing_classes)
cfg     = get_cfg()
cfg.merge_from_file("/home/set-spica/Desktop/test_jh/jihye/output/new_outputdata/config.yaml") #model_zoo.get_config_file("/home/set-spica/detectron2/jihye/output/config.yaml")) #COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #'/home/set-spica/detectron2/jihye/output/instances_predictions.pth'#
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TRAIN = ("HFradar_train",)
cfg.DATASETS.TEST = ("HFradar_test", )
predictor = DefaultPredictor(cfg)

#evaluator = COCOEvaluator("HFradar_train", cfg, False, output_dir="/home/set-spica/detectron2/jihye/output/")
#val_loader = build_detection_test_loader(cfg, "HFradar_train")
#inference_on_dataset(predictor.model, val_loader, evaluator)

#model_path = '/home/set-spica/detectron2/jihye/output/model_final.pth'

#model = build_model(cfg)
#DetectionCheckpointer(model).load(model_path)

WINDOW_NAME = "HFR detections"
dataset_dicts = get_HFradar_dicts('/home/set-spica/Desktop/test_jh/jihye/HFradar Segmentation/New_sample/test')

#for d in random.sample(dataset_dicts,5):    
#    im = cv2.imread(d["file_name"])
save_path 	= '/home/set-spica/Desktop/test_jh/jihye/HFradar Segmentation/New_sample/test/'
data_list 	= os.listdir(save_path)

npy_path = '/home/set-spica/Desktop/test_jh/jihye/npy'
if not os.path.exists(npy_path) :
    os.mkdir(npy_path)


for d in data_list:   
    tmp  = d.split('.')[-1]
    if tmp =='png': 
        print('aaaaaaaaaa', d)
        im = cv2.imread(save_path+d)
        outputs = predictor(im)
        #outputs = model(im)
        v = Visualizer(im[:, :, ::-1],
                   metadata=HFradar_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                   ) 
        print('v inint')
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print('v draw end')
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, v.get_image()[:, :, ::-1])
        if cv2.waitKey(0) == 27:
        	break  # esc to quit
        
#    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#    plt.show()
