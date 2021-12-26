from __future__ import print_function

from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
import os
import os.path as osp

import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode, Visualizer
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader


import argparse
import glob
import sys
import imgviz

import labelme
from PIL import Image

def get_HFradar_dicts(directory):
    classes = ['left first order region', 'right first order region']  # ['Left Sea Clutter', 'Right Sea Clutter'] 
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

base_dir = '/home/set-spica/Desktop/test_jh/jihye'

for d in ["train", "test"]:
    DatasetCatalog.register("HFradar_" + d, lambda d=d: get_HFradar_dicts(osp.join(base_dir, "HFradar Segmentation/old_sample/" + d)))
    MetadataCatalog.get("HFradar_" + d).set(thing_classes=['left first order region', 'right first order region']) # ['Left Sea Clutter', 'Right Sea Clutter']
HFradar_metadata = MetadataCatalog.get("HFradar_train")

#print(HFradar_metadata.thing_classes)
cfg     = get_cfg()
cfg.merge_from_file(osp.join(base_dir, "output/old_outputdata/config.yaml")) #model_zoo.get_config_file("/home/set-spica/detectron2/jihye/output/config.yaml")) #COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #'/home/set-spica/detectron2/jihye/output/instances_predictions.pth'#
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TRAIN = ("HFradar_train",)
cfg.DATASETS.TEST = ("HFradar_test", )
predictor = DefaultPredictor(cfg)

WINDOW_NAME = "HFR detections"
dataset_dicts = get_HFradar_dicts(osp.join(base_dir, 'HFradar Segmentation/old_sample/test'))
save_path 	= osp.join(base_dir, 'HFradar Segmentation/old_sample/test/')
data_list 	= os.listdir(save_path)

npy_path = osp.join(base_dir, 'npy')
if not os.path.exists(npy_path) :
    os.mkdir(npy_path)


for d in data_list:   
    tmp  = d.split('.')[-1]
    if tmp =='png': 
        print('aaaaaaaaaa', d)
        im = cv2.imread(save_path+d)
        outputs = predictor(im)

        if outputs['instances'].to("cpu").has("pred_masks"):
            masks = np.asarray(outputs['instances'].to("cpu").pred_masks)
            np.save(os.path.join(npy_path, "output_"+d[-10:-4]+'.npy'), masks)



input_dir = osp.join(base_dir, 'HFradar Segmentation/old_sample/test')

if not osp.exists(npy_path):
  os.makedirs(npy_path)
  
class_names = []
class_name_to_id = {}

label_path =  osp.join(base_dir, 'labels.txt')

for i, line in enumerate(open(label_path).readlines()):
    class_id = i - 1  # starts with -1
    class_name = line.strip()
    class_name_to_id[class_name] = class_id
    if class_id == -1:
        assert class_name == "__ignore__"
        continue
    elif class_id == 0:
        assert class_name == "_background_"
    class_names.append(class_name)
class_names = tuple(class_names)
print("class_names:", class_names)



for filename in glob.glob(osp.join(input_dir, "*.json")): 
  # print("Generating dataset from:", filename)
  label_file = labelme.LabelFile(filename=filename)
  base = osp.splitext(osp.basename(filename))[0]
  print(base)

  out_cls_file = osp.join(npy_path, "label_"+ base[-6:] + ".npy")

  img = labelme.utils.img_data_to_arr(label_file.imageData)

  cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
  
  ins[cls == -1] = 0  # ignore it.
  np.save(out_cls_file, cls) # npy 파일로 저장, 클래스 1개
  
  img_png = Image.fromarray(img)
  img_png.save(osp.join(npy_path, base+'.png'))

