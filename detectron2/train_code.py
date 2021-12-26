from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.model_zoo import model_zoo
import os
import torch

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
    classes = ['left first order region', 'right first order region'] #['Left Sea Clutter', 'Right Sea Clutter'] #['left first order region', 'right first order region']
    dataset_dicts = []
    image_id 	= 0
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
        record["image_id"] = image_id
        dataset_dicts.append(record)
        image_id += 1
    print('get HFradar dics end============')
    return dataset_dicts

 

for d in ["train", "test"]:
    DatasetCatalog.register("HFradar_" + d, lambda d=d: get_HFradar_dicts('/home/set-spica/detectron2/jihye/HFradar Segmentation/old_sample/' + d))
    MetadataCatalog.get("HFradar_" + d).set(thing_classes=['left first order region', 'right first order region'])#['Left Sea Clutter', 'Right Sea Clutter'])#['left first order region', 'right first order region'])
HFradar_metadata = MetadataCatalog.get("HFradar_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("HFradar_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

print('os makedir ===============')
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)     
trainer.resume_or_load(resume=False)
trainer.train()

#print('aaaaaaaa', cfg.OUTPUT_DIR)
#torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "mymodel.pth"))

print('end train')
#cfg     = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("HFradar_test", )
predictor = DefaultPredictor(cfg)
print('test start')

WINDOW_NAME = "HFR detections"
dataset_dicts = get_HFradar_dicts('/home/set-spica/detectron2/jihye/HFradar Segmentation/old_sample/test')
for d in random.sample(dataset_dicts,3):    
    im = cv2.imread(d["file_name"])
#save_path 	= '/home/set-spica/detectron2/jihye/HFradar Segmentation/test/'
#data_list 	= os.listdir(save_path)
#for d in data_list:   
#    tmp  = d.split('.')[-1]
#    if tmp =='png': 
#    im = cv2.imread(save_path+d)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                           metadata=HFradar_metadata, 
                           scale=0.8, 
                           instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels )
                       )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, v.get_image()[:, :, ::-1])
    if cv2.waitKey(0) == 27:
        break  # esc to quit
        

    
#    plt.figure(figsize = (14, 10))
#    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#    plt.show()

f = open('/home/set-spica/detectron2/jihye/output/config.yaml','w')
f.write(cfg.dump())
f.close()
#evaluator = COCOEvaluator("HFradar_test", cfg, False, output_dir="./output/")
#val_loader = build_detection_test_loader(cfg, "HFradar_test")
#inference_on_dataset(trainer.model, val_loader, evaluator)



