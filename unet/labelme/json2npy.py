
  #!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme
from PIL import Image
import matplotlib.pyplot as plt


##
input_dir = './data'
output_dir = './results'

if not osp.exists(output_dir):
  os.makedirs(output_dir)
  os.makedirs(osp.join(output_dir, "input"))
  os.makedirs(osp.join(output_dir, "label"))
  os.makedirs(osp.join(output_dir, "visual"))
  
##
class_names = []
class_name_to_id = {}

label_path =  './labels.txt'

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
out_class_names_file = osp.join(output_dir, "class_names.txt")
with open(out_class_names_file, "w") as f:
    f.writelines("\n".join(class_names))
print("Saved class_names:", out_class_names_file)

##
def BlacknWhite(cls) :
  for i in range(cls.shape[0]):
    for j in range(cls.shape[1]):
      if cls[i,j] != 0 :
        cls[i,j] = 255
  return cls


##
for filename in glob.glob(osp.join(input_dir, "*.json")): 
  # print("Generating dataset from:", filename)
  label_file = labelme.LabelFile(filename=filename)
  base = osp.splitext(osp.basename(filename))[0]
  print(filename)
  print(base)

  in_cls_file = osp.join(output_dir, "input", "input_"+ base[-6:] +".npy" )
  out_cls_file = osp.join(output_dir, "label", "label_"+ base[-6:] + ".npy")
  out_clsv_file = osp.join(output_dir, "visual", "visual_" + base[-6:] + ".jpg")

  img = labelme.utils.img_data_to_arr(label_file.imageData)

  cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
  
  img = img[:,:,:-1]

  # reshape
  cls_rsh = np.zeros((512,512), dtype=np.uint8)
  cls_rsh[16:-16, :] = cls[:, 64:-64]

  img_rsh = 255*np.ones((512, 512,3), dtype=np.uint8)
  img_rsh[16:-16, :, :] = img[:, 64:-64, :]


  clsv = imgviz.label2rgb(
                cls_rsh,
                imgviz.rgb2gray(img_rsh),
                label_names=class_names,
                font_size=15,
                loc="rb",
            )
  
  ins[cls == -1] = 0  # ignore it.

  # class label
  # labelme.utils.lblsave(out_clsp_file, cls_rsh) # PNG 파일로 저장
  np.save(out_cls_file, BlacknWhite(cls_rsh)) # npy 파일로 저장, 클래스 1개
  imgviz.io.imsave(out_clsv_file, clsv) # visual 결과 jpg 파일로 저장
  np.save(in_cls_file, imgviz.rgb2gray(img_rsh))


  
