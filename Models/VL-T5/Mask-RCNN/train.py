import json
#import ijson
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
import os
import argparse


train_data_dict = None
valid_data_dict = None
train_images = None
valid_images = None
def train_dataset_function():
  lst = []
  global train_data_dict
  global train_images
  with open(train_data_dict, 'rb') as fp:
    train_dict = pickle.load(fp)
  for image_num, data in train_dict.items():
    img_dict = {}
    bboxes = data['bboxes']
    masks = data['masks']
    img = cv2.imread(train_images+str(image_num)+".png")
    height, width,_ = img.shape
    img_dict['file_name'] = train_images+str(image_num)+".png"
    img_dict['width'] = width
    img_dict['height'] = height
    img_dict['image_id'] = image_num
    img_dict['annotations'] = []
    for i, bbox in enumerate(bboxes):
      annot_dict = {}
      cls, (x0, y0, x1, y1) = bbox
      mask = masks[i]
      mask = [float(a) for a in mask]
      annot_dict['bbox'] = [float(x0),float(y0),float(x1),float(y1)]
      annot_dict['bbox_mode'] = 0
      annot_dict['category_id'] = int(cls)
      annot_dict['segmentation'] = [mask]
      img_dict['annotations'].append(annot_dict)
    # Skip the charts with so many objects
    if len(img_dict['annotations']) > 60:
        continue
    lst.append(img_dict)
  return lst

def valid_dataset_function():
  lst = []
  global valid_data_dict
  global valid_images
  with open(valid_data_dict, 'rb') as fp:
      valid_dict = pickle.load(fp)
  for image_num, data in valid_dict.items():
    img_dict = {}
    bboxes = data['bboxes']
    masks = data['masks']
    img = cv2.imread(valid_images+str(image_num)+".png")
    height, width,_ = img.shape
    img_dict['file_name'] = valid_images+str(image_num)+".png"
    img_dict['width'] = width
    img_dict['height'] = height
    img_dict['image_id'] = image_num
    img_dict['annotations'] = []
    for i, bbox in enumerate(bboxes):
      annot_dict = {}
      cls, (x0, y0, x1, y1) = bbox
      mask = masks[i]
      mask = [float(a) for a in mask]
      annot_dict['bbox'] = [float(x0),float(y0),float(x1),float(y1)]
      annot_dict['bbox_mode'] = 0
      annot_dict['category_id'] = int(cls)
      annot_dict['segmentation'] = [mask]
      img_dict['annotations'].append(annot_dict)
    # Skip the charts with so many objects
    if len(img_dict['annotations']) > 60:
        continue
    lst.append(img_dict)
  return lst


def main():
    global train_data_dict
    global valid_data_dict
    global train_images
    global valid_images

    
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Train Mask RCNN')
    parser.add_argument('--train-data-dict', type=str, help='Path to the training data file')
    parser.add_argument('--valid-data-dict', type=str, help='Path to the validation data file')
    parser.add_argument('--train-images', type=str, help='Path to the training images')
    parser.add_argument('--valid-images', type=str, help='Path to the validation images')
    parser.add_argument('--output-dir', type=str, help='Path to the output directory for saving the checkpoints')
    parser.add_argument('--ITERS', type=int,help='Max number of iterations')
    parser.add_argument('--batch-size', type=int, help='Batch Size for the model')

    parser.add_argument('--checkpoint-period', type=int, default=3000, help='Saving checkpoints period')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--num-classes', type=float, default=15, help='Number of classes')

    args = parser.parse_args()




    train_data_dict= args.train_data_dict 
    valid_data_dict = args.valid_data_dict 
    train_images = args.train_images 
    valid_images = args.valid_images 
    output_dir = args.output_dir 
    ITERS = args.ITERS 
    batch_size = args.batch_size 

    # Register dataset.
    DatasetCatalog.register("train_set", train_dataset_function)
    DatasetCatalog.register("valid_set", valid_dataset_function)

    #Train
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_set",)
    cfg.DATASETS.TEST = ("valid_set",)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers #4
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period #3000
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = args.lr #0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = ITERS
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes #15  # number of classes. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
   
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()
