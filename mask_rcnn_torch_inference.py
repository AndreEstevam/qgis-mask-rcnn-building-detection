# -*- coding: utf-8 -*-
'''
CHECK ALL DESCRIPTIONS
CLEAN CODE
'''

# Commented out IPython magic to ensure Python compatibility.
import sys
import os
import getpass
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import backbone_utils as BU   # import resnet_fpn_backbone


def get_model_instance(backbone="resnet50+fpn", trainable_layers=0):
    '''
    Instantiate a mask-rcnn model with the given backbone for detecting edification;
    Args:
        - backbone[str]: "resnet50+fpn" or "resnet101+fpn"; hard coded to 
        "resnet50+fpn"
        - trainable_layers[int]: trainable bakcbone layers, being between 
        0 and 5, where 5 means all layers as trainable and 0 only heads;
    Returns:
        - model: model's instance;
    
    Hard coded to instantiate a "non-pretrained" resnet50+fpn backbone model;
    '''
    backbone = "resnet50+fpn"   # HARD CODING

    if backbone=="resnet50+fpn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
                                                                   trainable_backbone_layers=trainable_layers,
                                                                   min_size=300,
                                                                   max_size=300)
        
        # Updating Anchor Generator for RPN
        model.rpn.anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128)), aspect_ratios=((1, 1.5, 2)))

        # replacing predictors(cls+bbox and mask)
        in_features = model.roi_heads.box_predictor.cls_score.in_features               # 1024
        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=2)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels        # 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=256, num_classes=2)    # dim_reduced == n° of hidden layers

    elif backbone=="resnet101+fpn":
        # Updating Anchor Generator for RPN
        anchor_gen = AnchorGenerator(sizes=((8, 16, 32, 64, 128)), aspect_ratios=((1, 1.5, 2)))

        # Defining predictors(cls+bbox and mask)
        box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=2)
        mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=2)

        # Selecting backbone (.features (?))
        backbone = BU.resnet_fpn_backbone('resnet101',
                                          pretrained=True, 
                                          trainable_layers=trainable_layers)
        
        model = MaskRCNN(backbone=backbone, 
                         min_size=300,
                         max_size=300,
                         rpn_anchor_generator=anchor_gen,
                         box_predictor=box_predictor,
                         mask_predictor=mask_predictor)
        
    return model


def get_inference(weights_path, img_np, threshold):
    '''
    Loads the model and image to be processed and outputs result as np arrays.
    Args:
        - weights_path[str]: path to model's weights
        - img_np[numpy.ndarray]: image as a numpy array;
        - threshold[float]: minimum probability for the detection to contain a 
        building;
    Returns:
        - mask as a PIL.Image object and bounding boxes as numpy arrays with
        [x1, y1, x2, y2] format, where 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H;
        
    NEEDS GPU IMPLEMANTATION
    '''
    # Loading trained model
    model = get_model_instance()
    model.load_state_dict(torch.load(weights_path, map_location ='cpu'))
    model.eval()

    # Getting image tensor
    img_tensor = torchvision.transforms.functional.to_tensor(img_np)

    # Forwarding image through model
    output = model([img_tensor.float()])[0]

    # Processing output
    bboxes = output['boxes'][output['scores'] > threshold]
    mask = torch.torch.squeeze(output["masks"][output['scores'] > threshold])
    
    img_tensor = F.convert_image_dtype(img_tensor, dtype=torch.uint8)
    img_tensor_with_bbox = torchvision.utils.draw_bounding_boxes(img_tensor, bboxes, fill=False)
    
    # PIL_bbox = F.to_pil_image(img_tensor_with_bbox)
    # PIL_mask = F.to_pil_image(mask)
    
    return mask, img_tensor_with_bbox