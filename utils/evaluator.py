#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from models.resnet_fpn import fpn
from utils.metrics import ConfusionMatrix
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
import copy
from operator import itemgetter
import PIL
from torch import topk

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

transformer = transforms.Compose([
    transforms.ToTensor(),
])

features_blobs = [None]
def hook_feature(module, input, output):
    "hook works"
    features_blobs[0] = output.data.cpu().numpy()

def return_CAM(feature_conv, weight, class_idx):
    """
    return_CAM generates the CAMs and up-sample it to 224x224
    arguments:
    feature_conv: the feature maps of the last convolutional layer
    weight: the weights that have been extracted from the trained parameters
    class_idx: the label of the class which has the highest probability
    """
    size_upsample = (224, 224)
    
    # we only consider one input image at a time, therefore in the case of 
    # fpn, the shape is (1, 512, 56, 56)
    bz, nc, h, w = feature_conv.shape 
    output_cam = []
    for idx in class_idx:
        cam = np.matmul(weight[idx], feature_conv.reshape((nc, h*w))) # -> (1, 512) x (512, 49) = (1, 49)
        cam = cam.reshape(h, w) # -> (7 ,7)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        output_cam.append(cv2.resize(cam, size_upsample))
    return output_cam

def CAM_on_image(ori_image, CAMs, width, height, class_idx, mode, save_name):
    ori_image = cv2.cvtColor(np.array(ori_image), cv2.COLOR_RGB2BGR)
    ori_image = cv2.resize(ori_image,(width, height))
    results = [None] * len(class_idx)
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.2 + ori_image * 0.8
        if mode == 1:
            # put class label text on the result
            cv2.putText(result, str(int(class_idx[i])), (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(f"heatmaps/globalCAM_{save_name}_class{class_idx[i]}.png", result)
        results[int(class_idx[i])] = result
    return results
    
def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resized = list(images)
    for i in range(len(images)):
        if label:
            resized[i] = images[i].resize(shape, Image.NEAREST)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized

def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(int(m))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().cuda()

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs

def global2patch(images, labels_glb, p_size, ids):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''
    patches = []; coordinates = []; templates = []; sizes = []; ratios = [(0, 0)] * len(ids); patch_ones = np.ones(p_size); label_patches = []; patch_names = []
    for i in range(len(ids)):
        image_info = os.listdir(os.path.join('data/locals', ids[i]))[0].split('_')
        patches_list = os.listdir(os.path.join('data/locals', ids[i]))
        patches_list.sort()
        h, w = int(image_info[-2]), int(image_info[-1].split('.')[0])

        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])

        patches.append([None] * (len(patches_list)))
        label_patches.append([None] * (len(patches_list)))
        coordinates.append([(0, 0)] * (len(patches_list)))
        patch_names.append([None] * (len(patches_list)))

        for j, patch_name in enumerate(patches_list):
            x, y = int(patch_name.split('_')[-4]), int(patch_name.split('_')[-3])
            patch = Image.open(os.path.join('data/locals', ids[i], patch_name))
            coordinates[i][j] = (x, y)
            patches[i][j] = transforms.functional.crop(patch, 0, 0, p_size[0], p_size[1])
            label_patches[i][j] = labels_glb[i]
            patch_names[i][j] = patch_name

    return patches, coordinates, sizes, ratios, label_patches, patch_names

class Evaluator(object):
    def __init__(self, n_class, size_g, size_p, sub_batch_size=6, mode=1, visualization=False, test=False, task_name=None):
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_g = size_g
        self.size_p = size_p
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.test = test
        self.visualization = visualization
        self.task_name = task_name

        if test:
            self.flip_range = [False, True]
            self.rotate_range = [0, 1, 2, 3]
        else:
            self.flip_range = [False]
            self.rotate_range = [0]
    
    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def eval_test(self, sample, model, global_fixed):
        with torch.no_grad():
            images = sample['image']
            ids = sample['id']
            if not self.test:
                labels = sample['label'] # PIL images
                labels_glb = masks_transform(labels)

            width, height = images[0].size

            if width > self.size_g[0] or height > self.size_g[1]:
                images_global = resize(images, self.size_g) # list of resized PIL images
            else:
                images_global = list(images)

            if self.mode == 2 or self.mode == 3:
                images_local = [ image.copy() for image in images ]
                scores_local = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]
                scores = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]

            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images_global[b] = transforms.functional.rotate(images_global[b], 90) # rotate back!
                        images_global[b] = transforms.functional.hflip(images_global[b])
                        if self.mode == 2 or self.mode == 3:
                            images_local[b] = transforms.functional.rotate(images_local[b], 90) # rotate back!
                            images_local[b] = transforms.functional.hflip(images_local[b])
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images_global[b] = transforms.functional.rotate(images_global[b], 90)
                            if self.mode == 2 or self.mode == 3:
                                images_local[b] = transforms.functional.rotate(images_local[b], 90)

                    # prepare global images onto cuda
                    images_glb = images_transform(images_global) # b, c, h, w

                    if self.mode == 2 or self.mode == 3:
                        patches, coordinates, sizes, ratios, label_patches, patch_names = global2patch(images, labels_glb, self.size_p, ids)
                        predicted_patches = np.zeros((len(images), 4))
                        predicted_ensembles = np.zeros((len(images), 4))
                        outputs_global = [ None for i in range(len(images)) ]
                    if self.mode == 1:
                        # eval with only resized global image ##########################
                        model.module.fpn_global.smooth.register_forward_hook(hook_feature)
                        print(features_blobs)
                        exit()
                        if flip:
                            outputs_global += np.flip(np.rot90(model.forward(images_glb, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global, _ = model.forward(images_glb, None, None, None)
                        ################################################################

                    if self.mode == 2:
                        # eval with patches ###########################################
                        # get the softmax weight
                        params = list(model.parameters())
                        weight_softmax = np.squeeze(params[-8].cpu().numpy())
                        model.module.fpn_local.smooth.register_forward_hook(hook_feature)
                        for i in range(len(images)):
                            j = 0
                            if self.visualization:
                                bbox_min_w = min(coordinates[i],key=itemgetter(0))[0]
                                bbox_max_w = max(coordinates[i],key=itemgetter(0))[0]
                                bbox_min_h = min(coordinates[i],key=itemgetter(1))[1]
                                bbox_max_h = max(coordinates[i],key=itemgetter(1))[1]
                                blank_images = [np.zeros((int((bbox_max_w-bbox_min_w+1)*self.size_p[1]/1),int((bbox_max_h-bbox_min_h+1)*self.size_p[0]/1),3), np.uint8) for k in range(self.n_class) ] # rgb image (H, W, 3)                            

                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w                                
                                output_ensembles, output_global, output_patches, _ = model.forward(images_glb[i:i+1], patches_var, coordinates[i][j : j+self.sub_batch_size], ratios[i], mode=self.mode, n_patch=len(coordinates[i]))
                                if self.visualization:
                                    index = 0
                                    for output in output_ensembles:
                                        probs = F.softmax(output).cpu().squeeze()
                                        # get the class indices of top k probabilities
                                        class_idx = topk(probs, len(probs))[1].int()
                                        CAMs = return_CAM(features_blobs[0][[index],:], weight_softmax, class_idx)
                                        masks = CAM_on_image(patches[i][j+index], CAMs, int(self.size_p[0]/1), int(self.size_p[1]/1), class_idx, self.mode, patch_names[i][j+index])
                                        for ind, bi in enumerate(blank_images):
                                            bi[int((coordinates[i][j+index][0]-bbox_min_w)*self.size_p[1]/1):int((coordinates[i][j+index][0]-bbox_min_w)*self.size_p[1]/1)+int(self.size_p[1]/1), int((coordinates[i][j+index][1]-bbox_min_h)*self.size_p[0]/1):int((coordinates[i][j+index][1]-bbox_min_h)*self.size_p[0]/1)+int(self.size_p[0]/1)] = masks[ind]
                                        index += 1
                                    
                                                              
                                # patch predictions          
                                for pred_patch, pred_ensemble, index in zip(torch.max(output_patches.data, 1)[1].data, torch.max(output_ensembles.data, 1)[1].data, range(len(torch.max(output_patches.data, 1)[1].data))):
                                    predicted_patches[i][int(pred_patch)] += 1
                                    predicted_ensembles[i][int(pred_ensemble)] += 1
                                   
                                j += patches_var.size()[0]
                            outputs_global[i] = output_global
                            if self.visualization:
                                for ind, bi in enumerate(blank_images):
                                    cv2.putText(bi, str(ind), (200, 400), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2)
                                    cv2.imwrite(f"heatmaps/localCAM_{self.task_name}_{ids[i]}_class{ind}.png", bi)

                        outputs_global = torch.cat(outputs_global, dim=0)
                        ###############################################################

                    if self.mode == 3:
                        # eval global with help from patches ##################################################
                        # go through local patches to collect feature maps
                        # collect predictions from patches
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                _, output_patches = model.module.collect_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, n_patch_all=len(coordinates[i]))

                                for pred_patch in torch.max(output_patches.data, 1)[1].data:
                                    predicted_patches[i][int(pred_patch)] += 1
                            
                                j += self.sub_batch_size
                        # go through global image

                        tmp, fm_global = model.forward(images_glb, None, None, None, mode=self.mode)

                        if flip:
                            outputs_global += np.flip(np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global = tmp
                        # generate ensembles
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                fl = model.module.generate_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, n_patch_all=len(coordinates[i]))
                                fg = model.module._crop_global(fm_global[i:i+1], coordinates[i][j:j+self.sub_batch_size], ratios[i])[0]
                                fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                                output_ensembles = model.module.ensemble(fl, fg) # include cordinates

                                # ensemble predictions
                                for pred_ensemble in torch.max(output_ensembles.data, 1)[1].data:
                                    predicted_ensembles[i][int(pred_ensemble)] += 1

                                j += self.sub_batch_size
                        ###################################################

            _, predictions_global = torch.max(outputs_global.data, 1)

            if not self.test:
                self.metrics_global.update(labels_glb, predictions_global)

            if self.mode == 2 or self.mode == 3:
                # patch predictions ###########################
                predictions_local = predicted_patches.argmax(1)
                if not self.test:
                    self.metrics_local.update(labels_glb, predictions_local)
                ###################################################
                predictions = predicted_ensembles.argmax(1)
                if not self.test:
                    self.metrics.update(labels_glb, predictions)
                return predictions, predictions_global, predictions_local, outputs_global.data.cpu().numpy()
            else:
                return None, predictions_global, None, None
