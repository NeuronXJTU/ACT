import os
import time
from construct_feature_graph import construct_feature
import torch
import cv2
from tqdm import tqdm
import numpy as np


def getICIARlabel(line):
    if 'b' in line:
        return 0
    elif 'is' in line:
        return 1
    elif 'iv' in line:
        return 2
    elif 'n' in line:
        return 3
    else:
        return 4


def getCamelyon16label(line):
    if 'normal' in line:
        return 0
    elif 'tumor' in line:
        return 1


def getDigestlabel(line):
    if 'neg' in line:
        return 0
    elif 'pos' in line:
        return 1


def getCamelyon17label(line):
    if 'normal' in line:
        return 0
    elif 'tumor' in line:
        return 1


def getCRCabel(line):
    if 'Normal' in line:
        return 0
    elif 'Low-Grade' in line:
        return 1
    elif 'High-Grade' in line:
        return 2


def getlabel(line, data_name):
    if 'camelyon16' in data_name:
        return getCamelyon16label(line)
    elif data_name == 'iciar':
        return getICIARlabel(line)
    elif 'digest' in data_name:
        return getDigestlabel(line)
    elif 'camelyon17' in data_name:
        return getCamelyon17label(line)
    elif 'CRC_dataset' in data_name:
        return getCRCabel(line)
    elif 'CoNSep' in data_name:
        return
    else:
        exit(0)


def construct_patches_feature(data_name):
    """
    construct nuclei feature from patches
    Args:
        data_mask_dir: The input map of mask
        graph_save_dir: Saved folder of running graph data
        used_image_mask: The visualization results of the position center point of the extracted nucleus
    Returns:
    """
    data_name_new = 'CRC_new'
    data_mask_dir = './data/patches_masks/' + data_name
    graph_save_dir = 'graph_save_dir/' + data_name_new
    path = './txts'
    txt_path = os.path.join(path, data_name + '.txt')
    used_image_mask = os.path.join('usedimage_mask', data_name_new)
    images_path = []
    labeled = {}
    if os.path.exists(os.path.join('txts', data_name + '_labeled.txt')):
        for line in open(os.path.join('txts', data_name + '_labeled.txt')).readlines():
            line = line.strip().split('\t')
            labeled[line[0].split('/')[-1].split('.')[0]] = int(line[1].strip())
    if not os.path.exists(used_image_mask):
        os.makedirs(used_image_mask)
    if not os.path.exists(graph_save_dir):
        os.makedirs(graph_save_dir)
    data_images = open(txt_path)
    data_images = data_images.readlines()
    thisindex = 0
    for data_image_name in tqdm(data_images):
        data_image_name = data_image_name.strip()
        images_path.append(data_image_name)
        if (not os.path.exists(data_image_name)):
            continue
        image = cv2.imread(data_image_name)
        mask = cv2.imread(os.path.join(data_mask_dir,
                                       data_image_name.split('/')[-2] + '_' + data_image_name.split('/')[-1].split('.')[
                                           0] + '.png'))
        thisindex = thisindex + 1
        mask = mask[:, :, 0]
        if len(labeled) == 0:
            label = getlabel(data_image_name, data_name)
        else:
            label = labeled[data_image_name.split('/')[-1].split('.')[0]]
        subdata, node_coordinatelist = construct_feature(image, mask, data_image_name, label)
        mask[mask > 0] = 0
        w, h = image.shape[0], image.shape[1]
        if subdata != 0:
            torch.save(subdata, os.path.join(graph_save_dir, data_image_name.split('/')[-1].split('.jpg')[0] + '.pt'))
            for nodexy in node_coordinatelist:
                xb = int(nodexy[0]) - 2 if int(nodexy[0]) - 2 >= 0 else 0
                xe = int(nodexy[0]) + 2 if int(nodexy[0]) + 2 < w else w
                yb = int(nodexy[1]) - 2 if int(nodexy[1]) - 2 >= 0 else 0
                ye = int(nodexy[1]) + 2 if int(nodexy[1]) + 2 < h else h
                mask[xb:xe, yb:ye] = 255
                image[xb:xe, yb:ye, 0] = 0
                image[xb:xe, yb:ye, 1] = 0
                image[xb:xe, yb:ye, 2] = 255
            cv2.imwrite(os.path.join(used_image_mask, data_image_name.split('/')[-1] + '_ori_used.jpg'), image)
            cv2.imwrite(os.path.join(used_image_mask, data_image_name.split('/')[-1] + '_mask_used.jpg'), mask)


def get_txt(CoNSep_path_dir):
    f = open(os.path.join('./txts', 'pt_CRC_new_test.txt'), 'w')
    f1 = open(os.path.join('./txts', 'pt_CRC_new_train.txt'), 'w')
    data = []
    for line in os.listdir(CoNSep_path_dir):
        data.append(os.path.join(CoNSep_path_dir, line))
    import random
    random.shuffle(data)
    for i, line in enumerate(data):
        if i < 0.8 * len(data):
            f.write(line + '\n')
        else:
            f1.write(line + '\n')
    f.close()
    f1.close()


if __name__ == '__main__':
    data_name = 'CRC_datasets'
    construct_patches_feature(data_name)
