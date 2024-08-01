import numpy as np
import os
import sys
import argparse
from skimage.measure import regionprops
from skimage.filters.rank import entropy as Entropy
import skimage
from skimage.morphology import disk, remove_small_objects
from common.nuc_feature import nuc_stats_new, nuc_glcm_stats_new
import multiprocessing
import time
import cv2
import random
from torch_geometric.nn import radius_graph
import torch
from torch_geometric.data import Data
import copy
from common.utils import mkdirs, FarthestSampler, filter_sampled_indice
from graph_sampler import random_sample_graph2

H, W = 512, 512


def euc_dist(arr):
    arr_x = (arr[:, 0, np.newaxis].T - arr[:, 0, np.newaxis]) ** 2
    arr_y = (arr[:, 1, np.newaxis].T - arr[:, 1, np.newaxis]) ** 2
    arr = np.sqrt(arr_x + arr_y)
    arr = arr.astype(np.int16)

    return arr


def _sampling(num_sample, ratio, distance=None, sample_method='farthest'):
    num_subsample = int(num_sample * ratio)

    if sample_method == 'farthest':
        sampler = FarthestSampler()
        indice = sampler(distance, num_subsample)
    elif sample_method == 'fuse':
        # 70% farthest, 30% random
        sampler = FarthestSampler()
        far_num = int(num_subsample)
        rand_num = num_subsample - far_num
        far_indice = sampler(distance, far_num)
        remain_item = filter_sampled_indice(far_indice, num_sample)
        rand_indice = np.asarray(random.sample(remain_item, rand_num))
        indice = np.concatenate((far_indice, rand_indice), 0)
    else:
        # random
        indice = np.random.choice(num_subsample, num_sample, replace=False)

    return indice, num_subsample


def construct_feature(ori_image, mask, image_path, label):
    max_neighbours = 4
    epoch = 1
    graph_sampler = 'knn'
    sample_method = 'fuse'
    H, W = mask.shape
    # input('=<')
    image_path = image_path.split('/')[-1].split('.')[0]
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel)
    # mask = remove_small_objects(mask.astype(np.bool), 8).astype(np.uint8) * 255
    mask = remove_small_objects(mask.astype(np.bool), min_size=4, connectivity=2, in_place=True).astype(np.uint8) * 255
    mask[mask > 0] = 1
    int_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    int_image = cv2.resize(int_image, (W, H), interpolation=cv2.INTER_LINEAR)
    entropy = Entropy(int_image, disk(3))
    mlabel = skimage.measure.label(mask, connectivity=2)
    props = regionprops(mlabel)
    binary_mask = mask.copy()
    binary_mask[binary_mask > 0] = 1
    node_feature = []
    node_coordinate = []
    if len(props) <= 4:
        # print('flag1')
        return 0, 0
    for prop in props:
        bbox = prop.bbox
        cx, cy = prop.centroid
        if prop.area > 20000 or (bbox[2] - bbox[0]) > 300 or (bbox[3] - bbox[1]) > 300 or (
                int_image[int(cx), int(cy)] == 0):  # Remove all black boxes area, 80 * 80 is max area
            continue
        nuc_feats = []
        single_entropy = entropy[bbox[0]: bbox[2] + 1, bbox[1]:bbox[3] + 1]
        single_mask = (mlabel == prop.label) * 1
        single_mask = binary_mask[bbox[0]: bbox[2] + 1, bbox[1]:bbox[3] + 1].astype(np.uint8)
        single_int = int_image[bbox[0]: bbox[2] + 1, bbox[1]:bbox[3] + 1]
        coor = prop.centroid
        mean_im_out, diff, var_im, skew_im = nuc_stats_new(single_mask, single_int)
        glcm_feat = nuc_glcm_stats_new(single_mask, single_int)  # just breakline for better code
        glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = glcm_feat
        mean_ent = cv2.mean(single_entropy, mask=single_mask)[0]
        info = cv2.findContours(single_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = info[0][0]
        num_vertices = len(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            hull_area += 1
        solidity = float(area) / hull_area
        if num_vertices > 4:
            centre, axes, orientation = cv2.fitEllipse(cnt)
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
        else:
            orientation = 0
            majoraxis_length = 1
            minoraxis_length = 1
            continue
        perimeter = cv2.arcLength(cnt, True)
        eccentricity = np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)
        # not cell 
        if minoraxis_length / majoraxis_length < 0.1:
            # input('==>')
            continue
        nuc_feats.append(mean_im_out)
        nuc_feats.append(diff)
        nuc_feats.append(var_im)
        nuc_feats.append(skew_im)
        nuc_feats.append(mean_ent)
        nuc_feats.append(glcm_dissimilarity)
        nuc_feats.append(glcm_homogeneity)
        nuc_feats.append(glcm_energy)
        nuc_feats.append(glcm_ASM)
        nuc_feats.append(eccentricity)
        nuc_feats.append(area)
        nuc_feats.append(majoraxis_length)
        nuc_feats.append(minoraxis_length)
        nuc_feats.append(perimeter)
        nuc_feats.append(solidity)
        nuc_feats.append(orientation)
        feature = np.hstack(nuc_feats)
        node_feature.append(feature)
        node_coordinate.append(coor)
    node_coordinatelist = node_coordinate.copy()
    if (len(node_feature) == 0):
        # print('flag2')
        return 0, 0
    node_feature = np.vstack(node_feature)
    node_coordinate = np.vstack(node_coordinate)
    node_distance = euc_dist(node_coordinate)
    node_feature = node_feature.astype(np.float32)
    node_coordinate = node_coordinate.astype(np.float32)
    # prepare_cv
    nodes_features = np.concatenate((node_feature, node_coordinate), axis=-1)
    # print("nodes_features shape", nodes_features.shape)
    # nodes_features = node_feature
#     print(node_feature)
#     print(np.isnan(node_feature))
#     print(np.isnan(node_feature) == True)
    if (np.isnan(node_feature) == True).all():
        # print('flag3')
        return 0, 0
    nodes_features = torch.from_numpy(nodes_features).to(torch.float)
    node_coordinate = torch.from_numpy(node_coordinate).to(torch.float)
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=nodes_features, pos=node_coordinate, y=y)
    num_nodes = data.x.shape[0]
    num_sample = num_nodes
    subdata = copy.deepcopy(data)
    try:
        # print("label", label)
        # print("num_sample", num_sample)
        rate_cell_num = 1 if num_sample < 4 else 0.85
        choice, num_subsample = _sampling(num_sample, rate_cell_num, node_distance, sample_method)
        for key, item in subdata:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                subdata[key] = item[choice]
            # generate the graph
        if graph_sampler == 'knn':
            edge_index = radius_graph(subdata.x, 180, None, True, max_neighbours, num_workers=4)
        else:
            edge_index = random_sample_graph2(choice, node_distance, 4, True,
                                              n_sample=4, sparse=True)
        subdata.edge_index = edge_index

        return subdata, node_coordinatelist
    except Exception as identifier:
        input('wrong ')
        pass





