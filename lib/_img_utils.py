# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import cv2
import torch

import random
import numpy as np
import torchvision.transforms as transforms
from skimage.util.shape import view_as_windows


def split_into_chunks_pose(img_names, seqlen, stride, is_train=True, match_vibe=True):
    video_start_end_indices = []
    vid_names = []
    for item in img_names:
        vid_name = item[:-11]
        vid_names.append(vid_name)
    vid_names = np.array(vid_names)
    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        if stride != seqlen:
            if match_vibe:
                vibe_chunks = view_as_windows(indexes, (16,), step=16)
                for j in range(1,len(start_finish)+1):
                    if start_finish[-j][-1] == vibe_chunks[-1][-1]:
                        if j != 1:
                            start_finish = start_finish[:-j+1]
                        break
                    
        video_start_end_indices += start_finish
    video_start_end_indices = np.array(video_start_end_indices)
    return video_start_end_indices

def split_into_chunks_mesh(img_names, seqlen, stride, pose_params, is_train=True, match_vibe=True):
    video_start_end_indices = []
    vid_names = []
    for item in img_names:
        vid_name = item[:-11]
        vid_names.append(vid_name)
    vid_names = np.array(vid_names)
    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks_ori = view_as_windows(indexes, (seqlen,), step=stride)
        chunks_ori = chunks_ori.tolist()
        chunks = np.array([x for x in chunks_ori if len(pose_params[x[seqlen//2]]) != 1])
        if len(chunks) > 0:
            start_finish = chunks[:, (0, -1)].tolist()
        else:
            continue
        if stride != seqlen:
            if match_vibe:
                vibe_chunks = view_as_windows(indexes, (16,), step=16)
                for j in range(1,len(start_finish)+1):
                    if start_finish[-j][-1] == vibe_chunks[-1][-1]:
                        if j != 1:
                            start_finish = start_finish[:-j+1]
                        break

        video_start_end_indices += start_finish
    video_start_end_indices = np.array(video_start_end_indices)
    return video_start_end_indices