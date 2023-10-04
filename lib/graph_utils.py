# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Graph utilities
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch


def build_adj(joint_num, skeleton, flip_pairs):
    adj_matrix = np.zeros((joint_num, joint_num))
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1
    for lr in flip_pairs:
        adj_matrix[lr] = 1
        adj_matrix[lr[1], lr[0]] = 1

    return adj_matrix + np.eye(joint_num)

def build_verts_joints_relation(joints, vertices):
    '''
    get the nearest joints of every vertex
    '''
    vertix_num = vertices.shape[0]
    joints_num = joints.shape[0]
    nearest_relation = np.zeros((vertix_num))
    jv_sets = {}
    for (idx, v) in enumerate(vertices):
        nst_joint = v - joints
        nst_joint = nst_joint ** 2
        nst_joint = nst_joint.sum(1)
        nst_joint = np.argmin(nst_joint)
        nearest_relation[idx] = nst_joint
        if nst_joint not in jv_sets:
            jv_sets[nst_joint] = [idx]
        else:
            jv_sets[nst_joint].append(idx)
            
    return nearest_relation, jv_sets

def build_verts_joints_relation_and_adj(joints, vertices):
    '''
    get the nearest joints of every vertex
    vertices first, then joints
    '''
    vertix_num = vertices.shape[0]
    joints_num = joints.shape[0]
    adj_matrix = np.zeros((joints_num + vertix_num, joints_num + vertix_num))
    nearest_relation = np.zeros((vertix_num))
    jv_sets = {}
    for (idx, v) in enumerate(vertices):
        nst_joint = v - joints
        nst_joint = nst_joint ** 2
        nst_joint = nst_joint.sum(1)
        nst_joint = np.argmin(nst_joint)
        nearest_relation[idx] = nst_joint
        adj_matrix[idx, nst_joint + vertix_num] = 1
        adj_matrix[nst_joint + vertix_num, idx] = 1
        if nst_joint not in jv_sets:
            jv_sets[nst_joint] = [idx]
        else:
            jv_sets[nst_joint].append(idx)
            
    return nearest_relation, jv_sets, adj_matrix

def build_verts_joints_group_adj(group_num, joints_num, vertix_num, groups_joint, groups_verts):
    adj_matrix = np.zeros((joints_num + vertix_num, joints_num + vertix_num))
    for i in range(group_num):
        for joint in groups_joint[i]:
            for vert in groups_verts[i]:
                adj_matrix[vert, joint + vertix_num] = 1
                adj_matrix[joint + vertix_num, vert] = 1
    
    return adj_matrix


def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))

    return L


class my_sparse_mm(torch.autograd.Function):
    """
    this function is forked from https://github.com/xbresson/spectral_graph_convnets
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        print("CHECK sparse W: ", W.is_cuda)
        print("CHECK sparse x: ", x.is_cuda)
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx