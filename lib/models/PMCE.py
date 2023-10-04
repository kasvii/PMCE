import torch
import torch.nn as nn
from core.config import cfg as cfg
from models import CoevoDecoder, PoseEstimation


class PMCE(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(PMCE, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = PoseEstimation.get_model(num_joint, embed_dim, depth, pretrained=cfg.MODEL.posenet_pretrained)
        self.pose_mesh_coevo = CoevoDecoder.get_model(num_joint, embed_dim)

    def forward(self, pose2d, img_feat):
        pose3d = self.pose_lifter(pose2d, img_feat)
        pose3d = pose3d.reshape(-1, self.num_joint, 3)
        cam_pose, cam_mesh = self.pose_mesh_coevo(pose3d / 1000, img_feat)

        return cam_mesh, cam_pose, pose3d


def get_model(num_joint, embed_dim, depth):
    model = PMCE(num_joint, embed_dim, depth)

    return model


