import os
import os.path as osp
import numpy as np
import json
import math
import torch
from pycocotools.coco import COCO

from core.config import cfg
from Human36M.noise_stats import error_distribution
from noise_utils import synthesize_pose
from smpl import SMPL
from coord_utils import process_bbox, get_bbox, cam2pixel
from aug_utils import j2d_processing, j3d_processing, flip_2d_joint
import joblib


class MPII(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'MPII'
        self.data_split = 'train'
        self.annot_path = osp.join(cfg.data_dir, dataset_name, 'mpii_data')
        self.img_path = osp.join(self.annot_path, 'mpii')
        self.fitting_thr = 3.0  # following I2L-MeshNet

        # SMPL joint set
        self.mesh_model = SMPL()
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.face_kps_vertex = self.mesh_model.face_kps_vertex
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))

        # h36m skeleton
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.human36_joint_num = 17
        self.human36_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
                'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_error_distribution = self.get_stat()
        self.joint_regressor_h36m = self.mesh_model.joint_regressor_h36m

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_origin_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15), (5, 6), (11, 12))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),  # (5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        self.coco_root_joint_idx = self.coco_joints_name.index('Pelvis')
        self.joint_regressor_coco = self.mesh_model.joint_regressor_coco

        self.input_joint_name = cfg.DATASET.input_joint_set
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)
        self.seqlen = cfg.DATASET.seqlen
        self.stride = cfg.DATASET.stride
        self.img_paths, self.img_hws, \
        self.poses, self.shapes, self.transes, \
        self.cam_param_focals, self.cam_param_princpts, self.features = self.load_data()

    def get_stat(self):
        ordered_stats = []
        for joint in self.human36_joints_name:
            item = list(filter(lambda stat: stat['Joint'] == joint, error_distribution))[0]
            ordered_stats.append(item)

        print("error stat joint num: ", len(ordered_stats))
        return ordered_stats

    def generate_syn_error(self):
        noise = np.zeros((self.human36_joint_num, 2), dtype=np.float32)
        weight = np.zeros(self.human36_joint_num, dtype=np.float32)
        for i, ed in enumerate(self.human36_error_distribution):
            noise[i, 0] = np.random.normal(loc=ed['mean'][0], scale=ed['std'][0])
            noise[i, 1] = np.random.normal(loc=ed['mean'][1], scale=ed['std'][1])
            weight[i] = ed['weight']

        prob = np.random.uniform(low=0.0, high=1.0, size=self.human36_joint_num)
        weight = (weight > prob)
        noise = noise * weight[:, None]

        return noise

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def load_data(self):
        print('Load annotations of COCO')
        db = COCO(osp.join(self.annot_path, 'train.json'))
        with open(osp.join(self.annot_path, 'MPII_train_SMPL_NeuralAnnot.json')) as f:
            smpl_params = json.load(f)
            
        db_file = osp.join(self.annot_path, 'mpii_train_db.pt')
        if osp.isfile(db_file):
            img_db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        
        img_names = img_db['img_name']
        img_feats = img_db['features']
        img_aids = img_db['aid']

        img_paths, img_hws, poses, shapes, transes = [], [], [], [], []
        features, cam_param_focals, cam_param_princpts = [], [], []
        if self.data_split == 'train':
            for idx, aid in enumerate(db.anns.keys()):
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = img['file_name'].split('/')[-1]
                img_path = osp.join(self.img_path, imgname)
                width, height = img['width'], img['height']

                if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue

                # bbox
                bbox = process_bbox(ann['bbox'])
                if bbox is None: 
                    continue
                
                smpl_param = smpl_params[str(aid)]                
                pose_param = np.array(smpl_param['smpl_param']['pose'], dtype=np.float32)
                shape_param = np.array(smpl_param['smpl_param']['shape'], dtype=np.float32)
                trans_param = np.array(smpl_param['smpl_param']['trans'], dtype=np.float32)
                cam_param_focal = np.array(smpl_param['cam_param']['focal'], dtype=np.float32)
                cam_param_princpt = np.array(smpl_param['cam_param']['princpt'], dtype=np.float32)
                
                feature = np.array(img_feats[idx], dtype=np.float32)
                assert img_aids[idx] == aid, f"check: {img_aids[idx]} / {aid}"
                
                img_paths.append(img_path)
                img_hws.append(np.array((height, width), dtype=np.int32))
                poses.append(pose_param)
                shapes.append(shape_param)
                transes.append(trans_param)
                cam_param_focals.append(cam_param_focal)
                cam_param_princpts.append(cam_param_princpt)
                features.append(feature)

            img_paths, img_hws = np.array(img_paths), np.array(img_hws)
            poses, shapes, transes = np.array(poses), np.array(shapes), np.array(transes)
            cam_param_focals, cam_param_princpts, features = np.array(cam_param_focals), np.array(cam_param_princpts), np.array(features)
            
            return img_paths, img_hws, poses, shapes, transes, cam_param_focals, cam_param_princpts, features

    def get_smpl_coord(self, smpl_param):
        pose, shape, trans, gender = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param['gender']
        smpl_pose = torch.FloatTensor(pose).view(-1, 3);
        smpl_shape = torch.FloatTensor(shape).view(1, -1);
        # translation vector from smpl coordinate to 3dpw world coordinate
        smpl_trans = torch.FloatTensor(trans).view(-1, 3)
        smpl_pose = smpl_pose.view(1, -1)
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer[gender](smpl_pose, smpl_shape, smpl_trans)
        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)
        # meter -> milimeter
        smpl_mesh_coord *= 1000;
        smpl_joint_coord *= 1000;
        return smpl_mesh_coord, smpl_joint_coord

    def add_pelvis_and_neck(self, joint_coord):
        lhip_idx = self.coco_joints_name.index('L_Hip')
        rhip_idx = self.coco_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = self.coco_joints_name.index('L_Shoulder')
        rshoulder_idx = self.coco_joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1, -1))

        joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord

    def get_joints_from_mesh(self, mesh, joint_set_name, cam_param):
        joint_coord_cam = None
        if joint_set_name == 'human36':
            joint_coord_cam = np.dot(self.joint_regressor_h36m, mesh)
        elif joint_set_name == 'coco':
            joint_coord_cam = np.dot(self.joint_regressor_coco, mesh)
            joint_coord_cam = self.add_pelvis_and_neck(joint_coord_cam)
        # projection
        f, c = np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'], dtype=np.float32)
        joint_coord_img = cam2pixel(joint_coord_cam, f, c)

        joint_coord_img[:, 2] = 1
        return joint_coord_cam, joint_coord_img

    def get_fitting_error(self, bbox, coco_from_dataset, coco_from_smpl, coco_joint_valid):
        bbox = process_bbox(bbox.copy(), aspect_ratio=1.0)
        coco_from_smpl_xy1 = np.concatenate((coco_from_smpl[:,:2], np.ones_like(coco_from_smpl[:,0:1])),1)
        coco_from_smpl, _ = j2d_processing(coco_from_smpl_xy1, (64, 64), bbox, 0, 0, None)
        coco_from_dataset_xy1 = np.concatenate((coco_from_dataset[:,:2], np.ones_like(coco_from_smpl[:,0:1])),1)
        coco_from_dataset, trans = j2d_processing(coco_from_dataset_xy1, (64, 64), bbox, 0, 0, None)

        # mask joint coordinates
        coco_joint = coco_from_dataset[:,:2][np.tile(coco_joint_valid,(1,2))==1].reshape(-1,2)
        coco_from_smpl = coco_from_smpl[:,:2][np.tile(coco_joint_valid,(1,2))==1].reshape(-1,2)

        error = np.sqrt(np.sum((coco_joint - coco_from_smpl)**2,1)).mean()
        return error

    def __len__(self):
        return len(self.img_paths)
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def __getitem__(self, idx):
        flip, rot = 0, 0

        smpl_param = {'pose': self.poses[idx], 'shape': self.shapes[idx], 'trans': self.transes[idx], 'gender': 'neutral'}
        cam_param = {'focal': self.cam_param_focals[idx], 'princpt': self.cam_param_princpts[idx]}
        img_path = str(self.img_paths[idx])
        img_shape = self.img_hws[idx]
        img_feat = self.features[idx].copy()
        
        # regress h36m, coco joints
        mesh_cam, joint_cam_smpl = self.get_smpl_coord(smpl_param)
        joint_cam_h36m, joint_img_h36m = self.get_joints_from_mesh(mesh_cam, 'human36', cam_param)
        joint_cam_coco, joint_img_coco = self.get_joints_from_mesh(mesh_cam, 'coco', cam_param)

        # root relative camera coordinate
        mesh_cam = mesh_cam - joint_cam_h36m[:1]
        root_coord = joint_cam_h36m[:1].copy()
        joint_cam_coco = joint_cam_coco - root_coord
        joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

        if self.input_joint_name == 'coco':
            joint_img, joint_cam = joint_img_coco, joint_cam_coco
        elif self.input_joint_name == 'human36':
            joint_img, joint_cam = joint_img_h36m, joint_cam_h36m

        # make new bbox
        tight_bbox = get_bbox(joint_img)
        bbox = process_bbox(tight_bbox.copy())
        if not cfg.DATASET.use_gt_input:
            joint_img = self.replace_joint_img(joint_img, tight_bbox)
        if flip:
            joint_img = flip_2d_joint(joint_img, cfg.MODEL.input_shape[1], self.flip_pairs)
        joint_img = joint_img[:,:2]
        joint_img = self.normalize_screen_coordinates(joint_img, w=img_shape[1], h=img_shape[0])
        joint_img = np.array(joint_img, dtype=np.float32)
        joint_cam = j3d_processing(joint_cam, rot, flip, self.flip_pairs)
        
        joint_img = joint_img.reshape(1, len(joint_img), 2).repeat(self.seqlen, axis=0)
        img_feat = img_feat.reshape(1, 2048).repeat(self.seqlen, axis=0)

        if cfg.MODEL.name == 'PMCE':
            # default valid
            mesh_valid = np.ones((len(mesh_cam), 1), dtype=np.float32)
            reg_joint_valid = np.ones((len(joint_cam_h36m), 1), dtype=np.float32)
            lift_joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)

            inputs = {'pose2d': joint_img, 'img_feature': img_feat}
            targets = {'mesh': mesh_cam / 1000, 'lift_pose3d': joint_cam, 'reg_pose3d': joint_cam_h36m}
            meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

            return inputs, targets, meta

        elif cfg.MODEL.name == 'PoseEst':
            # default valid
            joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)

            return joint_img, joint_cam, joint_valid, img_feat

    def replace_joint_img(self, joint_img, bbox):
        if self.input_joint_name == 'coco':
            joint_img_coco = joint_img
            if self.data_split == 'train':
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                pt1 = np.array([xmin, ymin])
                pt2 = np.array([xmax, ymin])
                pt3 = np.array([xmax, ymax])
                area = math.sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2)) * math.sqrt(
                    pow(pt3[0] - pt2[0], 2) + pow(pt3[1] - pt2[1], 2))
                joint_img_coco[:17, :] = synthesize_pose(joint_img_coco[:17, :], area, num_overlap=0)
                return joint_img_coco

        elif self.input_joint_name == 'human36':
            joint_img_h36m = joint_img
            if self.data_split == 'train':
                joint_syn_error = (self.generate_syn_error() / 256) * np.array([cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]], dtype=np.float32)
                joint_img_h36m = joint_img_h36m[:, :2] + joint_syn_error
                return joint_img_h36m